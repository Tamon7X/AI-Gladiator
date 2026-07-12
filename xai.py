# xai.py
# Explainable AI fuer den trainierten Agenten: "Wohin schaut Rot?"
#
# Zwei Verfahren:
#   1) Vanilla-Gradient-Saliency (Simonyan et al. 2014):
#      |d max_a Q(s,a) / d s| -- welche Eingabepixel beeinflussen die
#      Entscheidung am staerksten? Schnell (1 Backward-Pass pro Frame).
#   2) Occlusion-Analyse (Zeiler & Fergus 2014, optional --occlusion):
#      Ein grauer Patch wird ueber die Beobachtung geschoben; gemessen
#      wird der Einbruch des maximalen Q-Werts. Langsamer, aber
#      modellagnostisch und intuitiv ("wird der Laser verdeckt, sinkt Q").
#
# Ausgabe: PNGs in runs/xai/ -- Beobachtung, Heatmap, Overlay nebeneinander,
# plus gewaehlte Aktion und Q-Wert im Titel.
#
# Nutzung:
#   python xai.py --model models/last_model.pth --auto-aim-red --frames 12
#   python xai.py --model models/last_model.pth --auto-aim-red --occlusion

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arena_env import GladiatorEnv, ACTIONS, N_ACTIONS
from model import CNN_QNet

ACTION_NAMES = []
for dx, dy, shoot in ACTIONS:
    move = {(-1, 0): "links", (1, 0): "rechts", (0, -1): "hoch", (0, 1): "runter",
            (0, 0): "stehen", (-1, -1): "links-hoch", (1, -1): "rechts-hoch",
            (-1, 1): "links-runter", (1, 1): "rechts-runter"}[(dx, dy)]
    ACTION_NAMES.append(move + (" + SCHUSS" if shoot else ""))


def gradient_saliency(model, state_uint8, device):
    """|dQ_max/ds| aggregiert ueber die 4 Stack-Frames -> [60, 80] in [0,1]."""
    s = (torch.as_tensor(state_uint8, device=device)
         .float().div_(255.0).unsqueeze(0)).requires_grad_(True)
    q = model(s)
    action = int(q.argmax(dim=1).item())
    q_max = q[0, action]
    model.zero_grad()
    q_max.backward()
    # Aggregation: Maximum ueber die Frame-Dimension (der neueste Frame
    # dominiert meist, aber Bewegungsinformation steckt in allen vieren)
    sal = s.grad.abs().squeeze(0).max(dim=0).values.cpu().numpy()
    if sal.max() > 0:
        sal = sal / sal.max()
    return sal, action, float(q_max.item())


def occlusion_map(model, state_uint8, device, patch=8, stride=4):
    """Q-Einbruch beim Verdecken: hoher Wert = Region ist entscheidungsrelevant.
    Batch-vektorisiert ueber alle Patch-Positionen."""
    s = (torch.as_tensor(state_uint8, device=device)
         .float().div_(255.0).unsqueeze(0))
    with torch.no_grad():
        q0 = model(s)
        action = int(q0.argmax(dim=1).item())
        base = float(q0[0, action].item())

    H, W = s.shape[2], s.shape[3]
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    heat = np.zeros((len(ys), len(xs)), dtype=np.float32)

    batch, coords = [], []
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            occ = s.clone()
            occ[0, :, y:y + patch, x:x + patch] = 0.5  # neutraler Grauwert
            batch.append(occ)
            coords.append((iy, ix))
    with torch.no_grad():
        for i in range(0, len(batch), 64):
            chunk = torch.cat(batch[i:i + 64], dim=0)
            qs = model(chunk)[:, action]
            for j, qv in enumerate(qs):
                iy, ix = coords[i + j]
                heat[iy, ix] = base - float(qv.item())

    heat = np.maximum(heat, 0)
    if heat.max() > 0:
        heat = heat / heat.max()
    # auf Beobachtungsgroesse hochskalieren
    heat_t = torch.as_tensor(heat)[None, None]
    heat_full = F.interpolate(heat_t, size=(H, W), mode="bilinear",
                              align_corners=False)[0, 0].numpy()
    return heat_full, action, base


def save_figure(obs_frame, heat, action, q_val, method, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].imshow(obs_frame, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Beobachtung (neuester Frame)")
    axes[1].imshow(heat, cmap="inferno")
    axes[1].set_title(f"{method}-Heatmap")
    axes[2].imshow(obs_frame, cmap="gray", vmin=0, vmax=255)
    axes[2].imshow(heat, cmap="inferno", alpha=0.55)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Gewaehlte Aktion: {ACTION_NAMES[action]}   |   "
                 f"Q = {q_val:+.2f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_QNet(output_size=N_ACTIONS).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    os.makedirs(args.out, exist_ok=True)
    env = GladiatorEnv(headless=True, training_mode=True, frame_skip=4,
                       blue_error_rate=args.blue_error,
                       auto_aim_red=args.auto_aim_red, seed=args.seed)

    state = env.reset()
    saved, step = 0, 0
    while saved < args.frames:
        if args.occlusion:
            heat, action, q_val = occlusion_map(model, state, device)
            method = "Occlusion"
        else:
            heat, action, q_val = gradient_saliency(model, state, device)
            method = "Saliency"

        # Nur jede k-te Entscheidung speichern, damit die Bilder eine
        # Episode abdecken statt 12 fast identischer Anfangsframes
        if step % args.every == 0:
            path = os.path.join(args.out, f"xai_{saved:02d}.png")
            save_figure(state[-1], heat, action, q_val, method, path)
            print(f"[{saved + 1}/{args.frames}] {path}  "
                  f"(Aktion: {ACTION_NAMES[action]}, Q={q_val:+.2f})")
            saved += 1

        state, _, done, _ = env.step(action)
        step += 1
        if done:
            state = env.reset()

    print(f"\nFertig: {saved} Analysen in {args.out}/")
    print("Interpretation: Helle Regionen beeinflussen die Q-Entscheidung am")
    print("staerksten. Erwartung bei gutem Modell: Fokus auf Gegner und")
    print("anfliegenden Lasern -- NICHT gleichmaessig oder auf dem Grid.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XAI: Saliency-/Occlusion-Analyse des trainierten Agenten")
    parser.add_argument("--model", type=str, default="models/last_model.pth")
    parser.add_argument("--auto-aim-red", action="store_true")
    parser.add_argument("--blue-error", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=12,
                        help="Anzahl zu speichernder Analysen")
    parser.add_argument("--every", type=int, default=20,
                        help="Jede k-te Entscheidung analysieren")
    parser.add_argument("--occlusion", action="store_true",
                        help="Occlusion-Analyse statt Gradient-Saliency (langsamer)")
    parser.add_argument("--out", type=str, default="runs/xai")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
