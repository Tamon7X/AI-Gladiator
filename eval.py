# eval.py
# Evaluierung: laesst das trainierte Modell (greedy, epsilon=0) N Episoden
# gegen den Skript-Bot spielen und gibt Win-Rate + Statistiken aus.
#
# Nutzung:
#   python eval.py --episodes 100 --headless
#   python eval.py --episodes 10 --render          # zum Zuschauen
#   python eval.py --model models/last_model.pth

import argparse
import random

import numpy as np
import torch

from arena_env import GladiatorEnv, N_ACTIONS
from model import CNN_QNet


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = None
    if args.policy == "model":
        model = CNN_QNet(output_size=N_ACTIONS).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        print(f"[Eval] Modell geladen: {args.model}")
    else:
        # ZUFALLS-BASELINE: misst die Win-Rate einer untrainierten Policy.
        # Referenzwert fuer den Lernnachweis: trainiert vs. random.
        print("[Eval] Zufalls-Policy (Baseline, kein Modell)")

    env = GladiatorEnv(headless=not args.render, training_mode=True,
                       frame_skip=4, blue_error_rate=args.blue_error,
                       auto_aim_red=args.auto_aim_red, seed=args.seed,
                       blue_mode=args.blue_mode)

    results = {"red": 0, "blue": 0, "timeout": 0}
    red_hps, ticks_list = [], []
    dists, shots, hits = [], [], []

    for ep in range(args.episodes):
        state = env.reset()
        done = False
        while not done:
            if model is None:
                action = random.randrange(N_ACTIONS)
            else:
                t = (torch.as_tensor(state, device=device)
                     .float().div_(255.0).unsqueeze(0))
                with torch.no_grad():
                    action = int(model(t).argmax(dim=1).item())
            state, _, done, info = env.step(action)

        results[info["winner"]] += 1
        red_hps.append(max(0, info["red_hp"]))
        ticks_list.append(info["ticks"])
        dists.append(info["avg_dist"])
        shots.append(info["red_shots"])
        hits.append(info["red_hits"])
        print(f"Episode {ep + 1:3d}/{args.episodes}: Sieger={info['winner']:7s} "
              f"| Modus {info['blue_mode']:10s} | Blau-HP {info['blue_hp']:4d} "
              f"| Ø-Dist {info['avg_dist']:5.0f} px "
              f"| Schuesse {info['red_shots']:3d} | Treffer {info['red_hits']}")

    n = args.episodes
    print("\n========== ERGEBNIS ==========")
    print(f"Episoden:        {n}")
    print(f"Siege Rot (KI):  {results['red']:4d}  ({results['red'] / n * 100:.1f} %)")
    print(f"Siege Blau:      {results['blue']:4d}  ({results['blue'] / n * 100:.1f} %)")
    print(f"Timeouts:        {results['timeout']:4d}  ({results['timeout'] / n * 100:.1f} %)")
    print(f"Mittl. Rot-HP:   {np.mean(red_hps):.1f}")
    print(f"Mittl. Dauer:    {np.mean(ticks_list):.0f} Ticks")
    print("\n---- Verhaltens-Metriken (Jagd- und Feuerdisziplin-Nachweis) ----")
    print(f"Ø-Distanz zu Blau:  {np.mean(dists):.0f} px  "
          f"(niedrig gegen passiven Bot = Agent rueckt aktiv an)")
    total_shots, total_hits = sum(shots), sum(hits)
    acc = 100 * total_hits / max(1, total_shots)
    print(f"Genauigkeit:        {total_hits}/{total_shots} Treffer "
          f"= {acc:.1f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Gladiator - Evaluation")
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    parser.add_argument("--policy", type=str, default="model",
                        choices=["model", "random"],
                        help="'random' = untrainierte Baseline messen")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--blue-error", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--auto-aim-red", action="store_true",
                        help="Muss gesetzt sein, wenn das Modell mit "
                             "--auto-aim-red trainiert wurde")
    parser.add_argument("--blue-mode", type=str, default="skirmisher",
                        choices=["skirmisher", "passive", "flee", "mixed"],
                        help="'passive' ist der Jagd-Test: rueckt der Agent "
                             "von selbst an?")
    args = parser.parse_args()
    evaluate(args)
