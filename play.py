import argparse
from collections import deque

import numpy as np
import pygame
import torch

from arena_env import GladiatorEnv, N_ACTIONS, STACK_SIZE
from model import CNN_QNet

AI_DECISION_INTERVAL = 4  


def play(args):
    print("=" * 50)
    print("SHOWDOWN: MENSCH (Blau) vs. KI (Rot)")
    print("Steuerung: Pfeiltasten = Bewegen | Leertaste = Schiessen")
    print("Du zielst in Bewegungsrichtung -- alle 8 Richtungen, Diagonalen")
    print("inklusive (Pfeil oben+rechts = Schraegschuss). Kein Timeout.")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_QNet(output_size=N_ACTIONS).to(device)

    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
    except FileNotFoundError:
        print(f"FEHLER: '{args.model}' nicht gefunden. Erst trainieren: "
              f"python train.py --headless")
        return
    model.eval()
    print(f"Modell geladen: {args.model}")


    env = GladiatorEnv(headless=False, training_mode=False, frame_skip=1,
                       max_ticks=None, auto_aim_red=args.auto_aim_red)


    ai_frames = deque(maxlen=STACK_SIZE)
    obs = env.reset()
    for _ in range(STACK_SIZE):
        ai_frames.append(obs[-1])

    tick = 0
    current_action = 0

    while True:
        if tick % AI_DECISION_INTERVAL == 0:
            state = np.stack(ai_frames, axis=0)
            t = (torch.as_tensor(state, device=device)
                 .float().div_(255.0).unsqueeze(0))
            with torch.no_grad():
                current_action = int(model(t).argmax(dim=1).item())

        obs, _, done, info = env.step(current_action)
        tick += 1
        if tick % AI_DECISION_INTERVAL == 0:
            ai_frames.append(obs[-1])

        if done:
            if info["winner"] == "red":
                print("Die KI hat gewonnen.")
            else:
                print("SIEG! Du hast die KI bezwungen.")
            pygame.time.wait(2000)
            obs = env.reset()
            ai_frames.clear()
            for _ in range(STACK_SIZE):
                ai_frames.append(obs[-1])
            tick = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Gladiator - Mensch vs. KI")
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    parser.add_argument("--auto-aim-red", action="store_true",
                        help="Muss gesetzt sein, wenn das Modell mit "
                             "--auto-aim-red trainiert wurde (sonst "
                             "schiesst Rot in seine Bewegungsrichtung "
                             "statt wie im Training gelernt)")
    args = parser.parse_args()
    play(args)
