# train.py
# Trainings-Loop: Rot (Double-DQN-Agent) lernt gegen den Skirmisher-Bot Blau.
#
# Nutzung:
#   python train.py --headless                 # Training ohne Fenster (schnell)
#   python train.py --render                   # Training mit Fenster
#   python train.py --headless --max-games 500 # Reproduzierbarer Kurzlauf
#   python train.py --headless --resume        # Training fortsetzen

import argparse
import csv
import os
import time
from collections import deque

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # Kein Display noetig -> laeuft headless auf Linux & Windows
import matplotlib.pyplot as plt

from arena_env import GladiatorEnv
from agent import Agent, EPSILON_MIN, EXPLORE_STEPS, TRAIN_EVERY

RUNS_DIR = "./runs"
PLOT_EVERY = 25          


def save_plot(rewards, mean_rewards, win_rates, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(rewards, alpha=0.4, label="Episoden-Reward")
    ax1.plot(mean_rewards, label=f"Mittel (letzte {ROLLING_WINDOW})")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(win_rates, color="green",
             label=f"Win-Rate (letzte {ROLLING_WINDOW})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win-Rate")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Vision-Based Double DQN - Trainingsverlauf")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Training auf: {device.type.upper()}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    os.makedirs(RUNS_DIR, exist_ok=True)
    log_path = os.path.join(RUNS_DIR, "training_log.csv")
    new_log = not (args.resume and os.path.exists(log_path))
    log_file = open(log_path, "w" if new_log else "a", newline="")
    logger = csv.writer(log_file)
    if new_log:
        logger.writerow(["episode", "reward", "winner", "red_hp", "blue_hp",
                         "ticks", "epsilon", "mean_reward", "win_rate",
                         "blue_error", "blue_mode", "buffer", "seconds"])

  
    blue_error = 0.7 if args.curriculum else args.blue_error

    agent = Agent(device, resume=args.resume)
    if args.resume:
        blue_error = agent.load_state_value("blue_error", blue_error)

   
    if args.explore_boost is not None:
        eps = max(EPSILON_MIN, min(1.0, args.explore_boost))
        agent.total_steps = int((1.0 - eps) / (1.0 - EPSILON_MIN)
                                * EXPLORE_STEPS)
        print(f"[Explore-Boost] Epsilon auf {agent.epsilon:.3f} gesetzt.")

    env = GladiatorEnv(headless=not args.render, training_mode=True,
                       frame_skip=4, blue_error_rate=blue_error,
                       auto_aim_red=args.auto_aim_red, seed=args.seed,
                       shot_cost=args.shot_cost, near_reward=args.near_reward,
                       blue_mode=args.blue_mode)

    rewards_hist, mean_hist, winrate_hist = [], [], []
    recent_rewards = deque(maxlen=ROLLING_WINDOW)
    recent_wins = deque(maxlen=ROLLING_WINDOW)


    manual_rewards = (args.shot_cost is not None
                      or args.near_reward is not None)
    discipline = bool(agent.load_state_value("discipline", False)) \
        if args.resume else False
    recent_acc = deque(maxlen=ROLLING_WINDOW)
    if discipline and not manual_rewards:
        env.r_shoot, env.r_near = -0.02, 0.0
        print("[Reward-Schedule] Feindisziplin-Phase aktiv (aus Resume).")
    best_mean = agent.load_state_value("best_mean", -float("inf")) \
        if args.resume else -float("inf")
    games_since_curriculum = 0

    state = env.reset()
    ep_reward = 0.0
    t0 = time.time()

    try:
        while args.max_games is None or agent.n_games < args.max_games:
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            if agent.total_steps % TRAIN_EVERY == 0:
                agent.train_batch()   # Minibatch-Gradientenschritt

            state = next_state
            ep_reward += reward

            if done:
                agent.n_games += 1
                recent_rewards.append(ep_reward)
                recent_wins.append(1.0 if info["winner"] == "red" else 0.0)
                recent_acc.append(info["red_hits"] / max(1, info["red_shots"]))

                # Adaptiver Reward-Schedule: Umschaltung auf Feindisziplin
                if (not discipline and not manual_rewards
                        and len(recent_acc) == ROLLING_WINDOW
                        and float(np.mean(recent_acc)) >= 0.03):
                    discipline = True
                    env.r_shoot, env.r_near = -0.02, 0.0
                    print(f"[Reward-Schedule] Genauigkeit "
                          f"{np.mean(recent_acc)*100:.1f} % >= 3 % -- "
                          f"Feindisziplin aktiviert (Schusskosten 0.02, "
                          f"Near-Miss aus).")

                mean_reward = float(np.mean(recent_rewards))
                win_rate = float(np.mean(recent_wins))
                rewards_hist.append(ep_reward)
                mean_hist.append(mean_reward)
                winrate_hist.append(win_rate)

                # Curriculum: Bot verstaerken, wenn Rot dominiert
                games_since_curriculum += 1
                if (args.curriculum and win_rate >= 0.60
                        and len(recent_wins) >= ROLLING_WINDOW
                        and games_since_curriculum >= ROLLING_WINDOW
                        and env.blue_error_rate > 0.001):
                    env.blue_error_rate = round(env.blue_error_rate - 0.1, 2)
                    games_since_curriculum = 0
                    recent_wins.clear()   # Win-Rate gegen neuen Bot neu messen
                   
                    best_mean = -float("inf")
                    print(f"[Curriculum] Blau verstaerkt: Fehlerquote jetzt "
                          f"{env.blue_error_rate:.2f}")

                logger.writerow([agent.n_games, f"{ep_reward:.3f}",
                                 info["winner"], info["red_hp"],
                                 info["blue_hp"], info["ticks"],
                                 f"{agent.epsilon:.3f}", f"{mean_reward:.3f}",
                                 f"{win_rate:.2f}", f"{env.blue_error_rate:.2f}",
                                 info["blue_mode"], len(agent.memory),
                                 f"{time.time() - t0:.0f}"])
                log_file.flush()

                print(f"Ep {agent.n_games:4d} | R {ep_reward:7.2f} | "
                      f"Mean {mean_reward:7.2f} | Win% {win_rate*100:5.1f} | "
                      f"Eps {agent.epsilon:.3f} | BlauErr {env.blue_error_rate:.2f} "
                      f"| Modus: {info['blue_mode']:10s} | Sieger: {info['winner']}")

                # Bestes Modell: hoechster GLEITENDER Mittelwert (statt
                # einzelner verrauschter Episoden-Rekorde)
                if (len(recent_rewards) >= ROLLING_WINDOW
                        and mean_reward > best_mean):
                    best_mean = mean_reward
                    agent.save("best_model.pth")
                    print(f"[Checkpoint] Neues bestes Modell "
                          f"(Mean-Reward {best_mean:.2f})")

                if agent.n_games % PLOT_EVERY == 0:
                    agent.save("last_model.pth")
                    agent.save_state(best_mean, env.blue_error_rate, discipline)
                    save_plot(rewards_hist, mean_hist, winrate_hist,
                              os.path.join(RUNS_DIR, "training_progress.png"))

                state = env.reset()
                ep_reward = 0.0
    except KeyboardInterrupt:
        print("\n[Abbruch] Training unterbrochen, speichere last_model.pth ...")
    finally:
        agent.save("last_model.pth")
        agent.save_state(best_mean, env.blue_error_rate, discipline)
        if rewards_hist:
            save_plot(rewards_hist, mean_hist, winrate_hist,
                      os.path.join(RUNS_DIR, "training_progress.png"))
        log_file.close()
        print(f"[Ende] {agent.n_games} Episoden, Log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Gladiator - Double-DQN-Training")
    parser.add_argument("--headless", action="store_true",
                        help="Ohne Fenster trainieren (Standard, schnell)")
    parser.add_argument("--render", action="store_true",
                        help="Mit Fenster trainieren (langsam, zum Zuschauen)")
    parser.add_argument("--resume", action="store_true",
                        help="Training aus models/last_model.pth fortsetzen")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Nach N Episoden stoppen (Standard: unbegrenzt)")
    parser.add_argument("--blue-error", type=float, default=0.30,
                        help="Fixe Fehlerquote des blauen Bots beim Ausweichen")
    parser.add_argument("--curriculum", action="store_true",
                        help="Auto-Curriculum: Blau startet bei Fehlerquote "
                             "0.7 und wird bei Win-Rate >= 60 %% schrittweise "
                             "bis auf 0.0 verstaerkt -- der ungedrosselte "
                             "Voll-Skirmisher als Endgegner (empfohlen)")
    parser.add_argument("--auto-aim-red", action="store_true",
                        help="Rot erhaelt dieselbe achsenbasierte Ziel-"
                             "korrektur wie der Skript-Bot (Symmetrie-"
                             "Mechanik). Ablations-Ergebnis: Ohne sie ist "
                             "Rots Zielen an die Bewegungsrichtung "
                             "gekoppelt -- es kann nicht gleichzeitig "
                             "ausweichen und feuern, Blau schon; das "
                             "deckelte die Win-Rate bei ~15 %%")
    parser.add_argument("--blue-mode", type=str, default="mixed",
                        choices=["skirmisher", "passive", "flee", "mixed"],
                        help="Gegner-Verhalten (Standard: mixed, sampelt pro "
                             "Episode 40/30/30 Skirmisher/Passiv/Flucht) -- "
                             "noetig, damit der Agent aktives Jagen lernt "
                             "statt nur den anrueckenden Skirmisher zu "
                             "kontern")
    parser.add_argument("--shot-cost", type=float, default=None,
                        help="Feindisziplin-Feintuning: Kosten pro Schuss "
                             "(z. B. 0.02). Erst einsetzen, wenn die Policy "
                             "bereits trifft -- fuer unerfahrene Policies "
                             "ist das eine Explorationsfalle")
    parser.add_argument("--near-reward", type=float, default=None,
                        help="Near-Miss-Shaping ueberschreiben (0 = aus). "
                             "Bootstrap-Hilfe; belohnt bei kompetenter "
                             "Policy nur noch Dauerfeuer-Spam")
    parser.add_argument("--explore-boost", type=float, default=None,
                        help="Epsilon beim (Re-)Start einmalig auf diesen "
                             "Wert anheben (z. B. 0.3), zerfaellt danach "
                             "normal. Sinnvoll nach Reward-Aenderungen oder "
                             "wenn die Policy in einem lokalen Optimum steckt")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed fuer Reproduzierbarkeit")
    args = parser.parse_args()
    train(args)
