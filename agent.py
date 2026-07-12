# agent.py
# Agent-Klasse: Epsilon-Greedy-Policy, Experience Replay, Target-Sync.
# Der Trainings-Loop selbst liegt in train.py.

import json
import os
import random
from collections import deque

import numpy as np
import torch

from arena_env import N_ACTIONS
from model import CNN_QNet, DQNTrainer

# ----------------------------------------------------------------------
# Hyperparameter
# ----------------------------------------------------------------------
MAX_MEMORY = 30_000      # Transitionen im Replay Buffer.
                         # RAM: 30000 * 2 * 4*60*80 Byte (uint8) ~ 1.15 GB.
                         # Bei 1-Minuten-Episoden (bis 900 Entscheidungen)
                         # deckt das ~35-100 Episoden ab; kleinere Buffer
                         # riskieren katastrophales Vergessen.
BATCH_SIZE = 64
LR = 2.5e-4
GAMMA = 0.99
N_STEP = 5               # n-Step-Returns (Rainbow, Hessel et al. 2018):
                         # y = sum_i gamma^i * r_i + gamma^n * Q_target(...).
                         # n=5 statt Standard n=3, WEIL: Laser fliegen
                         # 15 px/Tick * 4 Ticks = 60 px pro Entscheidung;
                         # auf Kampfdistanz 150-300 px braucht ein Schuss
                         # 3-5 Entscheidungen bis zum Ziel. Mit n=5 landet
                         # der Near-Miss-/Treffer-Reward fuer fast alle
                         # Distanzen DIREKT im n-Step-Return der Schuss-
                         # Aktion (statt ueber Bootstrapping durch
                         # Laser-im-Flug-Zustaende zu muessen).
WARMUP_STEPS = 2_000     # Erst sammeln, dann trainieren (stabilere Anfangsphase)
TRAIN_EVERY = 2          # Gradientenschritt nur jeden 2. Agenten-Schritt.
                         # DeepMind-Standard ist jeder 4.; jeder 1. (vorher)
                         # war auf CPU der Flaschenhals -- diese Aenderung
                         # verdoppelt die Trainingsgeschwindigkeit in
                         # Echtzeit bei praktisch gleichem Lernfortschritt.
TARGET_SYNC = 2_000      # Target-Netz alle N Gradienten-Schritte synchronisieren

# Epsilon-Schedule: linear von 1.0 auf 0.05 ueber EXPLORE_STEPS
# Agenten-Schritte (jeder Schritt = frame_skip Physik-Ticks).
EPSILON_START = 1.0
EPSILON_MIN = 0.10   # 0.10 statt 0.05: Bei langem Weg zum Sieg (7 Treffer)
                     # braucht es dauerhaft genug Exploration, um aus
                     # pessimistischen Q-Schaetzungen ("kaempfen lohnt
                     # nicht") wieder herauszufinden. Klassischer
                     # Atari-DQN-Wert (Mnih et al. 2015).
EXPLORE_STEPS = 100_000


class ReplayMemory:
    """FIFO-Puffer fuer Transitionen (s, a, r, s', done). States als uint8."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.stack(next_states),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, device, resume=False, model_dir="./models"):
        self.device = device
        self.model_dir = model_dir
        self.n_games = 0
        self.total_steps = 0      # Agenten-Schritte (fuer Epsilon-Schedule)
        self.train_steps = 0      # Gradienten-Schritte (fuer Target-Sync)

        self.memory = ReplayMemory(MAX_MEMORY)

        self.policy_net = CNN_QNet(output_size=N_ACTIONS).to(device)
        self.target_net = CNN_QNet(output_size=N_ACTIONS).to(device)

        if resume:
            path = os.path.join(model_dir, "last_model.pth")
            if os.path.exists(path):
                self.policy_net.load_state_dict(
                    torch.load(path, map_location=device))
                print(f"[Resume] Gewichte aus {path} geladen.")
                # Trainingszustand (Episodenzaehler, Epsilon-Fortschritt,
                # Bestwert) exakt wiederherstellen statt zu schaetzen.
                state_path = os.path.join(model_dir, "train_state.json")
                if os.path.exists(state_path):
                    with open(state_path) as f:
                        s = json.load(f)
                    self.n_games = s.get("n_games", 0)
                    self.total_steps = s.get("total_steps", 0)
                    print(f"[Resume] Zustand: Episode {self.n_games}, "
                          f"Epsilon {self.epsilon:.3f}")
                else:
                    # Fallback ohne State-Datei: nicht wieder voll explorieren
                    self.total_steps = EXPLORE_STEPS // 2
            else:
                print("[Resume] Kein Checkpoint gefunden, starte frisch.")

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.trainer = DQNTrainer(self.policy_net, self.target_net,
                                  lr=LR, gamma=GAMMA, n_step=N_STEP)

        # Zwischenpuffer fuer n-Step-Returns: haelt die letzten N_STEP
        # 1-Step-Transitionen der laufenden Episode.
        self.nstep_queue = deque()

    # ------------------------------------------------------------------
    def _emit_front(self):
        """Schreibt die vorderste Transition des n-Step-Puffers als
        akkumulierte n-Step-Transition in den Replay Buffer:
        (s_0, a_0, sum_i gamma^i r_i, s_letzte, done_letzte)."""
        n_return = 0.0
        for i, (_, _, r, _, _) in enumerate(self.nstep_queue):
            n_return += (GAMMA ** i) * r
        s0, a0, _, _, _ = self.nstep_queue[0]
        _, _, _, ns_last, done_last = self.nstep_queue[-1]
        self.memory.push(s0, a0, n_return, ns_last, done_last)

    # ------------------------------------------------------------------
    def save_state(self, best_mean, blue_error, discipline=False):
        """Persistiert den Trainingszustand fuer sauberes --resume."""
        os.makedirs(self.model_dir, exist_ok=True)
        with open(os.path.join(self.model_dir, "train_state.json"), "w") as f:
            json.dump({"n_games": self.n_games,
                       "total_steps": self.total_steps,
                       "best_mean": best_mean,
                       "blue_error": blue_error,
                       "discipline": discipline}, f)

    def load_state_value(self, key, default):
        path = os.path.join(self.model_dir, "train_state.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f).get(key, default)
        return default

    # ------------------------------------------------------------------
    @property
    def epsilon(self):
        """Linearer Epsilon-Decay ueber die Agenten-Schritte."""
        frac = min(1.0, self.total_steps / EXPLORE_STEPS)
        return EPSILON_START + frac * (EPSILON_MIN - EPSILON_START)

    # ------------------------------------------------------------------
    def get_action(self, state_uint8, training=True):
        """Epsilon-greedy. state_uint8: np.uint8 [4, 60, 80]."""
        if training:
            self.total_steps += 1
            if random.random() < self.epsilon:
                return random.randrange(N_ACTIONS)

        state = (torch.as_tensor(state_uint8, device=self.device)
                 .float().div_(255.0).unsqueeze(0))
        with torch.no_grad():
            return int(self.policy_net(state).argmax(dim=1).item())

    # ------------------------------------------------------------------
    def remember(self, state, action, reward, next_state, done):
        """Nimmt 1-Step-Transitionen entgegen und schreibt n-Step-
        Transitionen in den Replay Buffer. Am Episodenende werden alle
        Teilsequenzen geflusht -- deren Horizont ist kuerzer als n, aber
        da sie terminal sind (done=True), wird im Trainer ohnehin nicht
        gebootstrapt; die Returns sind exakt."""
        self.nstep_queue.append((state, action, reward, next_state, done))
        if done:
            while self.nstep_queue:
                self._emit_front()
                self.nstep_queue.popleft()
        elif len(self.nstep_queue) == N_STEP:
            self._emit_front()
            self.nstep_queue.popleft()

    # ------------------------------------------------------------------
    def train_batch(self):
        """Ein Gradienten-Schritt auf einem zufaelligen Minibatch.
        Wird ab WARMUP_STEPS jeden Agenten-Schritt aufgerufen."""
        if len(self.memory) < max(BATCH_SIZE, WARMUP_STEPS):
            return None

        loss = self.trainer.train_step(*self.memory.sample(BATCH_SIZE))
        self.train_steps += 1

        if self.train_steps % TARGET_SYNC == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss

    # ------------------------------------------------------------------
    def save(self, file_name):
        self.policy_net.save(file_name=file_name, folder=self.model_dir)
