# model.py
# CNN-Q-Netz (DeepMind-Atari-Architektur, angepasst auf 60x80-Input)
# + vektorisierter Double-DQN-Trainer.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ======================================================================
# 1. CNN-Q-NETZ
# Input:  [Batch, 4, 60, 80]  (4 gestapelte Graustufen-Frames, float in [0,1])
# Output: [Batch, N_ACTIONS]  (Q-Wert je Aktion)
# ======================================================================
class CNN_QNet(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        # in_channels=4: die 4 gestapelten Frames sind die Eingangskanaele.
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Feature-Map-Groessen (floor((n - k) / s) + 1):
        #   Input   60 x 80
        #   Conv1:  14 x 19
        #   Conv2:   6 x  8
        #   Conv3:   4 x  6
        # -> 64 * 4 * 6 = 1536 flache Features
        flatten_size = 64 * 4 * 6

        # DUELING-ARCHITEKTUR (Wang et al. 2016):
        # Q(s,a) = V(s) + A(s,a) - mean_a' A(s,a')
        # Der Value-Stream lernt, WIE GUT ein Zustand generell ist (z. B.
        # "Laser im Anflug = schlecht"), unabhaengig davon, welche Aktion
        # gewaehlt wird. Der Advantage-Stream lernt nur noch die RELATIVE
        # Guete der Aktionen. Das beschleunigt das Lernen deutlich, weil in
        # vielen Zustaenden die Aktionswahl kaum eine Rolle spielt.
        self.fc1 = nn.Linear(flatten_size, 512)
        self.value_head = nn.Linear(512, 1)
        self.advantage_head = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        value = self.value_head(x)                        # [B, 1]
        advantage = self.advantage_head(x)                # [B, N_ACTIONS]
        # Mittelwert-Subtraktion macht die Zerlegung identifizierbar
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def save(self, file_name="best_model.pth", folder="./models"):
        os.makedirs(folder, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(folder, file_name))


# ======================================================================
# 2. DOUBLE-DQN-TRAINER
#
# Bellman-Target (Double DQN, van Hasselt et al. 2016):
#   a*  = argmax_a Q_policy(s', a)          (Policy-Netz WAEHLT die Aktion)
#   y   = r + gamma * Q_target(s', a*)      (Target-Netz BEWERTET sie)
# Das entkoppelt Auswahl und Bewertung und reduziert die systematische
# Ueberschaetzung der Q-Werte des klassischen DQN.
# ======================================================================
class DQNTrainer:
    def __init__(self, policy_net, target_net, lr, gamma, n_step=1):
        self.policy_net = policy_net
        self.target_net = target_net
        self.gamma = gamma
        # n-Step-Bootstrap: die gespeicherten Rewards sind bereits
        # sum_{i<n} gamma^i r_i; der Bootstrap-Term wird deshalb mit
        # gamma^n diskontiert. Fuer terminale Transitionen (auch die am
        # Episodenende geflushten mit Horizont < n) ist der Bootstrap 0,
        # dort spielt der Exponent keine Rolle.
        self.gamma_boot = gamma ** n_step
        self.optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        # Huber-Loss (SmoothL1) statt MSE: robust gegen Ausreisser im
        # TD-Fehler -> stabileres Training (DeepMind-Standard).
        self.criterion = nn.SmoothL1Loss()

    def train_step(self, states, actions, rewards, next_states, dones):
        """
        Ein Gradienten-Schritt auf einem Minibatch. Vollstaendig
        vektorisiert (keine Python-Schleife ueber Samples).

        states, next_states: np.uint8 [B, 4, 60, 80]
        actions:             np.int64 [B]   (Aktions-Indizes)
        rewards:             np.float32 [B]
        dones:               np.float32 [B] (0.0 oder 1.0)
        """
        device = next(self.policy_net.parameters()).device

        # uint8 -> float in [0, 1] erst HIER (spart 4x RAM im Replay Buffer)
        states = torch.as_tensor(states, device=device).float().div_(255.0)
        next_states = torch.as_tensor(next_states, device=device).float().div_(255.0)
        actions = torch.as_tensor(actions, dtype=torch.long, device=device)
        rewards = torch.as_tensor(rewards, dtype=torch.float, device=device)
        dones = torch.as_tensor(dones, dtype=torch.float, device=device)

        # Q(s, a) fuer die tatsaechlich ausgefuehrten Aktionen
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double-DQN-Target (ohne Gradienten), Bootstrap mit gamma^n
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma_boot * next_q * (1.0 - dones)

        loss = self.criterion(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping gegen explodierende Gradienten
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()
