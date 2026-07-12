# AI Gladiator – Vision-Based Double DQN

Ein Reinforcement-Learning-Agent lernt in einer 2D-Arena (Pygame) **ausschließlich aus rohen Pixeln**, einen hardcodierten Skirmisher-Bot zu bekämpfen: ausweichen, positionieren, zielen, schießen, gewinnen. Der Agent erhält keinerlei Koordinaten oder Zustandsvariablen die gesamte Wahrnehmung läuft über ein Convolutional Neural Network auf 80×60-Graustufen-Frames

## Spielregeln

| Regel | Wert |
|---|---|
| HP pro Schiff | 100 |
| Schaden pro Treffer | 15 (→ 7 Treffer = Zerstörung) |
| Episoden-Timeout | Training: 3600 Physik-Ticks (= 1 Minute bei 60 FPS). Mensch-Modus: **kein Timeout**, Runde endet nur durch Zerstörung |
| Schussrichtungen | Beide Schiffe schießen ausschließlich in 4 Richtungen (oben/unten/links/rechts); die Blickrichtung rastet auf die Achse ein, bei diagonaler Bewegung zählt die Horizontale |
| Gegner (Blau) | Skript-Bot: jagt, hält Distanz 150–300 px, weicht Projektilen aus, umgeht Hindernisse per Ecken-Waypoints, schießt nur bei freier Sichtlinie. Schwierigkeitsregler: Mit Wahrscheinlichkeit `blue_error_rate` pro Tick friert Blau ein (simulierte Reaktionszeit) skaliert Jagdtempo, Feuerrate **und** Ausweichen gemeinsam. Frühere Version drosselte nur das Ausweichen; Blaus Offensive blieb auf jeder Stufe voll kompetent und war der lernenden Policy 6:1 überlegen |
| Agent (Rot) | Double-DQN, sieht nur Pixel |

## Architektur

### Beobachtung (State)
- 4 gestapelte Graustufen-Frames, 60×80 Pixel, uint8 → Tensor `[4, 60, 80]`, normalisiert auf [0, 1].
- **Frame-Skip 4** (Action-Repeat): eine Agenten-Entscheidung gilt für 4 Physik-Ticks. Der Stack umspannt dadurch 16 Ticks → Bewegungsrichtung und Geschwindigkeit von Projektilen sind aus dem Stack ablesbar.

### Aktionen (18 diskret)
`{9 Bewegungsrichtungen: 4 Achsen + 4 Diagonalen + Stillstand} × {Schuss, kein Schuss}`. Der Agent hat damit exakt dieselben Fähigkeiten wie ein Mensch mit Pfeiltasten + Leertaste: diagonales Ausweichen ist möglich, geschossen wird aber wie beim Menschen nur in 4 Richtungen (Facing rastet auf die Achse ein). Zielen ist eine **gelernte** Fähigkeit: Der Agent muss sich auf die Achse des Gegners bewegen und im richtigen Moment feuern; es gibt kein Auto-Aim.

### Netz (Dueling CNN-Q-Netz)
```
Input [B, 4, 60, 80]
→ Conv 32×8×8, Stride 4, ReLU   → [B, 32, 14, 19]
→ Conv 64×4×4, Stride 2, ReLU   → [B, 64,  6,  8]
→ Conv 64×3×3, Stride 1, ReLU   → [B, 64,  4,  6]
→ Flatten (1536)
→ FC 512, ReLU
→ Value-Head V(s) [B,1]  +  Advantage-Head A(s,a) [B,10]
→ Q(s,a) = V(s) + A(s,a) − mean_a' A(s,a')     (Dueling, Wang et al. 2016)
```

**Graustufen-Trennbarkeit:** Da die Beobachtung Graustufe ist, sind alle Spielobjekte in Luminanz-Bänder kodiert: Hintergrund ~12, Hindernisse 41–63, Gegner (Blau) 95–119, Agent (Rot) 181–215. Zwei Lektionen stecken in dieser Palette: (1) Mit der ursprünglichen Neon-Palette lagen beide Schiffe bei praktisch identischer Luminanz (114.7 vs. 117.1) Freund/Feind war nicht unterscheidbar. (2) Ein „dunkles Team" (36–82) auf fast schwarzem Hintergrund hat strukturell kaum Kontrast: Blaue Laser hatten im 10×-Downscale nur ~4 % Kontrast und waren für das Netz quasi unsichtbar weder Ausweichen noch die Kreditvergabe für eigene Schüsse (deren Flugphase der verbindende Zustand zwischen Abdrücken und Reward ist) waren damit lernbar. Nach Anhebung der Bänder und dickerer Laser-Geometrie (6×14 px): 10.3 % Kontrast für blaue, 23 % für rote Laser (mit der echten cv2-INTER_AREA-Pipeline nachgemessen).

### Lernverfahren: Double DQN 
Target-Berechnung:

```
a* = argmax_a Q_policy(s', a)          # Policy-Netz WÄHLT die Aktion
y  = r + γ · Q_target(s', a*)          # Target-Netz BEWERTET sie
```

Die Entkopplung von Auswahl und Bewertung reduziert die systematische Q-Wert-Überschätzung des klassischen DQN. Zusätzlich:

- **Experience Replay**: 30 000 Transitionen (Frames als uint8 gespeichert, ~1.15 GB), Minibatch 64, ein Gradienten-Schritt alle 2 Agenten-Schritte nach 2000 Warmup-Schritten (CPU-freundlich; DeepMind-Standard wäre alle 4).
- **n-Step-Returns (n = 5)** (Hessel et al. 2018, „Rainbow"): Targets `y = Σᵢ γⁱ·rᵢ + γ⁵·Q_target(s₊₅, argmax Q_policy)`. n = 5 statt Standard-3, hergeleitet aus der Laser-Flugzeit: 60 px pro Entscheidung bei 150–300 px Kampfdistanz = 3–5 Entscheidungen vom Abdrücken bis zum Einschlag — mit n = 5 landet der Treffer-/Near-Miss-Reward für fast alle Distanzen direkt im Return der Schuss-Aktion. Am Episodenende geflushte Teilsequenzen sind terminal und damit exakt (kein Bootstrap).
- **Target-Netz-Sync**: Hard-Update alle 2000 Gradienten-Schritte.
- **Huber-Loss (SmoothL1)** statt MSE, **Gradient Clipping** (max-Norm 10).
- **Epsilon-Greedy**: linear 1.0 → 0.10 über 100 000 Agenten-Schritte (Floor 0.10 statt 0.05: dauerhafte Exploration gegen pessimistische Q-Schätzungen bei langem Weg zum Sieg).
- **Auto-Curriculum** (`--curriculum`): Blaus Fehlerquote (Einfrier-Wahrscheinlichkeit pro Tick) startet bei 0.7 und sinkt um 0.1 bis auf 0.0 (ungedrosselter Voll-Skirmisher), sobald die Win-Rate über 50 Episoden ≥ 60 % liegt. Die Stufe, auf der sich das System einpendelt, ist damit direkt das gemessene Spielniveau des Agenten.
- Optimizer: Adam, lr = 2.5e-4, γ = 0.99.

### Reward-Design
Rewards sind bewusst klein skaliert (|r| ≤ 5), damit die TD-Fehler numerisch stabil bleiben:

| Ereignis | Reward |
|---|---|
| Rot trifft Blau | +1.0 |
| Rot wird getroffen | −0.5 |
| Sieg (Blau zerstört) | +5.0 |
| Niederlage | −5.0 |
| Near-Miss: roter Laser passiert Blau < 40 px | +0.02 (1× pro Projektil; dichtes Shaping-Signal Richtung Treffen. Frühere Schusskosten von −0.01 wurden entfernt: bei P(Treffer) ≪ 1 % einer unerfahrenen Policy dominierte die sichere Sofort-Strafe den seltenen Treffer-Reward und trainierte dem Agenten das Schießen ab – eine Explorationsfalle) |
| Timeout | −2.0 + min(0, 2.0 · (HP_Rot − HP_Blau)/100) — **nie positiv** |
| Roter Schuss trifft Wand | −0.02 |
| Zeitstrafe pro Tick | 0 — abgeschafft. Mit Zeitstrafe war „früh sterben" besser als „lang kämpfen und dann sterben"; sie bestrafte während der gesamten Lernphase genau das Zielverhalten |
| Shaping: Achsen-Ausrichtung auf Blau **bei freier Sichtlinie** | +0.0005/Tick (max. +1.8/Episode, bewusst < Sieg-Reward) |

Zwei Anti-Exploit-Maßnahmen gegen passives Wall-Camping (das dominante Fehlverhalten früherer Versionen):
1. Das Ausrichtungs-Shaping zählt nur bei freier Sichtlinie – hinter einer Wand lässt sich nichts farmen.
2. Der Timeout-Reward ist immer negativ (HP-Term auf ≤ 0 geklemmt): "einmal treffen, dann verstecken" ist keine Gewinnstrategie. Zusätzlich flankiert Blau bei blockierter Sichtlinie aktiv um Hindernisse herum, sodass Verstecken kein sicherer Zustand ist. Rot muss dadurch selbst lernen, Hindernisse zu umrunden – emergent, allein aus den Pixeln (Sichtlinien-Informationen fließen nur in Reward-Shaping und Bot-Verhalten ein, nie in die Beobachtung des Agenten).

## Installation

Voraussetzungen: Python ≥ 3.10, Linux oder Windows. Kein Display nötig (Training nutzt den SDL-Dummy-Treiber).

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bat
:: Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

GPU ist optional; CUDA wird automatisch genutzt, falls verfügbar.

## Nutzung

```bash
# Training (headless + Curriculum, empfohlen). Strg+C speichert und beendet sauber.
python train.py --headless --curriculum

# Training mit Fenster (langsamer, zum Zuschauen)
python train.py --render

# Training fortsetzen (Episodenzaehler, Epsilon, Bestwert und
# Curriculum-Stand werden aus models/train_state.json wiederhergestellt)
python train.py --headless --resume

# Reproduzierbarer Lauf mit festem Seed und Episoden-Limit
python train.py --headless --seed 42 --max-games 1000

# Evaluation: 100 Greedy-Episoden gegen den Bot, Win-Rate-Report
python eval.py --episodes 100 --headless

# Selbst gegen die KI spielen (Pfeiltasten + Leertaste)
python play.py
```

Artefakte:
- `models/best_model.pth` – bestes Modell (höchster gleitender Mittelwert über 50 Episoden)
- `models/last_model.pth` – letzter Stand (für `--resume`)
- `runs/training_log.csv` – Episoden-Log (Reward, Sieger, HP, Epsilon, Win-Rate, …)
- `runs/training_progress.png` – Reward- und Win-Rate-Kurven

## Reproduktion der Ergebnisse

```bash
# Linux
bash scripts/reproduce.sh

# Windows
scripts\reproduce.bat
```

Die Skripte legen ein venv an, installieren die Abhängigkeiten, trainieren 1000 Episoden mit Seed 42 und evaluieren anschließend über 100 Episoden. Hinweis: Vollständige Determinismus-Garantien gibt es bei CUDA-Backends nicht; die Seeds fixieren Environment-Zufall (Spawns, Hindernisse, Bot-Verhalten) und die Netz-Initialisierung.

## Projektstruktur

```
arena_env.py   Environment: Spiel-Logik, Rendering, Beobachtung, Rewards
model.py       CNN-Q-Netz + vektorisierter Double-DQN-Trainer
agent.py       Agent: Replay Buffer, Epsilon-Greedy, Target-Sync
train.py       Trainings-Loop, Logging, Checkpoints, Plots
eval.py        Evaluation (Win-Rate über N Episoden)
play.py        Mensch vs. KI
scripts/       Reproduktions-Skripte (Linux + Windows)
```

### Reward-Ordnung (verifiziert)
Sieg (+5) ≫ Timeout (−2 bis −4) > Tod (−5, zeitunabhängig). Länger leben ist nie schlechter, Sterben das Schlechteste, Gewinnen das Beste. Maximales Shaping-Farming bis zum Timeout (+1.8 Align + 4.8 Near-Miss − 2 Timeout = +4.6) bleibt unter dem Sieg (+5, plus 7 Treffer-Rewards auf dem Weg) Gewinnen dominiert strikt.

## Pipeline-Smoke-Test

Schneller Nachweis (~1–2 h CPU), dass die Lern-Pipeline funktioniert, bevor man den langen Hauptlauf startet: Training gegen einen Bot, der 90 % der Ticks eingefroren ist – die Win-Rate muss innerhalb von 200–400 Episoden deutlich ansteigen. (Hinweis: In einer früheren Version drosselte `--blue-error` nur das Ausweichen, nicht die Offensive dieser Test war damit wirkungslos; seit der Einfrier-Mechanik testet er die volle Kette.)

```bash
python train.py --headless --blue-error 0.9 --max-games 400 --seed 1
```

## Baseline-Vergleich (Lernnachweis)

```bash
python eval.py --policy random --episodes 100 --headless   # untrainierte Baseline
python eval.py --episodes 100 --headless                   # trainiertes Modell
```

## Mensch vs. KI

`python play.py` – Pfeiltasten bewegen, Leertaste schießt, kein Timeout. Du zielst in Bewegungsrichtung; wie für alle Schiffe rastet die Schussrichtung auf 4 Richtungen ein (bei diagonaler Bewegung zählt die Horizontale). Mensch und Agent haben identische Fähigkeiten gleiche Geschwindigkeit, gleiche 9 Bewegungsrichtungen, gleicher Cooldown, gleiche 4 Schussrichtungen.

## Referenzen

- Mnih et al. (2015): *Human-level control through deep reinforcement learning*, Nature.
- van Hasselt, Guez, Silver (2016): *Deep Reinforcement Learning with Double Q-learning*, AAAI.
- Wang et al. (2016): *Dueling Network Architectures for Deep Reinforcement Learning*, ICML.
- Hessel et al. (2018): *Rainbow: Combining Improvements in Deep Reinforcement Learning*, AAAI (hieraus: n-Step-Returns).

## Lizenz

MIT – siehe [LICENSE](LICENSE).

