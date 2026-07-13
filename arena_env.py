import os
import math
import random
from collections import deque

import numpy as np


import pygame
import cv2

# ----------------------------------------------------------------------
# Größe der Arena und Beobachtungsraum
# ----------------------------------------------------------------------
WIDTH, HEIGHT = 800, 600          # Arena-Groesse in Pixeln
FRAME_W, FRAME_H = 80, 60         # Beobachtungs-Aufloesung (Netz-Input)
STACK_SIZE = 4                    # Anzahl gestapelter Frames
TRAIN_MAX_TICKS = 3600            # Trainings-Timeout: 1 Minute bei 60 FPS.
                                 
# ----------------------------------------------------------------------
# Farbpallete und Beobachtbarkeit
# ----------------------------------------------------------------------

BLACK = (10, 10, 15)
GRID_COLOR = (30, 30, 45)
RED = (255, 150, 150)          # Ship-Outline Rot, Luminanz 181
BLUE = (0, 140, 255)           # Ship-Outline Blau, Luminanz 111
RED_FILL = (210, 210, 220)     # Rumpf Agent, Luminanz 211
BLUE_FILL = (60, 100, 160)     # Rumpf Gegner, Luminanz 95
RED_BULLET = (255, 200, 140)   # Luminanz 210
BLUE_BULLET = (0, 120, 255)    # Luminanz 100
BLUE_CORE = (40, 140, 220)     # Laser-Kern Blau, Luminanz 119
GREEN = (0, 255, 100)
CYAN = (0, 90, 90)             # Hindernis-Outline, Luminanz 63

# ----------------------------------------------------------------------
# Die 18 Mögliche Aktionen
# ----------------------------------------------------------------------
ACTIONS = [(dx, dy, shoot)
           for shoot in (False, True)
           for dy in (-1, 0, 1)
           for dx in (-1, 0, 1)]
N_ACTIONS = len(ACTIONS)  # 18

# ----------------------------------------------------------------------
# Die Reward Konstanten
# ----------------------------------------------------------------------
R_HIT = 1.0        # Rot trifft Blau
R_GOT_HIT = -0.5   # Rot wird getroffen
R_WIN = 5.0        # Blau zerstoert
R_LOSE = -5.0      # Rot zerstoert
R_TIMEOUT = -2.0   # Basis-Strafe bei Timeout. WICHTIG: Der HP-Differenz-
                  
R_TIMEOUT_HP = 2.0 # Zusatz-Strafe bei HP-Rueckstand zum Timeout-Zeitpunkt
R_SHOOT = 0.0      # Schusskosten DEAKTIVIERT. 
                   
R_NEAR = 0.02      # Near-Miss-Shaping: roter Laser passiert Blau < 40 px
                   
R_WALL = -0.02     # Roter Schuss trifft Hindernis (zusaetzlich zu R_SHOOT)
R_STEP = 0.0       # Zeitstrafe ABGESCHAFFT.
                  
R_APPROACH = 0.001 # Annaeherungs-Shaping, POTENTIALBASIERT (
                  
R_ALIGN = 0.0005  


class Particle:
    """Rein kosmetischer Explosions-Partikel."""

    def __init__(self, x, y, color):
        self.x, self.y = x, y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 6)
        self.dx = math.cos(angle) * speed
        self.dy = math.sin(angle) * speed
        self.timer = random.randint(15, 30)

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.timer -= 1

    def draw(self, surface):
        if self.timer > 0:
            size = max(1, int((self.timer / 30) * 4))
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), size)

# ----------------------------------------------------------------------
# Projektile
# ----------------------------------------------------------------------
class Bullet:
    """Projektil. Fliegt geradlinig, deaktiviert sich am Rand."""

    def __init__(self, x, y, dx, dy, color, owner):
        self.x, self.y = x, y
        self.dx, self.dy = dx, dy
        self.speed = 15
        self.color = color
        self.radius = 4
        self.damage = 15
        self.active = True
        self.owner = owner
        self.near_rewarded = False  # Near-Miss-Shaping nur 1x pro Projektil

    def update(self):
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed
        if self.x < 0 or self.x > WIDTH or self.y < 0 or self.y > HEIGHT:
            self.active = False

    def draw(self, surface):
       
        end_x = self.x - self.dx * 14
        end_y = self.y - self.dy * 14
        pygame.draw.line(surface, self.color,
                         (int(self.x), int(self.y)), (int(end_x), int(end_y)), 6)
        core = (255, 255, 255) if self.color == RED_BULLET else BLUE_CORE
        pygame.draw.circle(surface, core, (int(self.x), int(self.y)), 3)

# ----------------------------------------------------------------------
# Schiffe und Bewegung
# ----------------------------------------------------------------------

class Fighter:
    """Ein Schiff (Spieler oder Agent) mit HP, Bewegung und Waffe."""

    def __init__(self, x, y, color, start_dir, is_enemy=False):
        self.x, self.y = x, y
        self.color = color
        self.radius = 20
        self.speed = 5
        self.max_hp = 100
        self.hp = self.max_hp
        self.shoot_cooldown = 0
        self.facing_dir = start_dir
        self.is_enemy = is_enemy

    def move(self, dx, dy, obstacles):
        # Diagonale Bewegung normalisieren, damit sie nicht schneller ist.
        if dx != 0 and dy != 0:
            length = math.sqrt(dx ** 2 + dy ** 2)
            dx = (dx / length) * self.speed
            dy = (dy / length) * self.speed
        else:
            dx *= self.speed
            dy *= self.speed

        # Achsengetrennte Kollision -> Entlanggleiten an Waenden.
        self.x += dx
        rect = pygame.Rect(self.x - self.radius, self.y - self.radius,
                           self.radius * 2, self.radius * 2)
        for obs in obstacles:
            if rect.colliderect(obs):
                self.x -= dx

        self.y += dy
        rect = pygame.Rect(self.x - self.radius, self.y - self.radius,
                           self.radius * 2, self.radius * 2)
        for obs in obstacles:
            if rect.colliderect(obs):
                self.y -= dy

        self.x = max(self.radius, min(WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(HEIGHT - self.radius, self.y))

        if dx != 0 or dy != 0:
            # 8-WEGE-ZIELEN: Die Blickrichtung (= Schussrichtung) folgt
            # der normalisierten Bewegungsrichtung -- 4 Achsen + 4
            # Diagonalen. Gilt fuer ALLE Schiffe (Agent, Bot, Mensch).
            mag = math.hypot(dx, dy)
            self.facing_dir = (dx / mag, dy / mag)

    def update(self):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

    def shoot(self, bullets_list):
        if self.shoot_cooldown == 0:
            spawn_x = self.x + self.facing_dir[0] * self.radius
            spawn_y = self.y + self.facing_dir[1] * self.radius
            bullet_color = RED_BULLET if self.is_enemy else BLUE_BULLET
            bullets_list.append(Bullet(spawn_x, spawn_y,
                                       self.facing_dir[0], self.facing_dir[1],
                                       bullet_color, self))
            self.shoot_cooldown = 15

    def draw(self, surface):
        p1 = (self.x + self.facing_dir[0] * self.radius,
              self.y + self.facing_dir[1] * self.radius)
        perp = (-self.facing_dir[1], self.facing_dir[0])
        p2 = (self.x - self.facing_dir[0] * self.radius + perp[0] * 15,
              self.y - self.facing_dir[1] * self.radius + perp[1] * 15)
        p3 = (self.x - self.facing_dir[0] * self.radius - perp[0] * 15,
              self.y - self.facing_dir[1] * self.radius - perp[1] * 15)

        pygame.draw.polygon(surface, self.color, [p1, p2, p3], 3)
        fill = RED_FILL if self.is_enemy else BLUE_FILL
        pygame.draw.polygon(surface, fill, [p1, p2, p3])
        pygame.draw.circle(surface, self.color,
                           (int(self.x - self.facing_dir[0] * self.radius * 0.8),
                            int(self.y - self.facing_dir[1] * self.radius * 0.8)), 4)

        # HP-Balken
        bar_y = self.y - 35
        ratio = self.hp / self.max_hp if self.hp > 0 else 0
        pygame.draw.rect(surface, (100, 100, 100), (self.x - 25, bar_y, 50, 6), 1)
        pygame.draw.rect(surface, GREEN if ratio > 0.3 else RED,
                         (self.x - 24, bar_y + 1, 48 * ratio, 4))


# ======================================================================
# Das eigentliche Reinforcement-Learning-Environment
# ======================================================================
class GladiatorEnv:
    """
    API:
        obs = env.reset()                      # uint8, Shape [4, 60, 80]
        obs, reward, done, info = env.step(a)  # a = int in [0, N_ACTIONS)

    Parameter:
        headless        True -> SDL-Dummy-Treiber, kein Fenster (Training).
        training_mode   True -> Blau wird vom Skript gesteuert.
                        False -> Blau via Tastatur (play.py).
        frame_skip      Physik-Ticks pro Agenten-Entscheidung.
        blue_error_rate Wahrscheinlichkeit, dass Blau ein Ausweichmanoever
                        "verpennt" (Trainingsraeder fuer Rot).
        auto_aim_red    OPTIONALER Easy-Mode: Rot bekommt dieselbe
                        achsenbasierte Zielkorrektur wie Blau. Standard:
                        AUS -- mit Auto-Aim gewinnt bereits eine reine
                        Zufalls-Policy (~50 %% der Aktionen schiessen,
                        jeder Schuss ist gezielt, und Blau laeuft zum
                        eigenen Zielen selbst in die Schusslinie). Ohne
                        Auto-Aim folgt Rots Schussrichtung der Bewegung;
                        Zielen ist damit eine GELERNTE Faehigkeit.
        seed            Reproduzierbarkeit.
        max_ticks       Timeout in Physik-Ticks (Training: 3600 = 1 min).
                        None = kein Timeout (fuer Spiele gegen Menschen).
    """

    def __init__(self, headless=False, training_mode=False, frame_skip=4,
                 blue_error_rate=0.30, auto_aim_red=False, seed=None,
                 max_ticks=TRAIN_MAX_TICKS,
                 shot_cost=None, near_reward=None, blue_mode="skirmisher"):
        self.headless = headless
        self.training_mode = training_mode
        self.frame_skip = frame_skip
        self.blue_error_rate = blue_error_rate
        self.auto_aim_red = auto_aim_red
        self.max_ticks = max_ticks  

        self.r_shoot = R_SHOOT if shot_cost is None else -abs(shot_cost)
        self.r_near = R_NEAR if near_reward is None else near_reward
       
        assert blue_mode in ("skirmisher", "passive", "flee", "mixed")
        self.blue_mode = blue_mode
        self.episode_blue_mode = "skirmisher"

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Headless: Dummy-Videotreiber MUSS vor display.init gesetzt sein.
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()

        if self.headless:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        else:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("AI Gladiator - Vision-Based Double DQN")
        self.clock = pygame.time.Clock()

        self.frames = deque(maxlen=STACK_SIZE)
        self.tick = 0
        self.blue_wander_timer = 0
        self.blue_wander_dir = (0, 0)
        # Stuck-Erkennung: Positions-Historie der letzten 40 Ticks + zuletzt
        # kommandierte Bewegung. Ersetzt den rein zufaelligen Anti-Stuck.
        self.blue_pos_history = deque(maxlen=40)
        self.blue_last_move = (0, 0)
        self.blue_waypoint = None
        self.reset()

# ======================================================================
# Zufällige Hindernisse
# ======================================================================
    def generate_random_obstacles(self, p1_x, p1_y, p2_x, p2_y):
        """2-4 zufaellige Hindernisse mit Sicherheitszonen um beide Spawns."""
        obstacles = []
        num_obstacles = random.randint(2, 4)
        safe_p1 = pygame.Rect(p1_x - 80, p1_y - 80, 160, 160)
        safe_p2 = pygame.Rect(p2_x - 80, p2_y - 80, 160, 160)

        for _ in range(num_obstacles):
            for _attempt in range(50):
                w = random.choice([50, 100, 150])
                h = random.choice([50, 100, 150])
                x = random.randint(50, WIDTH - w - 50)
                y = random.randint(50, HEIGHT - h - 50)
                new_rect = pygame.Rect(x, y, w, h)

                if new_rect.colliderect(safe_p1) or new_rect.colliderect(safe_p2):
                    continue
                if any(new_rect.colliderect(e.inflate(40, 40)) for e in obstacles):
                    continue
                obstacles.append(new_rect)
                break
        return obstacles

# ======================================================================
# Eine neue Episode Beginnt
# ======================================================================
    def reset(self):
        p1_x = random.randint(50, 350)
        p1_y = random.randint(50, HEIGHT - 50)
        p2_x = random.randint(450, WIDTH - 50)
        p2_y = random.randint(50, HEIGHT - 50)

        self.player = Fighter(p1_x, p1_y, BLUE, (1, 0), is_enemy=False)   # Blau
        self.enemy = Fighter(p2_x, p2_y, RED, (-1, 0), is_enemy=True)     # Rot (Agent)
        self.bullets = []
        self.particles = []
        self.obstacles = self.generate_random_obstacles(p1_x, p1_y, p2_x, p2_y)

        self.tick = 0
        self.blue_wander_timer = 0
        self.blue_pos_history = deque(maxlen=40)
        self.blue_last_move = (0, 0)
        self.blue_waypoint = None

        
        if self.blue_mode == "mixed":
            self.episode_blue_mode = random.choices(
                ("skirmisher", "passive", "flee"), weights=(0.4, 0.3, 0.3))[0]
        else:
            self.episode_blue_mode = self.blue_mode


        self._dist_accum = 0.0
        self._red_shots = 0
        self._red_hits = 0

   
        self._render()
        first_frame = self._capture_frame()
        for _ in range(STACK_SIZE):
            self.frames.append(first_frame)

        return np.stack(self.frames, axis=0)

# ======================================================================
# Von Spielbild zu CNN Beobachtung
# ======================================================================
    def _capture_frame(self):
        """Screen -> Graustufen-uint8-Frame [60, 80] (Netz-Beobachtung)."""
        raw = pygame.surfarray.array3d(self.screen)      # [W, H, 3]
        raw = np.transpose(raw, (1, 0, 2))               # [H, W, 3]
        gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)                  # uint8: 4x weniger RAM

    # ------------------------------------------------------------------
    def _draw_grid(self):
        for x in range(0, WIDTH, 50):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, 50):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WIDTH, y))

    def _render(self):
        self.screen.fill(BLACK)
        self._draw_grid()
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (20, 40, 40), obs)
            pygame.draw.rect(self.screen, CYAN, obs, 3)
        for p in self.particles:
            p.draw(self.screen)
        for b in self.bullets:
            b.draw(self.screen)
        self.player.draw(self.screen)
        self.enemy.draw(self.screen)
        pygame.draw.rect(self.screen, (50, 50, 70), (0, 0, WIDTH, HEIGHT), 4)

    # ------------------------------------------------------------------
    def _line_of_sight(self, a, b):
        """True, wenn zwischen den Mittelpunkten von a und b kein
        Hindernis liegt. Nutzt pygame.Rect.clipline (Linien-Rechteck-
        Schnitt); leeres Tupel = kein Schnitt."""
        for obs in self.obstacles:
            if obs.clipline((a.x, a.y), (b.x, b.y)):
                return False
        return True

    def _blocking_obstacle(self, a, b):
        """Erstes Hindernis, das die Sichtlinie a->b schneidet (oder None)."""
        for obs in self.obstacles:
            if obs.clipline((a.x, a.y), (b.x, b.y)):
                return obs
        return None

    def _blue_is_stuck(self):
        """Stuck-Erkennung: Blau hat sich in den letzten 40 Ticks trotz
        kommandierter Bewegung kaum vom Fleck bewegt (Wall-Hugging)."""
        if len(self.blue_pos_history) < self.blue_pos_history.maxlen:
            return False
        if self.blue_last_move == (0, 0):
            return False
        ox, oy = self.blue_pos_history[0]
        return math.hypot(self.player.x - ox, self.player.y - oy) < 15

    def _pick_detour_corner(self, mover, target):
        """
        Wand-Umgehung: Waehlt als Zwischenziel (Waypoint) die Ecke des
        blockierenden Hindernisses (um Schiffsradius+Marge aufgeblaeht),
        die den Umweg d(mover, Ecke) + d(Ecke, target) minimiert.
        Einfaches Waypoint-Verfahren -- kein A* noetig bei 2-4 konvexen
        Hindernissen. Rueckgabe: (x, y) oder None.
        """
        obs = self._blocking_obstacle(mover, target)
        if obs is None:
            return None

        margin = mover.radius + 15
        inflated = obs.inflate(2 * margin, 2 * margin)
        corners = [(inflated.left, inflated.top),
                   (inflated.right, inflated.top),
                   (inflated.left, inflated.bottom),
                   (inflated.right, inflated.bottom)]
        # In die Arena klemmen, damit die Ecke erreichbar bleibt
        corners = [(max(mover.radius, min(WIDTH - mover.radius, cx)),
                    max(mover.radius, min(HEIGHT - mover.radius, cy)))
                   for cx, cy in corners]

        # Nur Ecken zulassen, die vom Mover aus geradlinig erreichbar sind
        # (Pfad zur Ecke darf das blockierende Hindernis nicht schneiden).
        reachable = [c for c in corners
                     if not obs.clipline((mover.x, mover.y), c)]
        candidates = reachable if reachable else corners

        # Deadlock-Schutz: Eine bereits (fast) erreichte Ecke ohne
        # Sichtlinie darf nicht erneut Ziel sein, sonst parkt der Bot dort.
        far = [c for c in candidates
               if math.hypot(c[0] - mover.x, c[1] - mover.y) > 25]
        if far:
            candidates = far

        return min(candidates, key=lambda c:
                   math.hypot(c[0] - mover.x, c[1] - mover.y)
                   + math.hypot(c[0] - target.x, c[1] - target.y))

    def _blue_detour_step(self):
        """
        Steuert Blau um Hindernisse. WICHTIG: Blau COMMITTET auf einen
        Waypoint und behaelt ihn, bis er erreicht ist (< 15 px). Wuerde
        die Ecke jeden Tick neu gewaehlt, oszilliert der Bot an der
        Kostengrenze zwischen zwei Ecken hin und her (Hysterese-Problem,
        im Test nachgewiesen). Rueckgabe: (dx, dy).
        """
        wp = self.blue_waypoint
        if wp is not None and math.hypot(wp[0] - self.player.x,
                                         wp[1] - self.player.y) < 15:
            self.blue_waypoint = None  # Ecke erreicht -> naechste waehlen
            wp = None

        if wp is None:
            wp = self._pick_detour_corner(self.player, self.enemy)
            self.blue_waypoint = wp
            if wp is None:
                return 0, 0

        ddx = wp[0] - self.player.x
        ddy = wp[1] - self.player.y
        dx = 0 if abs(ddx) < 4 else (1 if ddx > 0 else -1)
        dy = 0 if abs(ddy) < 4 else (1 if ddy > 0 else -1)
        return dx, dy

# ======================================================================
# Der geskriptete Blaue Bot
# ======================================================================
    def _blue_policy(self):
        """
        Hardcodierter Skirmisher-Bot (Gegner des Agenten).

        SCHWIERIGKEITSREGLER (Lektion aus dem Training): Mit
        Wahrscheinlichkeit blue_error_rate pro Tick "friert" Blau ein --
        keine Bewegung, kein Schuss (simulierte Reaktionszeit eines
        schwaecheren Spielers). Das skaliert Jagdtempo, Feuerrate UND
        Ausweichen mit einem Knopf. Vorher drosselte die Fehlerquote nur
        das Ausweichen; Blaus Offensive (Auto-Aim, Jagd, Feuerrate) lief
        auf jeder Stufe mit voller Kompetenz -- die leichteste Stufe war
        damit immer noch 6:1 ueberlegen und der Sieg-Gradient existierte
        fuer den Agenten nie.

        GEGNER-DIVERSITAET: Pro Episode wird ein Stil gesampelt --
        'hunter' (50 %, bisheriges Skirmisher-Verhalten), 'camper' (25 %,
        steht, weicht nur aus, schiesst auf Angreifer) und 'coward'
        (25 %, flieht aktiv). Gegen Camper/Feigling erfordern Treffer und
        Siege aktive Verfolgung durch den Agenten.

        Prioritaeten (wenn aktiv):
          1) Projektilen ausweichen,
          2) Stuck-Aufloesung (erkannt via Positions-Historie),
          3) Stil: Camper steht / Feigling flieht / Jaeger:
          4) Wand-Umgehung bei blockierter Sichtlinie (Ecken-Waypoints),
          5) Skirmisher: Achse ausrichten + Distanz 150-300 halten.
        Schiesst NUR bei freier Sichtlinie -> keine Wandschuesse, und Rot
        kann sich nicht dauerhaft risikofrei verstecken, weil Blau flankiert.
        Rueckgabe: (dx, dy, shoot)
        """
        # Kompetenz-Drossel: eingefrorener Tick
        if random.random() < self.blue_error_rate:
            self.blue_last_move = (0, 0)
            return 0, 0, False

        self.blue_pos_history.append((self.player.x, self.player.y))
        dx1, dy1 = 0, 0
        has_los = self._line_of_sight(self.player, self.enemy)
        if has_los:
            self.blue_waypoint = None  # Umgehung beendet

        # 1) Ausweichen (kein separater Fehler-Roll mehr: Ausweichfehler
        # entstehen jetzt durch eingefrorene Ticks -- gleiche Netto-Quote)
        danger = None
        for b in self.bullets:
            if b.owner == self.enemy and b.active:
                dist = math.hypot(b.x - self.player.x, b.y - self.player.y)
                if dist < 150:
                    danger = b
                    break

        if danger:
           
            px, py = -danger.dy, danger.dx
            to_cx = WIDTH / 2 - self.player.x
            to_cy = HEIGHT / 2 - self.player.y
            if px * to_cx + py * to_cy < 0:
                px, py = -px, -py
            dx1 = 0 if abs(px) < 0.3 else (1 if px > 0 else -1)
            dy1 = 0 if abs(py) < 0.3 else (1 if py > 0 else -1)
        elif self.blue_wander_timer > 0:
            # Laufendes Ausweichmanoever fortsetzen
            dx1, dy1 = self.blue_wander_dir
            self.blue_wander_timer -= 1
        elif self._blue_is_stuck():
            # 2) Stuck erkannt: 20-40 Ticks senkrecht zur bisherigen
            #    (blockierten) Richtung ausbrechen
            px, py = self.blue_last_move
            if px != 0:
                self.blue_wander_dir = (0, random.choice([-1, 1]))
            else:
                self.blue_wander_dir = (random.choice([-1, 1]), 0)
            self.blue_wander_timer = random.randint(20, 40)
            dx1, dy1 = self.blue_wander_dir
            self.blue_pos_history.clear()
        elif self.episode_blue_mode == "passive":
           
            pass  # dx1, dy1 bleiben (0, 0)
        elif self.episode_blue_mode == "flee":
        
            dist_x = self.enemy.x - self.player.x
            dist_y = self.enemy.y - self.player.y
            if abs(dist_x) > abs(dist_y):
                dx1 = -1 if dist_x > 0 else 1
            else:
                dy1 = -1 if dist_y > 0 else 1
            if random.random() < 0.15:
                if dx1 != 0:
                    dy1 = random.choice([-1, 0, 1])
                else:
                    dx1 = random.choice([-1, 0, 1])
        elif not has_los:
            # 3) Sichtlinie blockiert (nur Skirmisher flankiert aktiv)
            dx1, dy1 = self._blue_detour_step()
        else:
            dist_x = self.enemy.x - self.player.x
            dist_y = self.enemy.y - self.player.y
            abs_x, abs_y = abs(dist_x), abs(dist_y)

            if random.random() < 0.03:
                # Gelegentliches unvorhersehbares Manoever (schwerer zu treffen)
                self.blue_wander_timer = random.randint(10, 30)
                self.blue_wander_dir = (random.choice([-1, 0, 1]),
                                        random.choice([-1, 0, 1]))
                dx1, dy1 = self.blue_wander_dir
            else:
                # 4) Skirmisher: Achse ausrichten, Distanz regeln
                if abs_x > abs_y:
                    if abs_y > 15:
                        dy1 = 1 if dist_y > 0 else -1
                    else:
                        
                        if abs_x < 150:
                            dx1 = -1 if dist_x > 0 else 1
                        elif abs_x > 300:
                            dx1 = 1 if dist_x > 0 else -1
                        if random.random() < 0.30:
                            dy1 = random.choice([-1, 1])
                else:
                    if abs_x > 15:
                        dx1 = 1 if dist_x > 0 else -1
                    else:
                        if abs_y < 150:
                            dy1 = -1 if dist_y > 0 else 1
                        elif abs_y > 300:
                            dy1 = 1 if dist_y > 0 else -1
                        if random.random() < 0.30:
                            dx1 = random.choice([-1, 1])

        self.blue_last_move = (dx1, dy1)

        
        dist_x = self.enemy.x - self.player.x
        dist_y = self.enemy.y - self.player.y
        aligned = (abs(dist_x) < 20 or abs(dist_y) < 20
                   or abs(abs(dist_x) - abs(dist_y)) < 20)
        rate = 0.10 if self.episode_blue_mode == "flee" else 0.15
        shoot = aligned and has_los and random.random() < rate
        return dx1, dy1, shoot

    # ------------------------------------------------------------------
    def _axis_aim(self, fighter, target):
        """Achsenbasierte Zielkorrektur: Blick auf dominante Achse zum Ziel."""
        dist_x = target.x - fighter.x
        dist_y = target.y - fighter.y
        if abs(dist_x) > abs(dist_y):
            fighter.facing_dir = (1 if dist_x > 0 else -1, 0)
        else:
            fighter.facing_dir = (0, 1 if dist_y > 0 else -1)

# ======================================================================
# 8 Wege Zielkorrektur
# ======================================================================

    def _aim_8way(self, fighter, target):
        """8-Wege-Zielkorrektur: Bei diagonaler Lage zum Ziel
        (|dx| ~ |dy|, Toleranz 20 px) wird diagonal gezielt, sonst auf
        die dominante Achse. Gleiche Richtungs-Verteilung (4 Achsen +
        4 Diagonalen), die ein Mensch per Pfeiltasten erzeugen kann."""
        dist_x = target.x - fighter.x
        dist_y = target.y - fighter.y
        if abs(abs(dist_x) - abs(dist_y)) < 20:
            d = 0.7071067811865476  # 1/sqrt(2)
            fighter.facing_dir = (d if dist_x > 0 else -d,
                                  d if dist_y > 0 else -d)
        else:
            self._axis_aim(fighter, target)


# ======================================================================
# Wichtigster Environment-Schritt
# ======================================================================
    def step(self, action_idx):
        """
        Fuehrt eine Agenten-Aktion fuer `frame_skip` Ticks aus.
        Rueckgabe: (obs [4,60,80] uint8, reward float, done bool, info dict)
        """
        assert 0 <= action_idx < N_ACTIONS, f"Ungueltige Aktion: {action_idx}"
        dx2, dy2, shoot2 = ACTIONS[action_idx]

        total_reward = 0.0
        done = False
        winner = None

        for _skip in range(self.frame_skip):
            self.tick += 1
            dist_before = math.hypot(self.enemy.x - self.player.x,
                                     self.enemy.y - self.player.y)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

            # ---------------- BLAU ----------------
            if self.training_mode:
                dx1, dy1, shoot1 = self._blue_policy()
                self.player.move(dx1, dy1, self.obstacles)
                self._aim_8way(self.player, self.enemy)   # 8-Wege-Aim Bot
                if shoot1:
                    self.player.shoot(self.bullets)
            else:
              
                keys = pygame.key.get_pressed()
                dx1 = (-1 if keys[pygame.K_LEFT] else 0) + (1 if keys[pygame.K_RIGHT] else 0)
                dy1 = (-1 if keys[pygame.K_UP] else 0) + (1 if keys[pygame.K_DOWN] else 0)
                self.player.move(dx1, dy1, self.obstacles)
                if keys[pygame.K_SPACE]:
                    self.player.shoot(self.bullets)

            # ---------------- ROT (Agent) ----------------
            self.enemy.move(dx2, dy2, self.obstacles)
            if self.auto_aim_red:
                self._aim_8way(self.enemy, self.player)   # 8-Wege-Zielkorrektur
            if shoot2:
                
                if self.enemy.shoot_cooldown == 0:
                    total_reward += self.r_shoot
                    self._red_shots += 1
                self.enemy.shoot(self.bullets)

            self._dist_accum += math.hypot(self.player.x - self.enemy.x,
                                           self.player.y - self.enemy.y)

            self.player.update()
            self.enemy.update()

            
            dist_after = math.hypot(self.enemy.x - self.player.x,
                                    self.enemy.y - self.player.y)
            total_reward += R_APPROACH * (max(0.0, dist_before - 250.0)
                                          - max(0.0, dist_after - 250.0))

            # ---------------- Projektile ----------------
            old_player_hp = self.player.hp
            old_enemy_hp = self.enemy.hp

            for bullet in self.bullets:
                bullet.update()
                brect = pygame.Rect(bullet.x - bullet.radius, bullet.y - bullet.radius,
                                    bullet.radius * 2, bullet.radius * 2)
                hit_wall = False
                for obs in self.obstacles:
                    if brect.colliderect(obs):
                        bullet.active = False
                        hit_wall = True
                        if bullet.owner == self.enemy:
                            total_reward += R_WALL
                        for _ in range(5):
                            self.particles.append(Particle(bullet.x, bullet.y, CYAN))
                        break

                if not hit_wall:
                    for target in (self.player, self.enemy):
                        if bullet.owner is not target and bullet.active:
                            dist = math.hypot(bullet.x - target.x, bullet.y - target.y)
                            if dist < target.radius + bullet.radius:
                                target.hp -= bullet.damage
                                bullet.active = False
                                for _ in range(15):
                                    self.particles.append(
                                        Particle(bullet.x, bullet.y, bullet.color))

                    
                    if (bullet.owner is self.enemy and bullet.active
                            and not bullet.near_rewarded):
                        d_near = math.hypot(bullet.x - self.player.x,
                                            bullet.y - self.player.y)
                        if d_near < 40:
                            total_reward += self.r_near
                            bullet.near_rewarded = True

            self.bullets = [b for b in self.bullets if b.active]
            for p in self.particles:
                p.update()
            self.particles = [p for p in self.particles if p.timer > 0]

            # ---------------- Rewards ----------------
            if self.player.hp < old_player_hp:
                total_reward += R_HIT
                self._red_hits += 1
            if self.enemy.hp < old_enemy_hp:
                total_reward += R_GOT_HIT

            
            sdx = self.enemy.x - self.player.x
            sdy = self.enemy.y - self.player.y
            s_aligned = (abs(sdx) < 20 or abs(sdy) < 20
                         or abs(abs(sdx) - abs(sdy)) < 20)
            if s_aligned and self._line_of_sight(self.enemy, self.player):
                total_reward += R_ALIGN

            total_reward += R_STEP

            if self.player.hp <= 0:
                total_reward += R_WIN
                done, winner = True, "red"
            elif self.enemy.hp <= 0:
                total_reward += R_LOSE
                done, winner = True, "blue"
            elif self.max_ticks is not None and self.tick >= self.max_ticks:
                
                hp_diff = (self.enemy.hp - self.player.hp) / 100.0
                total_reward += R_TIMEOUT + min(0.0, R_TIMEOUT_HP * hp_diff)
                done, winner = True, "timeout"

   
            if not self.headless:
                self._render()
                pygame.display.flip()
                self.clock.tick(60)

            if done:
                break

    
        self._render()
        if not self.headless:
            pygame.display.flip()

        self.frames.append(self._capture_frame())
        obs = np.stack(self.frames, axis=0)

        info = {
            "winner": winner,
            "red_hp": self.enemy.hp,
            "blue_hp": self.player.hp,
            "ticks": self.tick,
            "blue_mode": self.episode_blue_mode,
            "avg_dist": self._dist_accum / max(1, self.tick),
            "red_shots": self._red_shots,
            "red_hits": self._red_hits,
        }
        return obs, total_reward, done, info
