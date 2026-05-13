# ============================================================
#  game_env.py  –  Môi trường game Chrome Dino hoàn chỉnh
#  PATCHED v3: dino x cố định khi boost, chỉ animation tăng tốc
# ============================================================
import os
import pygame
import random
import numpy as np
from shared.config import (
    SCREEN_W, SCREEN_H, FPS,
    INIT_SPEED, SPEED_INCREMENT, MAX_SPEED,
    STATE_SIZE, GROUND_Y_OFFSET, BG_COLOR,
    GRAVITY, JUMP_VEL,
)
from shared.renderer import SpriteLoader


BG_COLOR = (235, 235, 235)


class Dinosaur:
    """Khủng long – vật lý, animation & trạng thái.

    PATCH v3 LOG:
    - [Boost]  Bỏ hoàn toàn _vel_x drift — dino.x không còn thay đổi khi boost
    - [Boost]  Tốc độ chạy thể hiện qua: ground scroll nhanh hơn + animation
               frame_interval rút ngắn (1 thay vì 5 khi boosting) — đủ cảm giác
    - [Boost]  _jump_speed_factor giữ lại để API jump() không thay đổi
    """

    def __init__(self, sprites: SpriteLoader):
        self.sprites  = sprites
        self.x        = SCREEN_W // 15

        self.ground_y = SCREEN_H - GROUND_Y_OFFSET - sprites.dino_h
        self.y        = self.ground_y

        self._y_float = float(self.ground_y)
        self._vel_y   = 0.0
        # _vel_x đã bỏ — dino.x không thay đổi khi boost

        self.w      = sprites.dino_w
        self.h      = sprites.dino_h
        self.duck_w = sprites.duck_w
        self.duck_h = sprites.duck_h

        self.jump_vel = JUMP_VEL
        self.gravity  = GRAVITY

        self.is_jumping = False
        self.is_ducking = False
        self.is_dead    = False
        self.steps      = 0
        self.score      = 0
        self.counter    = 0
        self.anim_idx   = 0
        self._jump_speed_factor = 1.0

    # ── Helpers ───────────────────────────────────────────────

    def _update_mask(self):
        self.mask = pygame.mask.from_surface(self._get_current_sprite())

    def _get_current_sprite(self) -> pygame.Surface:
        if self.is_dead:
            return self.sprites.dino_dead()
        if self.is_ducking:
            return self.sprites.dino_duck_a() if self.anim_idx == 0 else self.sprites.dino_duck_b()
        if self.is_jumping:
            return self.sprites.dino_jump()
        return self.sprites.dino_stand() if self.anim_idx == 0 else self.sprites.dino_run_b()

    # ── Actions ───────────────────────────────────────────────

    def jump(self, speed_factor: float = 1.0):
        if not self.is_jumping and not self.is_ducking:
            self._jump_speed_factor = speed_factor
            self._vel_y     = -self.jump_vel
            self.is_jumping = True

    def duck(self):
        if not self.is_jumping:
            self.is_ducking = True
            self._y_float   = float(SCREEN_H - GROUND_Y_OFFSET - self.duck_h)
            self.y          = int(self._y_float)
            self.w          = self.duck_w
            self.h          = self.duck_h

    def unduck(self):
        if self.is_ducking:
            self.is_ducking = False
            self.w          = self.sprites.dino_w
            self.h          = self.sprites.dino_h
            self._y_float   = float(self.ground_y)
            self.y          = self.ground_y

    # ── Update mỗi frame ──────────────────────────────────────

    def update(self, game_speed: float, base_speed: float, is_boosting: bool = False):
        # ── Nhảy (vật lý y) ───────────────────────────────────
        if self.is_jumping:
            self._vel_y   += self.gravity
            self._y_float += self._vel_y
            self.y         = int(self._y_float)

            if self._y_float >= self.ground_y:
                self._y_float   = float(self.ground_y)
                self.y          = self.ground_y
                self.is_jumping = False
                self._vel_y     = 0.0

        # ── x hoàn toàn cố định ───────────────────────────────
        # Cảm giác tăng tốc đến từ ground/obstacle scroll nhanh hơn
        # và animation frame_interval bên dưới — KHÔNG drift x nữa.

        # ── Animation ─────────────────────────────────────────
        self.counter += 1
        if self.is_jumping:
            self.anim_idx = 0
            self.counter  = 0
        else:
            # Boost → frame_interval=1 (chạy cực nhanh)
            # Bình thường → frame_interval=5
            frame_interval = 1 if is_boosting else 5
            if self.counter >= frame_interval:
                self.anim_idx = (self.anim_idx + 1) % 2
                self.counter  = 0

        # Score +1 mỗi 7 frame
        if not self.is_dead and self.counter % 7 == 6:
            self.score += 1

        self._update_mask()

    # ── Getters ───────────────────────────────────────────────

    def get_sprite(self) -> pygame.Surface:
        return self._get_current_sprite()

    def get_rect(self) -> pygame.Rect:
        margin_x = max(6, self.w // 5)
        margin_y = max(5, self.h // 7)
        return pygame.Rect(
            self.x + margin_x,
            self.y + margin_y,
            self.w - margin_x * 2,
            self.h - margin_y * 2,
        )


class Obstacle:
    """Chướng ngại vật: xương rồng (nhỏ/lớn/đôi) hoặc chim."""

    def __init__(self, speed: float, sprites: SpriteLoader,
                 force_type: str | None = None, ptera_y: int | None = None):
        self.sprites  = sprites
        self.speed    = speed
        self.counter  = 0
        self.anim_idx = 0

        if force_type == 'ptera':
            self._init_bird(speed, ptera_y)
        elif force_type == 'cactus':
            r = random.random()
            if r < 0.6:
                self._init_small_cactus(speed)
            else:
                self._init_big_cactus(speed)
        else:
            r = random.random()
            if r < 0.45:
                self._init_small_cactus(speed)
            elif r < 0.65:
                self._init_big_cactus(speed)
            elif r < 0.82:
                self._init_double_cactus(speed)
            else:
                self._init_bird(speed)

    def _init_small_cactus(self, speed: float):
        v            = random.randint(0, 5)
        self.type_   = "cactus_small"
        self.variant = v
        self.sprite  = self.sprites.cactus_small(v)
        self.w       = self.sprite.get_width()
        self.h       = self.sprite.get_height()
        self.x       = SCREEN_W + 20
        self.y       = SCREEN_H - GROUND_Y_OFFSET - self.h
        self.mask    = pygame.mask.from_surface(self.sprite)

    def _init_big_cactus(self, speed: float):
        v            = random.randint(0, 4)
        sp           = self.sprites.cactus_big(v)
        self.type_   = "cactus_big"
        self.variant = v
        self.sprite  = sp
        self.w       = sp.get_width()
        self.h       = sp.get_height()
        self.x       = SCREEN_W + 20
        self.y       = SCREEN_H - GROUND_Y_OFFSET - self.h
        self.mask    = pygame.mask.from_surface(sp)

    def _init_double_cactus(self, speed: float):
        v   = random.randint(0, 2)
        sp  = self.sprites.cactus_small(v)
        sp2 = self.sprites.cactus_small((v + 1) % 3)
        w   = sp.get_width() + sp2.get_width() + 2
        h   = max(sp.get_height(), sp2.get_height())
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.blit(sp,  (0, h - sp.get_height()))
        surf.blit(sp2, (sp.get_width() + 2, h - sp2.get_height()))
        self.type_   = "cactus_double"
        self.variant = v
        self.sprite  = surf
        self.w       = w
        self.h       = h
        self.x       = SCREEN_W + 20
        self.y       = SCREEN_H - GROUND_Y_OFFSET - h
        self.mask    = pygame.mask.from_surface(surf)

    def _init_bird(self, speed: float, ptera_y: int | None = None):
        self.type_   = "bird"
        self.variant = 0
        self.sprite  = self.sprites.ptera(0)
        self.w       = self.sprites.ptera_w()
        self.h       = self.sprites.ptera_h()
        if ptera_y is not None:
            self.y = ptera_y
        else:
            ground_top = SCREEN_H - GROUND_Y_OFFSET
            heights = [
                ground_top - self.h,
                ground_top - self.h - 45,
                ground_top - self.h - 95,
            ]
            self.y = random.choice(heights)
        self.x    = SCREEN_W + 20
        self.mask = pygame.mask.from_surface(self.sprites.ptera(0))

    def update(self, game_speed: float | None = None):
        # Dùng game_speed thực tế (có boost) nếu được truyền vào,
        # fallback về self.speed (tốc độ lúc spawn) nếu không có.
        self.x       -= (game_speed if game_speed is not None else self.speed)
        self.counter += 1
        if self.type_ == "bird" and self.counter % 10 == 0:
            self.anim_idx = (self.anim_idx + 1) % 2

    def get_sprite(self) -> pygame.Surface:
        if self.type_ == "bird":
            return self.sprites.ptera(self.anim_idx)
        return self.sprite

    def get_mask(self) -> pygame.mask.Mask:
        if self.type_ == "bird":
            return pygame.mask.from_surface(self.sprites.ptera(self.anim_idx))
        return self.mask

    def get_rect(self) -> pygame.Rect:
        margin_x = max(4, self.w // 8)
        margin_y = max(4, self.h // 8)
        return pygame.Rect(
            self.x + margin_x,
            self.y + margin_y,
            self.w - margin_x * 2,
            self.h - margin_y * 2,
        )

    def get_mask_rect(self) -> tuple:
        return self.get_mask(), (self.x + 4, self.y + 4)

    def is_off_screen(self) -> bool:
        return self.x + self.w < 0

    # ── Collision rect (margin riêng cho bird) ─────────────────
    def get_rect(self) -> pygame.Rect:
        if self.type_ == "bird":
            margin_x = max(10, self.w // 4)
            margin_y = max(10, self.h // 3)
        else:
            margin_x = max(4, self.w // 8)
            margin_y = max(4, self.h // 8)
        return pygame.Rect(
            self.x + margin_x,
            self.y + margin_y,
            self.w - margin_x * 2,
            self.h - margin_y * 2,
        )


class Ground:
    def __init__(self, sprites: SpriteLoader):
        self.sprites = sprites
        self.sprite  = sprites.ground_sprite()
        self.w       = sprites.ground_sprite_w
        self.h       = sprites.ground_sprite_h
        self._x1     = 0.0
        self._x2     = float(self.w)
        self.y       = SCREEN_H - GROUND_Y_OFFSET - self.h

    def update(self, speed: float):
        self._x1 -= speed
        self._x2 -= speed
        if self._x1 + self.w < 0:
            self._x1 = self._x2 + self.w
        if self._x2 + self.w < 0:
            self._x2 = self._x1 + self.w

    def draw(self, surface: pygame.Surface):
        surface.blit(self.sprite, (int(self._x1), int(self.y)))
        surface.blit(self.sprite, (int(self._x2), int(self.y)))


class Scoreboard:
    def __init__(self, sprites: SpriteLoader, x: int, y: int):
        self.sprites = sprites
        self.x       = x
        self.y       = y
        self.score   = 0

    def update(self, score: int):
        self.score = score

    def draw(self, surface: pygame.Surface):
        digits = self._extract_digits(self.score)
        ox = 0
        for d in digits:
            surface.blit(self.sprites.digit(d), (self.x + ox, self.y))
            ox += self.sprites.digit(0).get_width()

    def _extract_digits(self, n: int) -> list:
        digits = []
        while len(digits) < 5:
            digits.append(n % 10)
            n //= 10
        digits.reverse()
        return digits


class DinoEnv:
    """Môi trường game hoàn chỉnh."""

    def __init__(self, render: bool = False):
        self.render     = render
        self.game_speed = INIT_SPEED
        self.ground_y   = SCREEN_H - GROUND_Y_OFFSET
        self.points     = 0
        self.obs        = []
        self.obs_timer  = 0
        self.last_obs_right     = 0.0
        self.clouds             = []
        self.nearest            = -1.0
        self.game_over          = False
        self.last_spawn_frame   = 0
        self.next_spawn_at      = 0
        self.last_obstacle_type = None
        self._base_speed        = INIT_SPEED
        self.boost_timer        = 0
        self.boost_cooldown     = 0
        self._dino_speed_factor = 1.0
        self.screen  = None
        self.clock   = None
        self.sprites = None
        self.ground  = None
        self.sounds  = {}

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("Chrome Dino AI")
            self.clock   = pygame.time.Clock()
            self.sprites = SpriteLoader()
            self.ground  = Ground(self.sprites)
            self._load_sounds()
            self._spawn_initial_clouds()
        else:
            self.sprites = SpriteLoader()
            self.ground  = Ground(self.sprites)

    def _load_sounds(self):
        base = os.path.join(os.path.dirname(__file__), "templates")
        for name, fname in [("jump", "jump.wav"),
                            ("die",  "die.wav"),
                            ("checkpoint", "checkPoint.wav")]:
            try:
                self.sounds[name] = pygame.mixer.Sound(os.path.join(base, fname))
            except Exception:
                pass

    def _play_sound(self, name: str):
        s = self.sounds.get(name)
        if s:
            s.play()

    def _spawn_initial_clouds(self):
        for i in range(4):
            self.clouds.append({
                "x":     random.randint(0, SCREEN_W),
                "y":     random.randint(10, SCREEN_H // 4),
                "speed": random.uniform(0.3, 0.8),
            })

    def reset(self) -> np.ndarray:
        self.game_speed         = INIT_SPEED
        self.points             = 0
        self.obs                = []
        self.obs_timer          = 0
        self.last_obs_right     = 0.0
        self.clouds             = []
        self.nearest            = -1.0
        self.game_over          = False
        self.last_spawn_frame   = 0
        self.next_spawn_at      = 0
        self.last_obstacle_type = None
        self._base_speed        = INIT_SPEED
        self.boost_timer        = 0
        self.boost_cooldown     = 0
        if self.ground:
            self.ground._x1 = 0.0
            self.ground._x2 = float(self.ground.w)
        if self.render:
            self._spawn_initial_clouds()
        return self._build_state()

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def step_single(self, dino: Dinosaur, action: int) -> tuple:
        if self.game_over:
            return self._build_state(), 0.0, True, {"points": dino.score}

        self.points += 1
        dino.steps  += 1

        self._base_speed = min(self._base_speed + SPEED_INCREMENT, MAX_SPEED)

        if self.boost_timer > 0:
            self.boost_timer -= 1
            self.game_speed = self._base_speed * 1.5  # x2 cố định
        else:
            self.game_speed = self._base_speed
            self.boost_cooldown -= 1
            if self.boost_cooldown <= 0 and self._base_speed > INIT_SPEED * 0.9:
                if random.random() < 0.005:
                    self.boost_timer = 60               # 1s @ 60fps
                    self.boost_cooldown = random.randint(180, 360)

        self._dino_speed_factor = self.game_speed / self._base_speed

        if action == 0:
            if not dino.is_jumping:
                dino.duck()
        elif action == 1:
            dino.unduck()
            if not dino.is_jumping:
                self._play_sound("jump")
            dino.jump(self._dino_speed_factor)
        else:
            dino.unduck()

        dino.update(self.game_speed, self._base_speed, is_boosting=self.boost_timer > 0)
        self._spawn_obstacle()
        self._update_obstacles()
        self._update_clouds()
        self._check_collision(dino)

        done   = dino.is_dead
        reward = -50.0 if done else 1.0
        if done:
            self.game_over = True
            self._play_sound("die")

        return self._build_state(), reward, done, {"points": dino.score}

    def _spawn_obstacle(self):
        frames_since_spawn = self.points - self.last_spawn_frame
        if frames_since_spawn < self.next_spawn_at:
            return

        # Đợi màn hình đủ thoáng mới spawn tiếp
        if self.obs:
            rightmost_x = max(ob.x + ob.w for ob in self.obs)
            if (SCREEN_W + 20) - rightmost_x < int(SCREEN_W * 0.55):
                return

        # Ptera spawn độc lập, xác suất tăng tuyến tính theo tốc độ
        # Speed 10 → 0%,  speed 18+ → 30%,  không spawn ptera liên tiếp
        spd          = self._base_speed
        ptera_chance = 0.0
        if spd >= 10 and self.last_obstacle_type != 'ptera':
            ptera_chance = min(0.30, (spd - 10) / 27.0)

        if random.random() < ptera_chance:
            self._spawn_ptera()
        else:
            self._spawn_cactus_cluster()

        self.last_spawn_frame = self.points
        self.next_spawn_at    = self._spawn_interval_frames()

    def _spawn_interval_frames(self) -> int:
        base = self._base_speed
        if base <= 8:
            gap_px = random.randint(600, 950)
        elif base <= 10:
            gap_px = random.randint(500, 800)
        elif base <= 13:
            gap_px = random.randint(400, 650)
        else:
            gap_px = random.randint(300, 500)
        return max(20, int(gap_px / base))

    def _spawn_cactus_cluster(self):
        base = self._base_speed
        if base < 8:
            sizes, weights = [1, 2],       [55, 45]
        elif base < 11:
            sizes, weights = [1, 2, 3],    [35, 35, 30]
        else:
            sizes, weights = [1, 2, 3, 4], [25, 30, 25, 20]

        offset = 0
        for _ in range(random.choices(sizes, weights=weights)[0]):
            obs    = Obstacle(base, self.sprites, force_type='cactus')
            obs.x += offset
            offset += obs.w + random.randint(8, 20)
            self.obs.append(obs)

        self.last_obstacle_type = 'cactus'

    def _spawn_ptera(self):
        """
        3 cao độ rõ ràng, mỗi cái yêu cầu hành động khác nhau:
          LOW  (sát đất)   → Duck
          MID  (ngang lưng) → Jump hoặc Duck
          HIGH (trên đầu)  → chạy thẳng (dino lọt qua dưới)
        Trọng số dịch dần về HIGH khi tốc độ tăng.
        """
        ground_top = SCREEN_H - GROUND_Y_OFFSET
        obs        = Obstacle(self._base_speed, self.sprites, force_type='ptera')
        ph         = obs.h
        dino_h     = self.sprites.dino_h

        low_y  = ground_top - ph - 5               # phải duck
        mid_y  = ground_top - ph - (dino_h // 2)   # jump hoặc duck
        high_y = ground_top - ph - dino_h - 25     # chạy thẳng lọt qua

        spd = self._base_speed
        if spd < 13:
            weights = [40, 45, 15]
        elif spd < 18:
            weights = [25, 40, 35]
        else:
            weights = [15, 30, 55]

        obs.y = random.choices([low_y, mid_y, high_y], weights=weights)[0]
        obs.x = SCREEN_W + 20
        self.obs.append(obs)
        self.last_obstacle_type = 'ptera'

    def _update_obstacles(self):
        for ob in self.obs:
            ob.update(self.game_speed)   # ← truyền tốc độ thực tế (có boost)
        self.obs = [ob for ob in self.obs if not ob.is_off_screen()]

    def _update_clouds(self):
        for c in self.clouds:
            c["x"] -= self.game_speed * 0.3
        self.clouds = [c for c in self.clouds if c["x"] + 90 > 0]
        if random.random() < 0.003 and len(self.clouds) < 6:
            self.clouds.append({
                "x":     SCREEN_W,
                "y":     random.randint(5, SCREEN_H // 4),
                "speed": random.uniform(0.3, 0.8),
            })

    def _check_collision(self, dino: Dinosaur):
        d_rect = dino.get_rect()
        for ob in self.obs:
            if d_rect.colliderect(ob.get_rect()):
                dino.is_dead = True
                return

    def _build_state(self) -> np.ndarray:
        state = [0.0] * STATE_SIZE
        if self.obs:
            ob       = self.obs[0]
            state[0] = max(0.0, ob.x - 80) / SCREEN_W
            state[1] = (self.ground_y - ob.y - ob.h) / self.ground_y
            state[2] = ob.w / 60.0
            state[3] = 1.0 if ob.type_ == "bird" else 0.0
            if ob.type_ == "bird":
                state[4] = (self.ground_y - ob.y) / self.ground_y
        state[5] = self.game_speed / MAX_SPEED
        return np.array(state, dtype=np.float32)

    def _draw_bg(self):
        self.screen.fill(BG_COLOR)

    def _draw_clouds(self):
        cloud_sprite = self.sprites.cloud_sprite()
        for c in self.clouds:
            self.screen.blit(cloud_sprite, (int(c["x"]), int(c["y"])))

    def _draw_ground(self, ground: Ground):
        ground.update(self.game_speed)
        ground.draw(self.screen)

    def _draw_dino(self, dino: Dinosaur):
        self.screen.blit(dino.get_sprite(), (int(dino.x), int(dino.y)))

    def _draw_obstacles(self):
        for ob in self.obs:
            self.screen.blit(ob.get_sprite(), (int(ob.x), int(ob.y)))

    def _draw_scores(self, current_score: int, hi_score: int):
        hi_label = self.sprites.hi_label()
        x = SCREEN_W * 0.65
        y = 8
        self.screen.blit(hi_label[0], (x, y))
        x += hi_label[0].get_width()
        self.screen.blit(hi_label[1], (x, y))
        x += hi_label[1].get_width() + 4
        for d in Scoreboard(self.sprites, 0, 0)._extract_digits(hi_score):
            self.screen.blit(self.sprites.digit(d), (x, y))
            x += self.sprites.digit(0).get_width()

        x2 = SCREEN_W * 0.88
        for d in Scoreboard(self.sprites, 0, 0)._extract_digits(current_score):
            self.screen.blit(self.sprites.digit(d), (x2, y))
            x2 += self.sprites.digit(0).get_width()

    def _draw_game_over(self):
        go_sprite = self.sprites.gameover_sprite()
        re_sprite = self.sprites.restart_sprite()
        go_x = (SCREEN_W - go_sprite.get_width()) // 2
        go_y = int(SCREEN_H * 0.35)
        re_x = (SCREEN_W - re_sprite.get_width()) // 2
        re_y = int(SCREEN_H * 0.52)
        self.screen.blit(go_sprite, (go_x, go_y))
        self.screen.blit(re_sprite, (re_x, re_y))

    def _draw_ai_info(self, ai_info: dict):
        font_name = pygame.font.get_default_font()
        font  = pygame.font.SysFont(font_name, 14, bold=False)
        small = pygame.font.SysFont(font_name, 12, bold=False)

        name       = ai_info.get("name", "AI")
        generation = ai_info.get("generation", 0)
        speed      = ai_info.get("speed", 0)
        boosting   = self.boost_timer > 0

        panel_w = 200
        panel_h = 80 if boosting else 62
        panel_x = SCREEN_W - panel_w - 12
        panel_y = 36

        pygame.draw.rect(self.screen, (40, 40, 40),
                         (panel_x, panel_y, panel_w, panel_h), border_radius=6)
        pygame.draw.rect(self.screen, (70, 70, 70),
                         (panel_x, panel_y, panel_w, panel_h), 2, border_radius=6)

        name_color = (255, 140, 60) if boosting else (100, 220, 100)
        self.screen.blit(font.render(name, True, name_color),
                         (panel_x + 8, panel_y + 8))

        speed_color = (255, 200, 80) if boosting else (180, 180, 180)
        self.screen.blit(small.render(f"Speed: {speed:.1f}", True, speed_color),
                         (panel_x + 8, panel_y + 30))

        act        = ai_info.get("last_action", -1)
        act_labels = {0: "DUCK", 1: "JUMP", 2: "RUN "}
        self.screen.blit(small.render(f"Act: {act_labels.get(act, '---')}",
                                      True, (255, 200, 80)),
                         (panel_x + 100, panel_y + 30))

        self.screen.blit(small.render(f"Steps: {ai_info.get('steps', 0)}",
                                      True, (180, 180, 180)),
                         (panel_x + 100, panel_y + 48))

        if boosting:
            boost_left = f"BOOST: {self.boost_timer}f"
            self.screen.blit(small.render(boost_left, True, (255, 80, 80)),
                             (panel_x + 8, panel_y + 64))

    def render_frame(self, dino: Dinosaur, hi_score: int = 0,
                     ai_info: dict | None = None):
        if self.screen is None:
            return

        self._draw_bg()
        self._draw_clouds()
        self._draw_ground(self.ground)
        self._draw_dino(dino)
        self._draw_obstacles()
        self._draw_scores(dino.score, hi_score)

        if ai_info:
            self._draw_ai_info(ai_info)
        if self.game_over:
            self._draw_game_over()

        if self.boost_timer > 0:
            boost_factor = self.game_speed / self._base_speed
            color = (255, 80, 80) if boost_factor >= 1.8 else (255, 180, 0)
            boost_font = pygame.font.SysFont("Arial", 28, bold=True)
            label = boost_font.render(f"x{boost_factor:.1f}", True, color)
            self.screen.blit(label, (12, 12))

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.render and self.screen is not None:
            pygame.quit()