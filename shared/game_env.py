# ============================================================
#  game_env.py  –  Môi trường game Chrome Dino hoàn chỉnh
#  PATCHED v2: smoother jump physics + object scale fixes
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

    PATCH v2 LOG:
    - [Scale]  DINO_SCALE giảm 0.50 → 0.45 (dino nhỏ hơn, cân xứng hơn)
    - [Jump]   GRAVITY giảm 1.156 → 0.703, JUMP_VEL 17.33 → 14.76
               T = 42 frames (0.70s), H = 155px → mượt, floaty tự nhiên
    - [Jump]   Vẫn dùng float sub-pixel _y_float để tránh bước nhảy thô
    """

    def __init__(self, sprites: SpriteLoader):
        self.sprites  = sprites
        self.x        = SCREEN_W // 15

        # ground_y tính từ sprite height đã scale
        self.ground_y = SCREEN_H - GROUND_Y_OFFSET - sprites.dino_h
        self.y        = self.ground_y

        # Float sub-pixel tracking – giữ nguyên từ v1
        self._y_float = float(self.ground_y)
        self._vel_y   = 0.0

        self.w      = sprites.dino_w
        self.h      = sprites.dino_h
        self.duck_w = sprites.duck_w
        self.duck_h = sprites.duck_h

        # ── v2: GRAVITY=0.703, JUMP_VEL=14.76 (từ config) ──
        self.jump_vel = JUMP_VEL   # 14.76 px/frame
        self.gravity  = GRAVITY    # 0.703 px/frame²

        self.is_jumping = False
        self.is_ducking = False
        self.is_dead    = False
        self.steps      = 0
        self.score      = 0
        self.counter    = 0
        self.anim_idx   = 0

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

    def jump(self):
        if not self.is_jumping and not self.is_ducking:
            self._vel_y     = -self.jump_vel   # -14.76 (âm = lên)
            self.is_jumping = True

    def duck(self):
        if not self.is_jumping and not self.is_ducking:
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

    def update(self):
        if self.is_jumping:
            # v2: gravity 0.703 → thay đổi vel ≤ 0.7px/frame → cực mượt
            self._vel_y   += self.gravity       # +0.703/frame
            self._y_float += self._vel_y
            self.y         = int(self._y_float)

            if self._y_float >= self.ground_y:
                self._y_float   = float(self.ground_y)
                self.y          = self.ground_y
                self.is_jumping = False
                self._vel_y     = 0.0

        # Animation frames
        self.counter += 1
        if self.is_jumping:
            self.anim_idx = 0
        elif self.counter % 5 == 4:
            self.anim_idx = (self.anim_idx + 1) % 2

        # Score +1 mỗi 7 frame
        if not self.is_dead and self.counter % 7 == 6:
            self.score += 1

        self._update_mask()

    # ── Getters ───────────────────────────────────────────────

    def get_sprite(self) -> pygame.Surface:
        return self._get_current_sprite()

    def get_rect(self) -> pygame.Rect:
        # Shrink collision rect 20% horizontal, 10% vertical → forgiving hitbox
        margin_x = max(4, self.w // 5)
        margin_y = max(2, self.h // 10)
        return pygame.Rect(
            self.x + margin_x,
            self.y + margin_y,
            self.w - margin_x * 2,
            self.h - margin_y * 2,
        )

    def reset(self):
        """Reset dino to initial state for a new game."""
        self.y          = self.ground_y
        self._y_float  = float(self.ground_y)
        self._vel_y    = 0.0
        self.w         = self.sprites.dino_w
        self.h         = self.sprites.dino_h
        self.is_jumping = False
        self.is_ducking = False
        self.is_dead    = False
        self.steps      = 0
        self.score      = 0
        self.counter    = 0
        self.anim_idx   = 0


# ─────────────────────────────────────────────────────────────
#  SpriteLoader PATCH – thêm vào shared/renderer.py
#  (scale cactus & bird bằng OBSTACLE_SCALE / BIRD_SCALE)
#
#  class SpriteLoader:
#      def __init__(self):
#          # ── Dino (giữ nguyên patch v1, chỉ DINO_SCALE thay đổi) ──
#          raw_dino   = pygame.image.load("dino.png").convert_alpha()
#          frame_w    = raw_dino.get_width() // 5
#          frame_h    = raw_dino.get_height()
#          self.dino_w = int(frame_w * DINO_SCALE)   # 0.45 → ~40px
#          self.dino_h = int(frame_h * DINO_SCALE)   # 0.45 → ~42px
#          self._dino_frames = [
#              pygame.transform.smoothscale(
#                  raw_dino.subsurface(pygame.Rect(i*frame_w, 0, frame_w, frame_h)),
#                  (self.dino_w, self.dino_h)
#              ) for i in range(5)
#          ]
#          # duck
#          raw_duck       = pygame.image.load("dino_ducking.png").convert_alpha()
#          duck_fw        = raw_duck.get_width() // 2
#          duck_fh        = raw_duck.get_height()
#          self.duck_w    = int(duck_fw * DINO_SCALE)
#          self.duck_h    = int(duck_fh * DINO_SCALE)
#          self._duck_frames = [
#              pygame.transform.smoothscale(
#                  raw_duck.subsurface(pygame.Rect(i*duck_fw, 0, duck_fw, duck_fh)),
#                  (self.duck_w, self.duck_h)
#              ) for i in range(2)
#          ]
#
#          # ── PATCH v2: Cactus scale ──────────────────────────
#          # Áp OBSTACLE_SCALE=0.80 lên tất cả cactus sprite
#          raw_cactus_small = pygame.image.load("cactus_small.png").convert_alpha()
#          # (giả sử 3 frame ngang)
#          cs_fw = raw_cactus_small.get_width() // 3
#          cs_fh = raw_cactus_small.get_height()
#          self._cactus_small_frames = [
#              pygame.transform.smoothscale(
#                  raw_cactus_small.subsurface(pygame.Rect(i*cs_fw, 0, cs_fw, cs_fh)),
#                  (int(cs_fw * OBSTACLE_SCALE), int(cs_fh * OBSTACLE_SCALE))
#              ) for i in range(3)
#          ]
#
#          raw_cactus_big = pygame.image.load("cactus_big.png").convert_alpha()
#          cb_fw = raw_cactus_big.get_width() // 5
#          cb_fh = raw_cactus_big.get_height()
#          self._cactus_big_frames = [
#              pygame.transform.smoothscale(
#                  raw_cactus_big.subsurface(pygame.Rect(i*cb_fw, 0, cb_fw, cb_fh)),
#                  (int(cb_fw * OBSTACLE_SCALE), int(cb_fh * OBSTACLE_SCALE))
#              ) for i in range(5)
#          ]
#
#          # ── PATCH v2: Bird (ptera) scale ────────────────────
#          raw_ptera    = pygame.image.load("ptera.png").convert_alpha()
#          pt_fw        = raw_ptera.get_width() // 2
#          pt_fh        = raw_ptera.get_height()
#          self._ptera_w = int(pt_fw * BIRD_SCALE)   # ~64px
#          self._ptera_h = int(pt_fh * BIRD_SCALE)   # ~56px
#          self._ptera_frames = [
#              pygame.transform.smoothscale(
#                  raw_ptera.subsurface(pygame.Rect(i*pt_fw, 0, pt_fw, pt_fh)),
#                  (self._ptera_w, self._ptera_h)
#              ) for i in range(2)
#          ]
#
#      def cactus_small(self, v): return self._cactus_small_frames[v]
#      def cactus_big(self, v):   return self._cactus_big_frames[v]
#      def ptera(self, i):        return self._ptera_frames[i]
#      def ptera_w(self):         return self._ptera_w
#      def ptera_h(self):         return self._ptera_h
# ─────────────────────────────────────────────────────────────


class Obstacle:
    """Chướng ngại vật: xương rồng (nhỏ/lớn/đôi) hoặc chim."""

    def __init__(self, speed: float, sprites: SpriteLoader,
                 force_type: str | None = None, ptera_y: int | None = None):
        self.sprites  = sprites
        self.speed    = speed
        self.counter  = 0
        self.anim_idx = 0
        self.counted  = False  # tránh reward trùng cho cùng 1 obstacle

        if force_type == 'ptera':
            self._init_bird(speed, ptera_y)
        elif force_type == 'cactus_small':
            self._init_small_cactus(speed)
        elif force_type == 'cactus_big':
            self._init_big_cactus(speed)
        elif force_type == 'cactus_double':
            self._init_double_cactus(speed)
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

    # ── Init helpers ──────────────────────────────────────────

    def _init_small_cactus(self, speed: float):
        n        = len(self.sprites._cactus_small)
        v        = random.randint(0, n - 1)
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
                ground_top - self.h,           # sát đất
                ground_top - self.h - 45,      # giữa
                ground_top - self.h - 95,      # cao
            ]
            self.y = random.choice(heights)
        self.x    = SCREEN_W + 20
        self.mask = pygame.mask.from_surface(self.sprites.ptera(0))

    # ── Update & getters ──────────────────────────────────────

    def update(self):
        self.x       -= self.speed
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
        if self.type_ == "bird":
            # Chim có cánh trong suốt → margin lớn hơn
            margin_x = max(6, self.w // 4)
            margin_y = max(6, self.h // 5)
        else:
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

    def __init__(self, render: bool = False, spawn_policy=None):
        from shared.spawn_policy import SpawnPolicy
        self.render     = render
        self.spawn_policy = spawn_policy or SpawnPolicy()
        self.game_speed = INIT_SPEED
        self.ground_y   = SCREEN_H - GROUND_Y_OFFSET
        self.points     = 0
        self.obs        = []
        self.obs_timer  = 0
        self.last_obs_right     = 0.0
        self.clouds             = []
        self.nearest            = -1.0
        self.game_over          = False
        self._cleared_count     = 0
        self.last_spawn_frame   = 0
        self.next_spawn_at      = 0
        self.last_obstacle_type = None
        self._base_speed        = INIT_SPEED
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
            font_name = pygame.font.get_default_font()
            self._font_14 = pygame.font.SysFont(font_name, 14, bold=False)
            self._font_12 = pygame.font.SysFont(font_name, 12, bold=False)
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

    def reset(self, dino: "Dinosaur | None" = None) -> np.ndarray:
        self.game_speed         = INIT_SPEED
        self.points             = 0
        self.obs                = []
        self.obs_timer          = 0
        self.last_obs_right     = 0.0
        self.clouds             = []
        self.nearest            = -1.0
        self.game_over          = False
        self._cleared_count     = 0
        self.last_spawn_frame   = 0
        self.next_spawn_at      = 0
        self.last_obstacle_type = None
        self._base_speed        = INIT_SPEED
        if self.ground:
            self.ground._x1 = 0.0
            self.ground._x2 = float(self.ground.w)
        if self.render:
            self._spawn_initial_clouds()

        if dino is not None:
            dino.reset()

        return self._build_state(dino)

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def step_single(self, dino: Dinosaur, action: int) -> tuple:
        if self.game_over:
            return self._build_state(dino), 0.0, True, {"points": dino.score}

        self.points += 1
        dino.steps  += 1

        if action == 0:
            if not dino.is_jumping:
                dino.duck()
        elif action == 1:
            dino.unduck()
            if not dino.is_jumping:
                self._play_sound("jump")
            dino.jump()
        else:
            dino.unduck()

        dino.update()
        cleared = [ob for ob in self.obs if ob.x + ob.w < dino.x and not ob.counted]
        for ob in cleared:
            ob.counted = True
        if cleared:
            self._cleared_count += len(cleared)
        self._update_obstacles()
        self._update_clouds()
        self._check_collision(dino)

        self._base_speed = min(self._base_speed + SPEED_INCREMENT, MAX_SPEED)
        self.game_speed = self._base_speed
        for ob in self.obs:
            ob.speed = self.game_speed

        self._spawn_obstacle()

        done   = dino.is_dead
        if done:
            reward = -1.0
        else:
            bonus = 0.0
            for ob in cleared:
                if ob.type_ == "bird":
                    # Chim cách ground bao xa → quyết định cách vượt
                    bird_ground_dist = self.ground_y - ob.y - ob.h
                    if bird_ground_dist < 40:        # chim sát đất → phải nhảy
                        bonus += 25.0
                    elif bird_ground_dist < 80:      # chim giữa → nhảy hoặc cúi
                        bonus += 15.0
                    else:                            # chim cao → nên cúi
                        bonus += 30.0 if dino.is_ducking else 5.0
                else:
                    bonus += 10.0                    # xương rồng = +10
            reward = 0.1 + bonus
        if done:
            self.game_over = True
            self._play_sound("die")

        return self._build_state(dino), reward, done, {"points": dino.score}

    # ── Spawn logic – Chrome Dino style ───────────────────────
    # Nguyên tắc Google gốc:
    #   • Tối đa 1 nhóm chướng ngại vật trên màn hình
    #   • Nhóm mới chỉ spawn khi nhóm cũ sắp ra khỏi màn (x < ~200px)
    #   • Khoảng cách tính theo pixel, không phải frame cố định
    #   • Cluster size tăng dần theo tốc độ

    # ── Spawn logic ─────────────────────────────────────────

    def _spawn_obstacle(self):
         frames_since_spawn = self.points - self.last_spawn_frame
         if frames_since_spawn < self.next_spawn_at:
             return
         if self.obs:
             rightmost_x = max(ob.x + ob.w for ob in self.obs)
             if (SCREEN_W + 20) - rightmost_x < int(SCREEN_W * 0.60):
                 return

         kind = self.spawn_policy.decide_type(
             self.game_speed, self.last_obstacle_type, self.points
         )
         if kind == "ptera":
             self._spawn_ptera()
         else:
             self._spawn_cactus_cluster()

         self.last_spawn_frame = self.points
         self.next_spawn_at    = self.spawn_policy.decide_interval(self.game_speed)

    # FIX: xóa _spawn_interval_frames — dead code, không được gọi ở đâu cả.
    # decide_interval từ policy đã thay thế hoàn toàn.

    def _spawn_cactus_cluster(self):
         from shared.config import JUMP_VEL, GRAVITY

         pattern = self.spawn_policy.decide_pattern(self.game_speed)
         spd     = self.game_speed
         start_x = SCREEN_W + 20

         # ── Tính jump_px: quãng đường ngang dino đi trong 1 chu kỳ nhảy ──
         # T_jump = 2 * JUMP_VEL / GRAVITY (frames)
         # jump_px = T_jump * spd (pixels)
         jump_px = int((2 * JUMP_VEL / GRAVITY) * spd)

         # ── FIX react_frames: GIẢM theo speed (khó hơn ở tốc độ cao) ──
         # Trước đây: slow=2, vfast=7 → chain DỄ hơn ở tốc độ cao (ngược!)
         # Sau fix:   slow=7, vfast=3 → chain KHÓ hơn ở tốc độ cao (đúng)
         react_frames = self.spawn_policy.decide_chain_react_frames(spd)
         react_px     = int(spd * react_frames)

         # ── FIX chain_gap: khoảng cách giữa right edge c_n và left edge c_{n+1} ──
         # chain_gap = jump_px + react_px
         #   - jump_px đảm bảo c_{n+1} chưa đến khi dino còn đang nhảy qua c_n
         #   - react_px là buffer sau khi landing → AI có react_frames frame để nhảy tiếp
         # Trước đây: landing_gap = spd * react_frames (thiếu jump_px!)
         #   → c2 gần như dính vào c1 → chain2 trông như 1 cactus đôi rộng
         chain_gap = jump_px + react_px

         def make_cactus(x_offset: int = 0) -> "Obstacle":
             """
             FIX: dùng decide_cactus_size để kích thước cactus tăng theo speed.
             Trước đây: random 60/40 split hardcode trong Obstacle.__init__.
             """
             size = self.spawn_policy.decide_cactus_size(spd)
             if size == "small":
                 ob = Obstacle(spd, self.sprites, force_type='cactus_small')
             elif size == "big":
                 ob = Obstacle(spd, self.sprites, force_type='cactus_big')
             else:  # double
                 ob = Obstacle(spd, self.sprites, force_type='cactus_double')
             ob.x = start_x + x_offset
             return ob

         def make_bird(x_offset: int = 0,
                       height: str = "any") -> "Obstacle":
             tmp = Obstacle(spd, self.sprites, force_type='ptera')
             ground_top = SCREEN_H - GROUND_Y_OFFSET
             tmp.y = self.spawn_policy.decide_bird_height(
                 ground_top, tmp.h, force=height
             )
             tmp.x = start_x + x_offset
             return tmp

         # ── single ──────────────────────────────────────────────
         if pattern == "single":
             self.obs.append(make_cactus())

         # ── chain2: 2 lần nhảy riêng biệt ──────────────────────
         # gap = chain_gap (= jump_px + react_px):
         #   → c2 chỉ đến khi dino đã landing từ c1 và có react_frames để quyết định
         elif pattern == "chain2":
             c1 = make_cactus(0)
             c2 = make_cactus(c1.w + chain_gap)
             self.obs += [c1, c2]

         # ── chain3: 3 lần nhảy riêng biệt ──────────────────────
         elif pattern == "chain3":
             c1 = make_cactus(0)
             c2 = make_cactus(c1.w + chain_gap)
             c3 = make_cactus(c1.w + chain_gap + c2.w + chain_gap)
             self.obs += [c1, c2, c3]

         # ── jump_duck: nhảy qua cactus → cúi né chim ───────────
         # Chim đặt cách c1 ≥ jump_px để dino đã landing trước khi chim đến.
         # Dino đứng → chim đâm; Dino cúi → chim bay qua.
         elif pattern == "jump_duck":
             c1          = make_cactus(0)
             # Thêm buffer nhỏ sau jump_px để dino có thời gian unduck (nếu đang bay)
             gap_to_bird = jump_px + random.randint(15, 45)
             # "mid" hoặc "high" → dino cúi mới qua được
             bird_height = random.choice(["mid", "high"])
             b1          = make_bird(c1.w + gap_to_bird, height=bird_height)
             self.obs   += [c1, b1]

         # ── duck_jump: cúi né chim → nhảy cactus ───────────────
         # Chim "high" → chạm đầu dino đứng → buộc phải cúi.
         # Sau chim đủ thời gian unduck + react → mới đến cactus.
         elif pattern == "duck_jump":
             b1 = make_bird(0, height="high")
             # Cần unduck (~ 2-3 frame) + react_frames trước khi nhảy
             unduck_frames = 3
             gap_to_cactus = int(spd * (unduck_frames + react_frames + 5))
             c1            = make_cactus(b1.w + gap_to_cactus)
             self.obs     += [b1, c1]

         # ── sandwich: cactus → chim → cactus (nhảy, cúi, nhảy) ─
         elif pattern == "sandwich":
             c1            = make_cactus(0)
             gap_to_bird   = jump_px + random.randint(10, 30)
             bird_height   = random.choice(["mid", "high"])
             b1            = make_bird(c1.w + gap_to_bird, height=bird_height)
             # Cactus cuối: sau chim, dino cần unduck + react frames
             unduck_frames = 3
             gap_to_c2     = int(spd * (unduck_frames + react_frames + 5))
             x_c2          = c1.w + gap_to_bird + b1.w + gap_to_c2
             c2            = make_cactus(x_c2)
             self.obs     += [c1, b1, c2]

         self.last_obstacle_type = 'cactus'

    def _spawn_ptera(self):
         tmp        = Obstacle(self.game_speed, self.sprites, force_type='ptera')
         ground_top = SCREEN_H - GROUND_Y_OFFSET
         ptera_y    = self.spawn_policy.decide_bird_height(ground_top, tmp.h)
         obs        = Obstacle(self.game_speed, self.sprites,
                               force_type='ptera', ptera_y=ptera_y)
         self.obs.append(obs)
         self.last_obstacle_type = 'ptera'

    def _update_obstacles(self):
        for ob in self.obs:
            ob.update()
        self.obs = [ob for ob in self.obs if not ob.is_off_screen()]

    def _update_clouds(self):
        for c in self.clouds:
            c["x"] -= c["speed"] * (self.game_speed / INIT_SPEED) * 0.3
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

    # ── State vector ──────────────────────────────────────────

    def _build_state(self, dino: "Dinosaur | None" = None) -> np.ndarray:
        state = [0.0] * STATE_SIZE  # 13

        sorted_obs = sorted(self.obs, key=lambda ob: ob.x)
        ref_x = dino.x if dino else 80

        # [0-3] + [4-7] 2 obstacles: dist, height, is_bird, bird_y
        for i in range(2):
            if i < len(sorted_obs):
                ob = sorted_obs[i]
                base = i * 4
                dist_px = ob.x - ref_x
                state[base + 0] = max(0.0, min(1.0, dist_px / SCREEN_W))
                state[base + 1] = min(1.0, ob.h / 160.0)
                state[base + 2] = 1.0 if ob.type_ == "bird" else 0.0
                state[base + 3] = ob.y / max(self.ground_y, 1) if ob.type_ == "bird" else 0.0

        # [8] Tốc độ
        state[8] = self.game_speed / MAX_SPEED

        # [9-10] Dino state
        if dino is not None:
            state[9] = 1.0 if dino.is_jumping else 0.0
            state[10] = 1.0 if dino.is_ducking else 0.0

        # [11] Jump safety cho obs1
        if len(sorted_obs) >= 1 and self.game_speed > 0.1:
            ob1 = sorted_obs[0]
            dist_px = ob1.x - ref_x
            jump_dur = 2 * JUMP_VEL / GRAVITY
            max_jump = JUMP_VEL**2 / (2 * GRAVITY)
            if ob1.h < max_jump:
                t = dist_px / self.game_speed
                state[11] = max(0.0, min(1.0, (jump_dur - t) / jump_dur))

        # [12] Dino vertical velocity (âm = đang bay lên, dương = đang rơi)
        if dino is not None and dino.is_jumping:
            state[12] = dino._vel_y / JUMP_VEL
        else:
            state[12] = 0.0

        return np.array(state, dtype=np.float32)

    # ── Render helpers ────────────────────────────────────────

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

    def _score_digits(self, n: int) -> list:
        digits = []
        while len(digits) < 5:
            digits.append(n % 10)
            n //= 10
        digits.reverse()
        return digits

    def _draw_scores(self, current_score: int, hi_score: int):
        hi_label = self.sprites.hi_label()
        x = SCREEN_W * 0.65
        y = 8
        self.screen.blit(hi_label[0], (x, y))
        x += hi_label[0].get_width()
        self.screen.blit(hi_label[1], (x, y))
        x += hi_label[1].get_width() + 4
        for d in self._score_digits(hi_score):
            self.screen.blit(self.sprites.digit(d), (x, y))
            x += self.sprites.digit(0).get_width()

        x2 = SCREEN_W * 0.88
        for d in self._score_digits(current_score):
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
        font      = self._font_14
        small     = self._font_12

        name       = ai_info.get("name", "AI")
        generation = ai_info.get("generation", 0)
        speed = ai_info.get("speed", 0)

        panel_w = 200
        panel_h = 62
        panel_x = SCREEN_W - panel_w - 12
        panel_y = 36

        pygame.draw.rect(self.screen, (40, 40, 40),
                         (panel_x, panel_y, panel_w, panel_h), border_radius=6)
        pygame.draw.rect(self.screen, (70, 70, 70),
                         (panel_x, panel_y, panel_w, panel_h), 2, border_radius=6)

        self.screen.blit(font.render(name, True, (100, 220, 100)),
                         (panel_x + 8, panel_y + 8))

        self.screen.blit(small.render(f"Speed: {speed:.1f}", True, (180, 180, 180)),
                         (panel_x + 8, panel_y + 30))

        act        = ai_info.get("last_action", -1)
        act_labels = {0: "DUCK", 1: "JUMP", 2: "RUN "}
        self.screen.blit(small.render(f"Act: {act_labels.get(act, '---')}",
                                      True, (255, 200, 80)),
                         (panel_x + 100, panel_y + 30))

        self.screen.blit(small.render(f"Steps: {ai_info.get('steps', 0)}",
                                      True, (180, 180, 180)),
                         (panel_x + 100, panel_y + 48))

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

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.render and self.screen is not None:
            pygame.quit()