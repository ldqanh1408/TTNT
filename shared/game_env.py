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
    GRAVITY, JUMP_VEL, SCORE_DISTANCE,
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
        self._score_accum = 0.0   # điểm tính theo quãng đường (float, chuẩn Chrome Dino)
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

        # Điểm KHÔNG còn cộng ở đây: trước cộng +1 mỗi 7 frame (cố định, không
        # phản ánh tốc độ). Giờ DinoEnv.step_single cộng theo quãng đường thực
        # (game_speed/SCORE_DISTANCE mỗi frame) — chuẩn Chrome Dino.

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
        self._score_accum = 0.0
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
            self._font_14 = pygame.font.Font(None, 14)
            self._font_12 = pygame.font.Font(None, 12)
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

    def reset(self, dino: "Dinosaur | None" = None,
              start_speed: float | None = None) -> np.ndarray:
        # start_speed: tốc độ khởi đầu episode. None → INIT_SPEED (game thật).
        # Training truyền giá trị ngẫu nhiên để replay buffer phủ hết dải tốc
        # độ — nếu không, dino chỉ từng train ở speed ~6-13 (episode kết thúc
        # trước khi tốc độ kịp tăng) và phản xạ ở tốc độ cao là vùng chưa học.
        if start_speed is None:
            base = INIT_SPEED
        else:
            base = float(min(max(start_speed, INIT_SPEED), MAX_SPEED))
        self.game_speed         = base
        self._base_speed        = base
        # points là bộ đếm frame điều khiển spawn (chim, pattern phức tạp) —
        # KHÔNG phải điểm số (score là dino.score). Offset theo tốc độ để
        # episode tốc-độ-cao có spawn variety giống late-game thật.
        self.points             = int((base - INIT_SPEED) / SPEED_INCREMENT)
        self.obs                = []
        self.obs_timer          = 0
        self.last_obs_right     = 0.0
        self.clouds             = []
        self.nearest            = -1.0
        self.game_over          = False
        self._cleared_count     = 0
        self.last_spawn_frame   = self.points
        self.next_spawn_at      = 0
        self.last_obstacle_type = None
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

        was_on_ground = not dino.is_jumping  # snapshot trước khi action thay đổi trạng thái
        was_ducking   = dino.is_ducking

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

        # Score-based acceleration: speed climbs faster as agent scores higher.
        # At score 0:   factor=1.0  (same as before)
        # At score 200: factor=1.5  (50% faster acceleration)
        # At score 400: factor=2.0  (twice as fast)
        # At score 800: factor=3.0  (aggressive ramp for strong agents)
        score_factor = 1.0 + dino.score / 400.0
        self._base_speed = min(self._base_speed + SPEED_INCREMENT * score_factor, MAX_SPEED)
        self.game_speed = self._base_speed
        for ob in self.obs:
            ob.speed = self.game_speed

        # ── Điểm theo quãng đường (chuẩn Chrome Dino) ──
        # Mỗi frame dino đi được game_speed pixel → cộng game_speed/SCORE_DISTANCE
        # điểm. Tốc độ tăng dần → điểm tăng nhanh dần (thay vì +1/7 frame cố định).
        if not dino.is_dead:
            dino._score_accum += self.game_speed / SCORE_DISTANCE
            dino.score = int(dino._score_accum)

        self._spawn_obstacle()

        done   = dino.is_dead
        if done:
            reward = -25.0
        else:
            bonus = 0.0
            for ob in cleared:
                if ob.type_ == "bird":
                    bird_ground_dist = self.ground_y - ob.y - ob.h
                    if bird_ground_dist < 40:        # sát đất → phải nhảy
                        bonus += 12.0
                    elif bird_ground_dist < 80:      # giữa → đứng/cúi an toàn, nhảy nguy hiểm
                        if dino.is_jumping:
                            bonus += 2.0             # sống sót dù sai hành động
                        else:
                            bonus += 12.0            # đúng: ở mặt đất
                    else:                            # cao → không được nhảy
                        if dino.is_jumping:
                            bonus += -8.0            # SAI: nhảy vào vùng nguy hiểm
                        else:
                            bonus += 12.0            # đúng: ở mặt đất
                else:
                    bonus += 12.0                    # xương rồng

            # ── Penalty cho hành động không cần thiết ──
            action_penalty = 0.0

            # Phạt nhảy khi không có chướng ngại cần nhảy trong tầm
            if action == 1 and was_on_ground:
                jump_dur  = 2.0 * JUMP_VEL / GRAVITY
                jump_dist = jump_dur * self.game_speed * 1.3
                needs_jump = False
                for ob in self.obs:
                    if ob.x > dino.x:
                        dist = ob.x - dino.x
                        if dist < jump_dist:
                            if ob.type_ != 'bird':
                                needs_jump = True     # xương rồng → cần nhảy
                            else:
                                bbd = self.ground_y - ob.y - ob.h
                                if bbd < 40:         # low bird → cần nhảy
                                    needs_jump = True
                            break
                if not needs_jump:
                    action_penalty -= 0.5

            # Phạt nhỏ khi chọn JUMP khi đã đang trên không (spam)
            if action == 1 and not was_on_ground:
                action_penalty -= 0.02

            # Phạt nhỏ khi bắt đầu cúi không cần thiết (không có mid bird /
            # tường chim nào cần cúi ở phía trước)
            if action == 0 and not was_ducking and was_on_ground:
                if not self._duck_needed_ahead(dino):
                    action_penalty -= 0.15

            # Phạt duy trì cúi khi không còn mid bird / tường chim nào gần
            # Trước đây chỉ phạt lúc bắt đầu cúi → dino có thể cúi mãi không unduck
            if dino.is_ducking:
                if not self._duck_needed_ahead(dino):
                    action_penalty -= 0.06

            reward = 0.002 + bonus + action_penalty
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
        from shared.config import JUMP_VEL, GRAVITY
        frames_since_spawn = self.points - self.last_spawn_frame
        if frames_since_spawn < self.next_spawn_at:
            return

        spd     = self.game_speed
        spawn_x = SCREEN_W + 20

        # Gap check theo cactus — nhóm mới KHÔNG được tới khi dino còn đang trên
        # không vì cú nhảy qua nhóm cũ (dino không thể nhảy 2 lần trong 1 cú nhảy).
        # min_clear_px phải ≥ jump_px (quãng đường 1 chu kỳ nhảy) + buffer phản ứng.
        # BUG cũ: công thức 220 + spd*15 tụt xuống DƯỚI jump_px khi speed > ~12
        #   (vd speed 20: 220+300=520px < jump_px 672px) → nhóm cactus kế tiếp
        #   spawn ép dino chết khi vẫn đang bay → đúng "spawn buộc ép dino chết".
        cactus_obs = [ob for ob in self.obs if ob.type_ != 'bird']
        if cactus_obs:
            rightmost_cactus = max(ob.x + ob.w for ob in cactus_obs)
            jump_px          = int((2 * JUMP_VEL / GRAVITY) * spd)
            min_clear_px     = jump_px + int(spd * 8)   # +8 frame để landing & phản ứng
            if spawn_x - rightmost_cactus < min_clear_px:
                return

        # Gap check theo bird — ngăn chướng ngại kế tiếp spawn quá gần sau chim.
        #
        # BUG ĐÃ SỬA: case "mid" trước đây dùng bird_min_gap = rb.w + spd*6
        #   (≈ 14-20px-frame) << jump_px (≈ 34 frame). Công thức này GIẢ ĐỊNH
        #   dino sẽ CÚI con chim — nhưng dino hoàn toàn có thể NHẢY con chim.
        #   Khi cactus spawn chỉ cách chim ~spd*6, nó lọt gọn vào TRONG tầm
        #   một cú nhảy → dino nhảy MỘT phát vượt qua cả chim lẫn cactus →
        #   không bao giờ phải cúi. Đây chính là lỗi "quá gần xương rồng".
        #
        # FIX: dino được tự do nhảy mọi con chim, nên chướng ngại kế tiếp
        #   PHẢI cách >= jump_px + buffer ở MỌI độ cao chim — để dù dino
        #   nhảy con chim thì cũng đã đáp đất xong trước chướng kế tiếp,
        #   không thể gom hai chướng vào một cú nhảy. (Combo "ép cúi" chặt
        #   chẽ là việc của _spawn_cactus_cluster, không phải gap check này.)
        bird_obs = [ob for ob in self.obs if ob.type_ == 'bird']
        if bird_obs:
            jump_px  = int((2 * JUMP_VEL / GRAVITY) * spd)
            react_px = int(spd * 6)
            rb        = max(bird_obs, key=lambda ob: ob.x + ob.w)
            rb_right  = rb.x + rb.w
            bird_min_gap = jump_px + react_px
            if spawn_x - rb_right < bird_min_gap:
                return

        kind = self.spawn_policy.decide_type(
            spd, self.last_obstacle_type, self.points
        )
        if kind == "ptera":
            self._spawn_ptera()
            spawned_pattern = None
        else:
            spawned_pattern = self._spawn_cactus_cluster()

        interval = self.spawn_policy.decide_interval(spd)
        gap_ok = kind == "cactus" and spawned_pattern not in ("jump_duck", "duck_jump", "sandwich")
        if gap_ok and self._maybe_spawn_gap_birds(interval):
            self.last_obstacle_type = 'ptera'
        self.last_spawn_frame = self.points
        self.next_spawn_at    = interval

    def _maybe_spawn_gap_birds(self, gap_frames: int) -> bool:
        """Spawn 1-2 chim trong gap giữa 2 cụm cactus với combo hợp lệ, đảm bảo passable."""
        from shared.config import JUMP_VEL, GRAVITY
        spd    = self.game_speed
        gap_px = gap_frames * spd
        if gap_px < 280 or not self.obs or self.points < 700:
            return False

        # jump_px: quãng đường obstacle đi trong 1 chu kỳ nhảy đầy đủ của dino
        jump_px  = int((2 * JUMP_VEL / GRAVITY) * spd)
        react_px = int(spd * 4)  # 4 frame buffer để AI nhận diện và phản ứng

        # Bird PHẢI đến sau khi dino đã landing từ bước nhảy qua obstacle trước.
        # Nếu không: duck() bị ignore khi is_jumping=True → impossible case.
        min_safe_dist = jump_px + react_px
        if gap_px < min_safe_dist + 80:
            return False

        prob = min(0.50, 0.15 + spd * 0.015)
        if random.random() > prob:
            return False

        ground_top  = SCREEN_H - GROUND_Y_OFFSET
        rightmost_x = max(ob.x + ob.w for ob in self.obs)
        bird_h      = self.sprites.ptera_h()
        bird_w      = self.sprites.ptera_w()

        def _bird_y(height_str: str) -> int:
            offsets = {"low": 10, "mid": 50, "high": 120}
            return ground_top - bird_h - offsets[height_str]

        def _make_bird(x: int, height_str: str) -> "Obstacle":
            b   = Obstacle(spd, self.sprites, force_type='ptera', ptera_y=_bird_y(height_str))
            b.x = x
            return b

        b1_x = rightmost_x + min_safe_dist

        # Combo spawn khi gap đủ lớn — mỗi combo đảm bảo LUÔN có thể vượt qua:
        #   (low,  high): nhảy b1 sát đất → chạy qua b2 cao (không cần hành động)
        #   (high, low):  chạy qua b1 cao → nhảy b2 sát đất
        #   (mid,  high): cúi b1 giữa    → chạy qua b2 cao
        #   (high, mid):  chạy qua b1    → cúi b2 giữa
        _COMBOS = [
            ("low",  "high"),
            ("high", "low"),
            ("mid",  "high"),
            ("high", "mid"),
        ]

        use_combo = gap_px > min_safe_dist + jump_px + 150 and random.random() < 0.38

        if use_combo:
            h1, h2 = random.choice(_COMBOS)
            b1 = _make_bird(b1_x, h1)

            # Spacing b1→b2: đủ để AI thực hiện xong hành động b1 và phản ứng b2
            if h1 == "low":      # dino nhảy → phải đáp xuống đất trước khi b2 đến
                spacing = jump_px + react_px
            elif h1 == "mid":    # dino cúi  → phải đứng dậy trước khi b2 đến
                spacing = bird_w + int(spd * 5)
            else:                # high      → chạy qua, cần bird width + nhỏ buffer
                spacing = bird_w + int(spd * 3)

            b2_x     = b1_x + spacing
            max_b2_x = rightmost_x + gap_px - 80  # không lấn vào vùng nhóm tiếp theo

            if b2_x <= max_b2_x:
                self.obs += [b1, _make_bird(b2_x, h2)]
                return True
            # Không đủ chỗ cho b2 → chỉ spawn b1
            self.obs.append(b1)
            return True

        # Single bird tại vị trí an toàn
        bird_x = rightmost_x + max(min_safe_dist, int(gap_px * 0.45))
        bird_y = self.spawn_policy.decide_bird_height(ground_top, bird_h, game_speed=spd)
        b      = Obstacle(spd, self.sprites, force_type='ptera', ptera_y=bird_y)
        b.x    = bird_x
        self.obs.append(b)
        return True

    # FIX: xóa _spawn_interval_frames — dead code, không được gọi ở đâu cả.
    # decide_interval từ policy đã thay thế hoàn toàn.

    def _spawn_cactus_cluster(self):
         from shared.config import JUMP_VEL, GRAVITY

         pattern = self.spawn_policy.decide_pattern(self.game_speed)
         # Score < ~100: hạn chế pattern có chim
         if self.points < 700 and pattern in ("jump_duck", "duck_jump", "sandwich"):
             pattern = "single"
         # duck_stretch cần CÚI LIÊN TỤC → unlock muộn hơn (score ~900)
         if self.points < 900 and pattern == "duck_stretch":
             pattern = "single"
         # weave_stretch & mixed_stretch cần adaptive behavior → unlock ~1000
         if self.points < 1000 and pattern in ("weave_stretch", "mixed_stretch"):
             pattern = "single"
         spd     = self.game_speed
         start_x = SCREEN_W + 20

         # ── Tính jump_px: quãng đường ngang dino đi trong 1 chu kỳ nhảy ──
         # T_jump = 2 * JUMP_VEL / GRAVITY (frames)
         # jump_px = T_jump * spd (pixels)
         jump_px = int((2 * JUMP_VEL / GRAVITY) * spd)

         # ── react_frames TĂNG theo speed: slow=3 → vfast=9 ──
         # Speed thấp → ít frame (gap nhỏ, ép phản xạ); speed cao → nhiều frame (gap lớn, đủ thở)
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

         def make_wall(x_offset: int = 0):
             """Tường chim 3 con tách rời — đứng/nhảy đều CHẾT, chỉ CÚI mới sống.
             Returns (birds_list, wall_width)."""
             birds, ww = self._create_bird_wall(start_x + x_offset, spd)
             return birds, ww

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

         # ── jump_duck: nhảy qua cactus → CÚI dưới tường chim ───
         # Tường chim ép cúi tuyệt đối (nhảy = chết). gap_to_wall = jump_px +
         # spd*6: dù dino nhảy c1 sớm hay muộn, nó luôn đáp đất xong (dư >=9
         # frame) RỒI tường mới tới → luôn kịp cúi. Nhảy tường = chết chắc.
         elif pattern == "jump_duck":
             c1          = make_cactus(0)
             gap_to_wall = jump_px + int(spd * 6)
             w1_birds, _ = make_wall(c1.w + gap_to_wall)
             self.obs   += [c1] + w1_birds

         # ── duck_jump: CÚI dưới tường chim → nhảy cactus ───────
         # Tường lo việc "ép cúi"; gap sau tường chỉ cần đủ rộng để dino
         # unduck+jump cactus an toàn (spd*14 ≈ thừa thời gian phản ứng).
         elif pattern == "duck_jump":
             w1_birds, ww = make_wall(0)
             gap_to_cactus = int(spd * 14)
             c1            = make_cactus(ww + gap_to_cactus)
             self.obs     += w1_birds + [c1]

         # ── sandwich: cactus → tường chim → cactus (nhảy, CÚI, nhảy)
         # Buộc CÚI giữa hai cú nhảy: nhảy c1 → đáp đất → cúi tường → nhảy c2.
         elif pattern == "sandwich":
             c1            = make_cactus(0)
             gap_to_wall   = jump_px + int(spd * 6)
             w1_birds, ww  = make_wall(c1.w + gap_to_wall)
             gap_to_c2     = int(spd * 14)
             x_c2          = c1.w + gap_to_wall + ww + gap_to_c2
             c2            = make_cactus(x_c2)
             self.obs     += [c1] + w1_birds + [c2]

         # ── duck_stretch: cactus → dàn chim ngang → cactus ──────
         # Nhảy c1 → đáp đất → CÚI LIÊN TỤC qua 2-3 chim mid trải ngang
         # → đứng dậy → nhảy c2. Khác sandwich: thay tường dọc bằng dàn
         # chim NGANG cùng độ cao — buộc duy trì cúi lâu, không chỉ cúi
         # 1 lần rồi thôi. Khoảng cách giữa các chim đủ rộng để trông
         # như đàn thật, đủ hẹp để không đứng lên được.
         elif pattern == "duck_stretch":
             c1 = make_cactus(0)
             # Symmetric side gaps → birds CENTERED between the two cacti.
             # gap_side = max(landing_gap, prep_gap) — whichever dominates.
             # landing_gap: jump_px + spd*4 (dino lands from c1, reacts, ducks)
             # prep_gap:    spd*10 (dino stands up, reacts, jumps c2)
             side_gap     = max(jump_px + int(spd * 4), int(spd * 10))
             stretch_birds = self._create_bird_stretch(
                 start_x + c1.w + side_gap, spd
             )
             bird_span = (max(b.x + b.w for b in stretch_birds) -
                          min(b.x for b in stretch_birds)) if stretch_birds else 0
             x_c2      = c1.w + side_gap + bird_span + side_gap
             c2        = make_cactus(x_c2)
             self.obs += [c1] + stretch_birds + [c2]

         # ── weave_stretch: cactus → low→high→low birds → cactus ──
         # Nhảy c1 → NHẢY low bird → CHẠY/CÚI high bird → NHẢY low bird → nhảy c2.
         # Ép dino liên tục chuyển đổi jump→ground→jump, không được nhảy high bird.
         elif pattern == "weave_stretch":
             c1 = make_cactus(0)
             side_gap = max(jump_px + int(spd * 4), int(spd * 10))
             weave_birds = self._create_weave_stretch(
                 start_x + c1.w + side_gap, spd
             )
             bird_span = (max(b.x + b.w for b in weave_birds) -
                          min(b.x for b in weave_birds)) if weave_birds else 0
             x_c2 = c1.w + side_gap + bird_span + side_gap
             c2 = make_cactus(x_c2)
             self.obs += [c1] + weave_birds + [c2]

         # ── mixed_stretch: cactus → random-height birds → cactus ──
         # Nhảy c1 → thích ứng từng con (low=nhảy, mid=cúi, high=chạy) → nhảy c2.
         # Không theo pattern cố định, buộc dino đọc state và phản ứng linh hoạt.
         elif pattern == "mixed_stretch":
             c1 = make_cactus(0)
             side_gap = max(jump_px + int(spd * 4), int(spd * 10))
             mixed_birds = self._create_mixed_stretch(
                 start_x + c1.w + side_gap, spd
             )
             bird_span = (max(b.x + b.w for b in mixed_birds) -
                          min(b.x for b in mixed_birds)) if mixed_birds else 0
             x_c2 = c1.w + side_gap + bird_span + side_gap
             c2 = make_cactus(x_c2)
             self.obs += [c1] + mixed_birds + [c2]

         # Patterns có bird → set ptera để ngăn standalone bird ngay sau
         if pattern in ("jump_duck", "duck_jump", "sandwich", "duck_stretch",
                        "weave_stretch", "mixed_stretch"):
             self.last_obstacle_type = 'ptera'
         else:
             self.last_obstacle_type = 'cactus'
         return pattern

    def _safe_bird_spawn_x(self) -> int:
        """x spawn cho chim/đàn chim: nếu phía trước còn obstacle thì lùi đủ
        xa để dino KỊP đáp đất rồi mới vào tư thế cúi/né — gap an toàn
        >= jump_px + buffer phản ứng (thiếu là rơi vào impossible case)."""
        from shared.config import JUMP_VEL, GRAVITY
        spawn_x = SCREEN_W + 20
        if self.obs:
            rightmost = max(ob.x + ob.w for ob in self.obs)
            jump_px   = (2.0 * JUMP_VEL / GRAVITY) * self.game_speed
            react_px  = self.game_speed * 6.0
            spawn_x   = max(spawn_x, int(rightmost + jump_px + react_px))
        return spawn_x

    def _spawn_ptera(self):
        """Spawn chim bay theo COMBO — single / wall / stream / triple."""
        spd   = self.game_speed
        combo = self.spawn_policy.decide_bird_combo(spd, self.points)

        if combo == "stream":
            self._spawn_bird_stream()
            return

        if combo == "triple":
            self._spawn_bird_triple_line()
            return

        spawn_x = self._safe_bird_spawn_x()

        if combo == "wall":
            birds, _ = self._create_bird_wall(spawn_x, spd)
            self.obs.extend(birds)
        else:  # "single" — 1 con chim low/mid/high thường
            tmp        = Obstacle(spd, self.sprites, force_type='ptera')
            ground_top = SCREEN_H - GROUND_Y_OFFSET
            ptera_y    = self.spawn_policy.decide_bird_height(
                ground_top, tmp.h, game_speed=spd
            )
            obs = Obstacle(spd, self.sprites,
                           force_type='ptera', ptera_y=ptera_y)
            obs.x = spawn_x
            self.obs.append(obs)

        self.last_obstacle_type = 'ptera'

    def _spawn_bird_stream(self):
        """Combo 'lướt chim' NGANG — 2-3 con chim thẳng hàng vùng DUCK.

        Sub-styles (chọn ngẫu nhiên):
          pair   — 2 chim, gap spd×24 (nhanh gọn, ép cúi ngắn)
          trio   — 3 chim, gap spd×24 (bền bỉ, ép cúi dài)
          wide   — 2-3 chim, gap spd×40 (dàn trải, duck lâu)
          rising — 2-3 chim, gap spd×28, độ cao tăng dần 42→50→58

        Tất cả chim trong vùng DUCK (h_off 40-65, bbd 40-65) → DUCK hint.
        Gap ≥ spd×24 đảm bảo 2-bird span > clear zone, không thể nhảy vượt.
        """
        spd        = self.game_speed
        ground_top = SCREEN_H - GROUND_Y_OFFSET
        bird_h     = self.sprites.ptera_h()
        bird_w     = self.sprites.ptera_w()

        style = random.choice(["pair", "trio", "wide", "rising"])

        if style == "pair":
            n_birds = 2
            gap = max(int(spd * 24), 200)
        elif style == "trio":
            n_birds = 3
            gap = max(int(spd * 24), 200)
        elif style == "wide":
            n_birds = random.choice([2, 3])
            gap = max(int(spd * 40), 400)
        else:  # rising
            n_birds = random.choice([2, 3])
            gap = max(int(spd * 28), 250)

        x_cursor = self._safe_bird_spawn_x()
        for i in range(n_birds):
            if style == "rising":
                h_off = 42 + i * 8 + random.randint(-3, 3)
            else:
                h_off = 47 + random.randint(-5, 5)
            y = ground_top - bird_h - h_off
            b = Obstacle(spd, self.sprites, force_type='ptera', ptera_y=y)
            b.x        = x_cursor
            b.counter  = random.randint(0, 9)
            b.anim_idx = random.randint(0, 1)
            self.obs.append(b)
            jitter = random.randint(-int(gap * 0.18), int(gap * 0.18))
            x_cursor += bird_w + gap + jitter
        self.last_obstacle_type = 'ptera'

    def _spawn_bird_triple_line(self):
        """Combo 3 CHIM THẲNG HÀNG NGANG — KHÔNG THỂ nhảy vượt, PHẢI cúi.

        3 con chim ở CÙNG độ cao DUCK (h_off=47, bbd=47), dàn ngang với
        gap = max(spd×30, 280). Tổng span 3 chim ≥ 800px (spd=6) đến
        ≥ 2040px (spd=30) — vượt xa clear zone ~133-665px → nhảy bao
        nhiêu cũng không qua hết, buộc phải CÚI.

        KHÔNG có y-jitter → thẳng hàng tuyệt đối, trông như hàng rào ngang.
        x-jitter ±12% → tự nhiên, không cứng nhắc.
        """
        spd        = self.game_speed
        ground_top = SCREEN_H - GROUND_Y_OFFSET
        bird_h     = self.sprites.ptera_h()
        bird_w     = self.sprites.ptera_w()

        # h_off=47 → bbd=47 → DUCK hint, dino đứng va chạm, cúi lọt
        h_off = 47
        y = ground_top - bird_h - h_off

        # gap = max(spd*30, 280): rộng hơn trio (spd*24, min 200)
        # → span 3-bird luôn >> clear zone ở mọi tốc độ
        gap = max(int(spd * 30), 280)

        x_cursor = self._safe_bird_spawn_x()
        for i in range(3):
            b = Obstacle(spd, self.sprites, force_type='ptera', ptera_y=y)
            b.x        = x_cursor
            b.counter  = random.randint(0, 9)
            b.anim_idx = random.randint(0, 1)
            self.obs.append(b)
            jitter = random.randint(-int(gap * 0.12), int(gap * 0.12))
            x_cursor += bird_w + gap + jitter
        self.last_obstacle_type = 'ptera'

    def _create_bird_wall(self, center_x: float, spd: float) -> list:
        """Tạo TƯỜNG CHIM 3 con TÁCH RỜI — tự nhiên nhưng vẫn ép cúi.

        Hình học (với sprite 64×56, margin_y=11 → mỗi con collision 34px):
          Bird H (cao nhất):   sprite_y=179  → rect (190,224)
          Bird M (giữa):       sprite_y=253  → rect (264,298)
          Bird L (thấp nhất):  sprite_y=327  → rect (338,372)

        Gaps 40px < 61px (dino height) → nhảy luôn bị chặn.
        Dino CÚI top=378 > 372 → hở 6px, luôn sống được.

        Mỗi con lệch x ±10px, lệch pha cánh ngẫu nhiên → đàn tự nhiên.
        Bird L (thấp nhất) đặt GẦN dino nhất → obs1 trong state → DUCK hint.
        """
        ground_top = SCREEN_H - GROUND_Y_OFFSET  # 430
        bird_h = self.sprites.ptera_h()           # 56

        # 3 vị trí y: thấp nhất (gần mid height) → obs1, 2 con cao hơn phía sau
        y_positions = [
            ground_top - bird_h - 47,   # Bird L: offset ~47 → mid range → DUCK hint
            ground_top - bird_h - 121,  # Bird M: offset ~121 → high
            ground_top - bird_h - 195,  # Bird H: offset ~195 → high
        ]

        # Dynamic spacing: flock stretches horizontally with speed for natural look.
        # At spd=6:  spacing=18 → total span ~116px (tight, early game)
        # At spd=14: spacing=42 → total span ~164px (flowing)
        # At spd=20: spacing=60 → total span ~200px (stretched, fast-paced)
        gap = max(14, int(spd * 3.0))
        x_offsets = [-gap, 0, gap]  # lowest bird closest to dino

        birds = []
        for i, y in enumerate(y_positions):
            b = Obstacle(spd, self.sprites, force_type='ptera', ptera_y=y)
            b.x = center_x + x_offsets[i] + random.randint(-4, 4)
            b.counter = random.randint(0, 9)
            b.anim_idx = random.randint(0, 1)
            birds.append(b)

        # Total span: from leftmost x to rightmost right-edge
        wall_width = (max(b.x + b.w for b in birds) -
                      min(b.x for b in birds))
        return birds, wall_width

    def _create_bird_stretch(self, start_x: float, spd: float) -> list:
        """Dàn chim NGANG vùng DUCK — nhiều kiểu spacing khác nhau.

        Sub-styles (chọn ngẫu nhiên):
          pair   — 2 chim, gap spd×24 (nhanh gọn)
          trio   — 3 chim, gap spd×24 (bền bỉ)
          wide   — 2-3 chim, gap spd×40 (dàn trải, duck lâu)
          rising — 2-3 chim, gap spd×28, độ cao tăng dần 42→50→58

        Tất cả chim trong vùng DUCK (h_off 40-65, bbd 40-65) → DUCK hint.
        Gap ≥ spd×24 đảm bảo 2-bird span > clear zone, không thể nhảy vượt.
        """
        ground_top = SCREEN_H - GROUND_Y_OFFSET
        bird_h = self.sprites.ptera_h()

        style = random.choice(["pair", "trio", "wide", "rising"])

        if style == "pair":
            n_birds = 2
            gap = max(int(spd * 24), 200)
        elif style == "trio":
            n_birds = 3
            gap = max(int(spd * 24), 200)
        elif style == "wide":
            n_birds = random.choice([2, 3])
            gap = max(int(spd * 40), 400)
        else:  # rising
            n_birds = random.choice([2, 3])
            gap = max(int(spd * 28), 250)

        birds = []
        x_cursor = start_x
        for i in range(n_birds):
            if style == "rising":
                h_off = 42 + i * 8 + random.randint(-3, 3)
            else:
                h_off = 47 + random.randint(-5, 5)
            y = ground_top - bird_h - h_off
            b = Obstacle(spd, self.sprites, force_type='ptera', ptera_y=y)
            b.x = x_cursor
            b.counter = random.randint(0, 9)
            b.anim_idx = random.randint(0, 1)
            birds.append(b)
            jitter = random.randint(-int(gap * 0.18), int(gap * 0.18))
            x_cursor += b.w + gap + jitter
        return birds

    def _create_weave_stretch(self, start_x: float, spd: float) -> list:
        """Tạo dàn 3 chim xen kẽ độ cao: low → high → low trải ngang giữa 2 cactus.

        Pattern: nhảy c1 → low bird (NHẢY) → high bird (CHẠY/CÚI, ko được nhảy)
        → low bird (NHẢY) → nhảy c2. Ép dino chuyển đổi jump→ground→jump liên tục.

        Heights:
          low  = ground - bird_h - 10  (sát đất, bắt buộc NHẢY)
          high = ground - bird_h - 120 (trên cao, STAND/DUCK an toàn, JUMP va chạm)

        Spacing: jump_px + react_px giữa low→high (dino nhảy low bird xong phải
        đáp đất trước khi high bird tới), react_px giữa high→low (dino chạy qua
        high bird rồi nhảy low bird tiếp).
        """
        from shared.config import JUMP_VEL, GRAVITY
        ground_top = SCREEN_H - GROUND_Y_OFFSET
        bird_h = self.sprites.ptera_h()
        bird_w = self.sprites.ptera_w()

        jump_px = int((2 * JUMP_VEL / GRAVITY) * spd)
        react_px = int(spd * 5)

        low_y = ground_top - bird_h - 10
        high_y = ground_top - bird_h - 120

        # low, high, low — 3 birds forcing jump→ground→jump
        heights = [low_y, high_y, low_y]

        birds = []
        x_cursor = start_x
        for i, y in enumerate(heights):
            b = Obstacle(spd, self.sprites, force_type='ptera', ptera_y=y)
            b.x = x_cursor
            b.counter = random.randint(0, 9)
            b.anim_idx = random.randint(0, 1)
            birds.append(b)

            if i == 0:
                # low→high: jump_px + react_px → dino lands fully before high bird
                spacing = jump_px + react_px
            else:
                # high→low: wide gap so dino runs under high bird, then has
                # ample time to react and initiate a fresh jump for the next low bird
                spacing = max(int(spd * 16), jump_px)
            jitter = random.randint(-int(spacing * 0.12), int(spacing * 0.12))
            x_cursor += bird_w + spacing + jitter

        return birds

    def _create_mixed_stretch(self, start_x: float, spd: float) -> list:
        """Tạo dàn 2-3 chim với độ cao NGẪU NHIÊN [low, mid, high] giữa 2 cactus.

        Không theo pattern cố định — dino phải thích ứng với từng con một.
        Low → phải nhảy. Mid → phải cúi. High → phải chạy/không nhảy.
        """
        from shared.config import JUMP_VEL, GRAVITY
        ground_top = SCREEN_H - GROUND_Y_OFFSET
        bird_h = self.sprites.ptera_h()
        bird_w = self.sprites.ptera_w()

        jump_px = int((2 * JUMP_VEL / GRAVITY) * spd)
        react_px = int(spd * 5)

        height_offsets = {
            "low":  10,   # sát đất → phải NHẢY
            "mid":  47,   # giữa    → phải CÚI
            "high": 120,  # cao     → CHẠY, không nhảy
        }

        n_birds = random.choice([2, 3])
        # Đảm bảo không toàn high (quá dễ) hoặc toàn low (quá khó một chiều)
        height_names = random.choices(["low", "mid", "high"], k=n_birds)

        birds = []
        x_cursor = start_x
        for i, h_name in enumerate(height_names):
            y = ground_top - bird_h - height_offsets[h_name]
            b = Obstacle(spd, self.sprites, force_type='ptera', ptera_y=y)
            b.x = x_cursor
            b.counter = random.randint(0, 9)
            b.anim_idx = random.randint(0, 1)
            birds.append(b)

            # Spacing: nếu bird trước là low (phải nhảy) → cần jump_px để đáp đất
            if i > 0 and height_names[i - 1] == "low":
                spacing = jump_px + react_px
            else:
                # wide gap between non-low birds → natural scattered look
                spacing = max(int(spd * 18), jump_px)
            jitter = random.randint(-int(spacing * 0.15), int(spacing * 0.15))
            x_cursor += bird_w + spacing + jitter

        return birds

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

    def _duck_needed_ahead(self, dino: "Dinosaur") -> bool:
        """Có chướng ngại CẦN CÚI (chim mid hoặc tường chim) ngay trước dino?

        Dùng để KHÔNG phạt dino khi nó cúi đúng lúc — nếu thiếu, dino cúi
        đúng dưới tường chim vẫn bị trừ điểm → triệt tiêu việc 'ép cúi'.
        """
        for ob in self.obs:
            if ob.x <= dino.x:
                continue
            if ob.x - dino.x >= self.game_speed * 12:
                continue
            if ob.type_ == 'bird':
                bbd = self.ground_y - ob.y - ob.h
                if 40 <= bbd < 80:
                    return True
        return False

    # ── State vector ──────────────────────────────────────────

    def _build_state(self, dino: "Dinosaur | None" = None) -> np.ndarray:
        state = [0.0] * STATE_SIZE  # 15

        ref_x = dino.x if dino else 80
        # Chỉ giữ obstacle CHƯA vượt hẳn qua dino. Nếu giữ cả obstacle đã qua,
        # một cactus vừa nhảy xong vẫn nằm trong self.obs (chờ ra khỏi màn) và
        # vì có x nhỏ nhất nên chiếm slot obs1 như "bóng ma" (state[0]=0),
        # đẩy cactus kế tiếp xuống slot obs2. Policy phản ứng theo obs1 → không
        # nhảy cây thứ 2 kịp → chết ngay cụm xương rồng đầu tiên.
        sorted_obs = sorted(
            (ob for ob in self.obs if ob.x + ob.w > ref_x),
            key=lambda ob: ob.x,
        )

        # [0-4] + [5-9] 2 obstacles gần nhất — mỗi obstacle 5 feature
        for i in range(2):
            if i < len(sorted_obs):
                ob = sorted_obs[i]
                base = i * 5
                dist_px = ob.x - ref_x

                # [base+0] time_to_obstacle: frames đến obstacle / 60 (tốc độ-nhận biết)
                # Cũ: dist/SCREEN_W → mù với speed (dist=100px luôn = 0.125 dù speed 8 hay 20)
                # Mới: ở speed 20 dist=100px → 0.083 (gấp gáp), speed 8 → 0.208 (còn thời gian)
                t_frames = max(0.0, dist_px) / max(self.game_speed, 0.1)
                state[base + 0] = min(1.0, t_frames / 60.0)

                # [base+1] chiều cao obstacle
                state[base + 1] = min(1.0, ob.h / 160.0)

                # [base+2] chiều RỘNG obstacle — feature MỚI.
                # cactus_small w=26, cactus_double w=54, cactus_big w=36-81, chim w=80.
                # small & double CÙNG height=56 → nếu thiếu width agent không phân
                # biệt được → nhảy giống nhau → chết ở cây rộng. Cây càng rộng dino
                # phải canh nhảy sao cho đỉnh arc trùm hết bề ngang.
                # Chuẩn hoá /100 → small 0.26, double 0.54, big 0.36-0.81, chim 0.80.
                state[base + 2] = min(1.0, ob.w / 100.0)

                # [base+3] is_bird
                state[base + 3] = 1.0 if ob.type_ == "bird" else 0.0

                # [base+4] action_hint: tín hiệu hành động rõ ràng
                # 0.0=NHẢY | 0.5=CÚI | 1.0=CHẠY QUA (không nhảy)
                if ob.type_ == "bird":
                    bird_bottom_dist = self.ground_y - ob.y - ob.h
                    if bird_bottom_dist < 40:
                        state[base + 4] = 0.0   # chim sát đất → phải nhảy
                    elif bird_bottom_dist < 80:
                        state[base + 4] = 0.5   # chim giữa → phải cúi
                    else:
                        state[base + 4] = 1.0   # chim cao → chạy qua, không nhảy
                else:
                    state[base + 4] = 0.0       # xương rồng → phải nhảy

        # [10] Tốc độ
        state[10] = self.game_speed / MAX_SPEED

        # [11-12] Trạng thái dino
        if dino is not None:
            state[11] = 1.0 if dino.is_jumping else 0.0
            state[12] = 1.0 if dino.is_ducking else 0.0

        # [13] remaining_airtime: thời gian còn lại trên không / jump_dur
        # Cũ: jump_safety (chỉ dành cho obs1, phức tạp, = 0 với chim)
        # Mới: tín hiệu vật lý chung — 1.0=vừa nhảy, 0.5=ở đỉnh, 0.0=sắp đáp/trên đất
        if dino is not None and dino.is_jumping:
            height_above_ground = dino.ground_y - dino._y_float  # > 0 khi đang bay
            a_q  = 0.5 * GRAVITY
            b_q  = dino._vel_y
            c_q  = -height_above_ground
            disc = b_q ** 2 - 4.0 * a_q * c_q
            if disc >= 0:
                sqrt_d = disc ** 0.5
                t1     = (-b_q - sqrt_d) / (2.0 * a_q)
                t2     = (-b_q + sqrt_d) / (2.0 * a_q)
                cands  = [t for t in (t1, t2) if t > 1e-3]
                t_land = min(cands) if cands else 0.0
            else:
                t_land = 0.0
            jump_dur  = 2.0 * JUMP_VEL / GRAVITY
            state[13] = min(1.0, max(0.0, t_land / jump_dur))
        else:
            state[13] = 0.0

        # [14] Vận tốc dọc (âm=lên, dương=xuống)
        if dino is not None and dino.is_jumping:
            state[14] = dino._vel_y / JUMP_VEL
        else:
            state[14] = 0.0

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