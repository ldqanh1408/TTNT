# ============================================================
#  renderer.py  –  Load & slice sprite sheets từ thư mục media
#  PATCHED v3: sizes calibrated for 1200×450 screen
#
#  Target sizes (px):
#    Dino standing : 60 × 65    (~14% screen height)
#    Cactus small  : 32 × ~40   (easy jump)
#    Cactus big    : 48 × ~80   (tall, needs good jump)
#    Bird (ptera)  : 80 × 55
# ============================================================
import os
import pygame
from PIL import Image

_MEDIA_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _pil_to_pygame(pil_img: Image.Image) -> pygame.Surface:
    mode = pil_img.mode
    size = pil_img.size
    raw  = pil_img.tobytes("raw", mode)
    return pygame.image.frombuffer(raw, size, mode)


def _load_sheet(filename: str, nx: int, ny: int,
                scalex: int = -1, scaley: int = -1) -> list:
    pil_img = Image.open(os.path.join(_MEDIA_DIR, filename)).convert("RGBA")
    w, h = pil_img.size
    sw = w // nx
    sh = h // ny
    frames = []
    for row in range(ny):
        for col in range(nx):
            frame = pil_img.crop((col * sw, row * sh,
                                  (col + 1) * sw, (row + 1) * sh))
            if scalex != -1 or scaley != -1:
                frame = frame.resize(
                    (scalex if scalex != -1 else sw,
                     scaley if scaley != -1 else sh),
                    Image.LANCZOS
                )
            frames.append(_pil_to_pygame(frame))
    return frames


def _load_cactus_sprites(filename: str, target_h: int) -> list:
    """Detect cactus sprites by green pixel, scale to target_h, NEAREST (no blur)."""
    pil_img = Image.open(os.path.join(_MEDIA_DIR, filename)).convert("RGBA")
    px = pil_img.load()
    w, h = pil_img.size

    def is_cactus(r, g, b, a):
        return a > 30 and g > max(r, b) + 20

    col_has = []
    for x in range(w):
        for y in range(h):
            if is_cactus(*px[x, y]):
                col_has.append(x)
                break

    sprites = []
    in_sprite, start_x = False, 0
    for x in range(w):
        if x in col_has:
            if not in_sprite:
                start_x = x
                in_sprite = True
        else:
            if in_sprite:
                x1, x2 = start_x, x - 1
                top, bot = h, 0
                for sx in range(x1, x2 + 1):
                    for sy in range(h):
                        if is_cactus(*px[sx, sy]):
                            top = min(top, sy)
                            bot = max(bot, sy)
                pad = 1
                crop = pil_img.crop((
                    max(0, x1 - pad), max(0, top - pad),
                    min(w, x2 + pad + 1), min(h, bot + pad + 1)
                ))
                ratio = target_h / crop.height
                new_w = int(crop.width * ratio)
                crop = crop.resize((new_w, target_h), Image.NEAREST)
                sprites.append(_pil_to_pygame(crop))
                in_sprite = False
    if in_sprite:
        x1, x2 = start_x, w - 1
        top, bot = h, 0
        for sx in range(x1, x2 + 1):
            for sy in range(h):
                if is_cactus(*px[sx, sy]):
                    top = min(top, sy)
                    bot = max(bot, sy)
        pad = 1
        crop = pil_img.crop((
            max(0, x1 - pad), max(0, top - pad),
            min(w, x2 + pad + 1), min(h, bot + pad + 1)
        ))
        ratio = target_h / crop.height
        new_w = int(crop.width * ratio)
        crop = crop.resize((new_w, target_h), Image.NEAREST)
        sprites.append(_pil_to_pygame(crop))
    return sprites


def load_image(filename: str,
               scalex: int = -1, scaley: int = -1) -> pygame.Surface:
    pil_img = Image.open(os.path.join(_MEDIA_DIR, filename)).convert("RGBA")
    if scalex != -1 or scaley != -1:
        pil_img = pil_img.resize(
            (scalex if scalex != -1 else pil_img.width,
             scaley if scaley != -1 else pil_img.height),
            Image.LANCZOS
        )
    return _pil_to_pygame(pil_img)


class SpriteLoader:

    def __init__(self):
        # ── Dinosaur: native 88×95 → target 70×75 ────────────
        _DW = 70
        _DH = 75
        dino_frames = _load_sheet("dino.png", nx=5, ny=1,
                                  scalex=_DW, scaley=_DH)
        self._dino_stand = dino_frames[0]
        self._dino_run   = [dino_frames[1], dino_frames[2]]
        self._dino_jump  = dino_frames[3]
        self._dino_dead  = dino_frames[4]

        # ── Duck: native 118×95, shorter than standing ───────
        _DUCK_W = int(118 * (_DH / 95))
        _DUCK_H = int(50 * (_DH / 65))
        duck_frames = _load_sheet("dino_ducking.png", nx=2, ny=1,
                                  scalex=_DUCK_W, scaley=_DUCK_H)
        self._dino_duck = duck_frames

        # ── Cactus: pixel-detection → NEAREST scale, no blur ──
        self._cactus_small = _load_cactus_sprites("cacti-small.png",
                                                  target_h=int(_DH * 0.75))
        self._cactus_big   = _load_cactus_sprites("cacti-big.png",
                                                  target_h=int(_DH * 1.05))

        # ── Pterodactyl: native 92×81 → 80×55 ───────────────
        ptera_frames = _load_sheet("ptera.png", nx=2, ny=1,
                                   scalex=80, scaley=55)
        self._ptera = ptera_frames

        # ── Environment ──────────────────────────────────────
        self._cloud   = load_image("cloud.png",  scalex=120, scaley=45)
        self._ground  = load_image("ground.png", scalex=-1,  scaley=-1)
        self._ground_w = self._ground.get_width()
        self._ground_h = self._ground.get_height()

        # ── UI ───────────────────────────────────────────────
        nums = _load_sheet("numbers.png", nx=12, ny=1)
        self._digits   = nums[:10]
        self._hi_label = [nums[10], nums[11]]

        self._gameover = load_image("game_over.png")
        self._restart  = load_image("replay_button.png", scalex=35, scaley=31)

    # ── Dino ─────────────────────────────────────────────────
    def dino_stand(self) -> pygame.Surface: return self._dino_stand
    def dino_run_a(self) -> pygame.Surface: return self._dino_run[0]
    def dino_run_b(self) -> pygame.Surface: return self._dino_run[1]
    def dino_jump(self) -> pygame.Surface:  return self._dino_jump
    def dino_duck_a(self) -> pygame.Surface: return self._dino_duck[0]
    def dino_duck_b(self) -> pygame.Surface: return self._dino_duck[1]
    def dino_dead(self) -> pygame.Surface:   return self._dino_dead

    @property
    def dino_w(self) -> int: return self._dino_stand.get_width()
    @property
    def dino_h(self) -> int: return self._dino_stand.get_height()
    @property
    def duck_w(self) -> int: return self._dino_duck[0].get_width()
    @property
    def duck_h(self) -> int: return self._dino_duck[0].get_height()

    # ── Obstacles ─────────────────────────────────────────────
    def cactus_small(self, variant: int = 0) -> pygame.Surface:
        return self._cactus_small[variant % len(self._cactus_small)]

    def cactus_big(self, variant: int = 0) -> pygame.Surface:
        return self._cactus_big[variant % len(self._cactus_big)]

    def ptera(self, frame: int = 0) -> pygame.Surface:
        return self._ptera[frame % 2]

    def ptera_w(self) -> int: return self._ptera[0].get_width()
    def ptera_h(self) -> int: return self._ptera[0].get_height()

    # ── Environment ───────────────────────────────────────────
    def cloud_sprite(self) -> pygame.Surface: return self._cloud
    def ground_sprite(self) -> pygame.Surface: return self._ground

    @property
    def ground_sprite_w(self) -> int: return self._ground_w
    @property
    def ground_sprite_h(self) -> int: return self._ground_h

    # ── UI ────────────────────────────────────────────────────
    def digit(self, n: int) -> pygame.Surface:
        return self._digits[n % 10]

    def hi_label(self) -> tuple:
        return self._hi_label[0], self._hi_label[1]

    def gameover_sprite(self) -> pygame.Surface: return self._gameover
    def restart_sprite(self) -> pygame.Surface:  return self._restart
