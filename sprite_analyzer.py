#!/usr/bin/env python3
"""
Sprite sheet analyzer — auto-detects frames via column/row scanning,
falls back to known grid dimensions for tightly-packed sheets.
"""
from PIL import Image
import json
import os

TEMPLATE_DIR = "/mnt/c/Users/CYBORG/PycharmProjects/PythonProject/shared/templates"
OUTPUT_DIR   = os.path.join(TEMPLATE_DIR, "output")


def detect_bg(img: Image.Image) -> tuple:
    """Sample 4 corners (3x3) to find dominant background RGB."""
    w, h = img.size
    samples = []
    for cx, cy in [(0, 0), (w - 3, 0), (0, h - 3), (w - 3, h - 3)]:
        for dx in range(3):
            for dy in range(3):
                px = img.getpixel((cx + dx, cy + dy))
                samples.append(px[:3])
    return tuple(int(sum(c) / len(samples)) for c in zip(*samples))


def col_is_bg(img: Image.Image, x: int, bg: tuple, tol: int = 40) -> bool:
    """True if every pixel in column x matches background (within tolerance)."""
    for y in range(img.height):
        px = img.getpixel((x, y))
        if len(px) == 4 and px[3] < 30:
            continue
        if any(abs(px[i] - bg[i]) > tol for i in range(3)):
            return False
    return True


def row_is_bg(img: Image.Image, y: int, bg: tuple, tol: int = 40) -> bool:
    """True if every pixel in row y matches background."""
    for x in range(img.width):
        px = img.getpixel((x, y))
        if len(px) == 4 and px[3] < 30:
            continue
        rgb = px[:3]
        if any(abs(rgb[i] - bg[i]) > tol for i in range(3)):
            return False
    return True


def scan_columns(img: Image.Image, bg: tuple) -> list[tuple[int, int]]:
    """Column-scan: return [(x_start, x_end), ...] for each detected frame."""
    frames = []
    in_frame, start = False, 0
    for x in range(img.width):
        empty = col_is_bg(img, x, bg)
        if not empty and not in_frame:
            start, in_frame = x, True
        elif empty and in_frame:
            if x - start > 8:
                frames.append((max(0, start - 1), min(img.width, x + 1)))
            in_frame = False
    if in_frame and img.width - start > 8:
        frames.append((max(0, start - 1), img.width))
    return frames


def scan_rows(img: Image.Image, bg: tuple) -> list[tuple[int, int]]:
    """Row-scan: return [(y_start, y_end), ...] for each row band."""
    bands = []
    in_band, start = False, 0
    for y in range(img.height):
        empty = row_is_bg(img, y, bg)
        if not empty and not in_band:
            start, in_band = y, True
        elif empty and in_band:
            if y - start > 8:
                bands.append((max(0, start - 1), min(img.height, y + 1)))
            in_band = False
    if in_band and img.height - start > 8:
        bands.append((max(0, start - 1), img.height))
    return bands or [(0, img.height)]


def crop_auto(img: Image.Image, bg: tuple) -> list[Image.Image]:
    """Full auto-detection: scan rows, then columns within each row."""
    row_bands = scan_rows(img, bg)
    frames = []
    for ry1, ry2 in row_bands:
        band = img.crop((0, ry1, img.width, ry2))
        cols = scan_columns(band, bg)
        for x1, x2 in cols:
            frames.append(img.crop((x1, ry1, x2, ry2)))
    return frames


def crop_grid(img: Image.Image, nx: int, ny: int) -> list[Image.Image]:
    """Grid-based crop: divide image into nx*ny equal frames."""
    fw, fh = img.width // nx, img.height // ny
    frames = []
    for row in range(ny):
        for col in range(nx):
            frames.append(img.crop((
                col * fw, row * fh,
                (col + 1) * fw, (row + 1) * fh,
            )))
    return frames


def save_frames(frames: list[Image.Image], prefix: str,
                names: list[str]) -> list[dict]:
    """Save frames as PNG, return metadata list."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    for i, frame in enumerate(frames):
        fname = f"{prefix}_{i}.png"
        path = os.path.join(OUTPUT_DIR, fname)
        frame.save(path)
        entry = {
            "name": fname,
            "width": frame.width,
            "height": frame.height,
        }
        if i < len(names):
            entry["sprite_name"] = names[i]
        else:
            entry["sprite_name"] = f"{prefix}_{i}"
        results.append(entry)
    return results


# ─── Known grid fallbacks for tightly-packed sheets ───
GRID_FALLBACK = {
    "dino.png":          (5, 1),
    "dino_ducking.png":  (2, 1),
    "ptera.png":         (2, 1),
    "numbers.png":       (12, 1),
    "red_dino1.png":     (5, 1),
}

# ─── Cactus: use dynamic content bounds ───
CACTUS_SHEETS = {
    "cacti-small.png": 3,
    "cacti-big.png":   5,
}

# ─── Single images: never auto-split ───
SINGLE_IMAGE = {"cloud.png", "ground.png", "game_over.png", "replay_button.png"}


def detect_content_bounds(img: Image.Image) -> tuple[int, int]:
    """Find first and last row with non-background content."""
    bg = img.getpixel((0, 0))[:3]
    top, bot = img.height, 0
    for y in range(img.height):
        for x in range(img.width):
            px = img.getpixel((x, y))
            if len(px) == 4 and px[3] < 30:
                continue
            if any(abs(px[i] - bg[i]) > 30 for i in range(3)):
                top = min(top, y)
                bot = max(bot, y)
    return max(0, top - 1), min(img.height, bot + 2)


def crop_cactus_dynamic(image_path: str, prefix: str) -> list[Image.Image]:
    """Crop cactus: grid-split columns, then trim per-variant content height."""
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    nx = CACTUS_SHEETS[os.path.basename(image_path)]
    sw = w // nx
    frames = []
    for col in range(nx):
        x1, x2 = col * sw, (col + 1) * sw
        strip = img.crop((x1, 0, x2, h))
        y1, y2 = detect_content_bounds(strip)
        frames.append(img.crop((x1, y1, x2, y2)))
    return frames

# ─── Sprite naming ───
SPRITE_NAMES = {
    "dino":  ["dino_idle", "dino_run_1", "dino_run_2", "dino_jump", "dino_dead"],
    "dino_ducking": ["dino_duck_1", "dino_duck_2"],
    "ptera": ["ptera_1", "ptera_2"],
    "cacti-small": [f"cactus_small_{i}" for i in range(6)],
    "cacti-big":   [f"cactus_big_{i}" for i in range(6)],
    "numbers": [f"char_{i}" for i in range(10)] + ["char_H", "char_I"],
    "red_dino1": ["red_dino_idle", "red_dino_run_1", "red_dino_run_2",
                  "red_dino_jump", "red_dino_dead"],
    "cloud": ["cloud"],
    "game_over": ["game_over"],
    "replay_button": ["replay_button"],
    "ground": ["ground"],
}


def process_sprite(filename: str) -> tuple[str, list[dict], str]:
    """Process one sprite file. Returns (filename, frames_list, method_used)."""
    path = os.path.join(TEMPLATE_DIR, filename)
    img = Image.open(path).convert("RGBA")
    prefix = filename.rsplit(".", 1)[0]
    names = SPRITE_NAMES.get(prefix, [])
    method = ""

    if filename in SINGLE_IMAGE:
        frames = [img]
        method = "single image (forced)"
    elif filename in CACTUS_SHEETS:
        frames = crop_cactus_dynamic(path, prefix)
        method = f"dynamic height ({len(frames)} cols, per-variant bounds)"
    elif filename in GRID_FALLBACK:
        # Try auto-detection first
        bg = detect_bg(img)
        auto_frames = crop_auto(img, bg)
        nx, ny = GRID_FALLBACK[filename]
        expected = nx * ny

        if len(auto_frames) == expected:
            frames = auto_frames
            method = "auto (column-scan)"
        else:
            frames = crop_grid(img, nx, ny)
            method = f"grid {nx}x{ny} (auto found {len(auto_frames)}, expected {expected})"
    else:
        # Single image or atlas
        bg = detect_bg(img)
        auto_frames = crop_auto(img, bg)
        if not auto_frames:
            frames = [img]
            method = "single image"
        else:
            frames = auto_frames
            method = f"auto (column/row-scan, {len(auto_frames)} regions)"

    results = save_frames(frames, prefix, names)
    return filename, results, method


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sprite_files = [
        # Multi-frame sprite sheets
        "dino.png",
        "dino_ducking.png",
        "ptera.png",
        "cacti-small.png",
        "cacti-big.png",
        "numbers.png",
        "red_dino1.png",
        "offline-sprite-2x-black.png",
        # Single images
        "cloud.png",
        "ground.png",
        "game_over.png",
        "replay_button.png",
    ]

    all_maps = {}
    total = 0
    warnings = []

    print(f"{'File':<35} {'Frames':>7}  {'Method':<50}")
    print("-" * 95)

    for fname in sprite_files:
        path = os.path.join(TEMPLATE_DIR, fname)
        if not os.path.exists(path):
            warnings.append(f"Not found: {fname}")
            continue
        key, frames, method = process_sprite(fname)
        all_maps[key] = frames
        total += len(frames)
        name_sample = ", ".join(f["sprite_name"] for f in frames[:3])
        if len(frames) > 3:
            name_sample += ", ..."
        print(f"  {fname:<33} {len(frames):>7}  {method:<50}")

    print("-" * 95)
    print(f"  {'TOTAL':<33} {total:>7}  frames")
    print()

    for w in warnings:
        print(f"  WARNING: {w}")

    json_path = os.path.join(OUTPUT_DIR, "spritesheet_map.json")
    with open(json_path, "w") as f:
        json.dump(all_maps, f, indent=2)

    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  Map:    {json_path}")


if __name__ == "__main__":
    main()
