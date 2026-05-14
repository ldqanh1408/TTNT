# ============================================================
#  config.py  –  Hằng số dùng chung cho toàn bộ dự án
#  PATCHED v2: smoother jump + better object scaling
# ============================================================

# --- Màn hình ---
SCREEN_W        = 1200
SCREEN_H        = 450
FPS             = 60
GROUND_Y_OFFSET = 20

# --- Màu nền Chrome Dino thực ---
BG_COLOR = (235, 235, 235)

# --- Khủng long ---
DINO_X = 900 // 15

# ─── SCALE: Dino ──────────────────────────────────────────
# Sprite gốc: 88×93px mỗi frame
# DINO_SCALE=0.45 → content ~40×42px  (nhỏ hơn bản trước 0.50)
# Ratio dino_h / cactus_small_h ≈ 42/32 = 1.31 (gần Google gốc 1.34)
DINO_SCALE = 0.45   # ← giảm từ 0.50 → dino nhỏ hơn, cân xứng hơn

# ─── SCALE: Obstacle ──────────────────────────────────────
# Cactus sprite gốc: small ~35px, big ~90px (quá to với screen 450px)
# OBSTACLE_SCALE=0.80 → small ~28px, big ~72px (hợp lý hơn)
OBSTACLE_SCALE      = 0.80   # ← THÊM MỚI: SpriteLoader dùng khi load cactus

# Bird (ptera) sprite gốc: ~92×80px → quá to
# BIRD_SCALE=0.70 → ~64×56px (vừa mắt hơn)
BIRD_SCALE          = 0.70   # ← THÊM MỚI: SpriteLoader dùng khi load ptera

# ─── JUMP PHYSICS (float, smooth) ─────────────────────────
#
# Vấn đề cũ (v1):
#   GRAVITY=1.156, JUMP_VEL=17.33 → T=30 frames (0.5s), H=130px
#   → bước y lớn/không đều ở đỉnh → cảm giác "sượng"
#   → H=130px chưa đủ vượt cactus_big (90px) khi dino đứng trên đất
#
# Phân tích đúng:
#   Dino đứng ở y_ground ≈ 450-20-42 = 388
#   Cactus_big đỉnh ở   y_top  ≈ 450-20-90 = 340
#   Cần clearance: dino_bottom phải lên tới ≤ 340 → cần H ≥ 48px trên ground
#   Nhưng thực ra dino cần VƯỢT QUA cactus, tức đỉnh dino phải lên cao hơn
#   đỉnh cactus_big. Với dino cao 42px, cần dino.y ≤ 340 → phải nhảy ≥ 48px
#   Tuy nhiên 130px là đủ, vấn đề chính là jump quá nhanh và cảm giác sượng.
#
# Fix v2 – mục tiêu:
#   H = 155px  (vượt thoải mái cactus_big 72px sau scale)
#   T = 42 frames (0.70s tại 60fps) → floaty, tự nhiên
#
#   Công thức:   gravity  = 8H / T²  = 8×155 / 1764 ≈ 0.703
#                jump_vel = gravity × T/2 = 0.703 × 21 ≈ 14.76
#
#   Kết quả: y thay đổi ≤ 1.7px ở đỉnh (cực mượt), đường cong parabola rõ
#
GRAVITY  = 1.1     # px/frame² — mượt, arc tự nhiên
JUMP_VEL = 18.5    # px/frame  — clearance ~148px

# ─── Tốc độ game ──────────────────────────────────────────
INIT_SPEED      = 6.0
SPEED_INCREMENT = 0.005
MAX_SPEED       = 16.0

# ─── Chướng ngại vật (kích thước tham khảo – sau scale) ───
# Giá trị này chỉ dùng để tham khảo / fallback; kích thước thật
# lấy từ sprite sau khi SpriteLoader áp OBSTACLE_SCALE / BIRD_SCALE
CACTUS_TYPES = [
    {"w": 40, "h": 32},   # small (sau scale 0.80)
    {"w": 72, "h": 72},   # big   (sau scale 0.80)
    {"w": 72, "h": 72},
]
BIRD_W = 64   # sau BIRD_SCALE=0.70
BIRD_H = 56

# ─── State / Action ───────────────────────────────────────
# 12D: 1 cụm obstacle × 5 + bird_x + jump_safety + speed + 4 dino state
#   [0-4]   cụm: dist, total_width, max_height, has_bird, bird_y
#   [5]     bird_x / SCREEN_W (0 nếu ko có chim)
#   [6]     jump_safety
#   [7]     game_speed / MAX_SPEED
#   [8]     dino_y / ground_y
#   [9]     dino_vel_y / jump_vel
#   [10]    is_jumping (0/1)
#   [11]    is_ducking (0/1)
STATE_SIZE  = 12
ACTION_SIZE = 3

# ─── Training ─────────────────────────────────────────────
MAX_STEPS_PER_EPISODE = 20_000
RENDER_EVERY          = 50