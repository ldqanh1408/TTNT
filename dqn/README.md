# DQN Dino AI — Cấu hình Chi tiết

Deep Q-Network điều khiển Chrome Dino tự động né chướng ngại vật.

## 1. State Vector (12D)

| Index | Tên | Công thức | Ý nghĩa |
|---|---|---|---|
| 0 | **dist** | `(cluster_x - dino.x) / 1200` | Khoảng cách đến cụm (0=sát mặt, 1=xa) |
| 1 | **width** | `cluster_width / 200` | Tổng độ rộng cụm (vật cách ≤100px gộp lại) |
| 2 | **max_h** | `max_height / 120` | Chiều cao cây cao nhất trong cụm |
| 3 | **has_bird** | `0` hoặc `1` | Có chim trong cụm? |
| 4 | **bird_y** | `bird.y / ground_y` | Vị trí dọc của chim |
| 5 | **bird_x** | `bird.x / 1200` | Tọa độ x của chim |
| 6 | **jump_safety** | `(34 - dist/speed) / 34` | Nhảy bây giờ có clear không? (1=an toàn) |
| 7 | **speed** | `game_speed / 16` | Tốc độ game |
| 8 | **dino_y** | `dino.y / ground_y` | Vị trí dọc dino |
| 9 | **dino_vel** | `dino.vel_y / 18.5` | Vận tốc dọc (âm=lên, dương=xuống) |
| 10 | **is_jumping** | `0` hoặc `1` | Dino đang nhảy? |
| 11 | **is_ducking** | `0` hoặc `1` | Dino đang cúi? |

### Cụm obstacle (cluster)

Obstacle cách nhau ≤ 100px → gộp 1 cụm. State lấy:
- **dist**: khoảng cách đến vật đầu tiên
- **width**: tổng rộng từ đầu đến cuối cụm
- **max_h**: chiều cao nhất trong cụm
- **bird_x/y**: tọa độ chim nếu có

### Jump Safety

```
jump_duration = 2 × 18.5 / 1.1 ≈ 34 frames
jump_safety  = (34 - distance/speed) / 34   (clip 0→1)

  dist=300, speed=10 → time=30 → safety=0.12 (có thể nhảy)
  dist=170, speed=10 → time=17 → safety=0.50 (lý tưởng)
  dist=50,  speed=10 → time=5  → safety=0.85 (hơi trễ)
```

> Nếu `max_h > 155px` → jump_safety = 0 (vật quá cao, không nhảy qua được)

## 2. Action Space

| Action | Hành động |
|---|---|
| 0 | **DUCK** — cúi |
| 1 | **JUMP** — nhảy |
| 2 | **RUN** — chạy thẳng |

## 3. Reward Function

| Sự kiện | Reward |
|---|---|
| Chết | **-1.0** |
| Sống mỗi frame | **+0.1** |
| Vượt xương rồng | **+10.0** |
| Vượt chim thấp (<40px ground) — phải nhảy | **+25.0** |
| Vượt chim giữa (40-80px) | **+15.0** |
| Vượt chim cao (>80px) + cúi | **+30.0** |
| Vượt chim cao (>80px) + nhảy | **+5.0** |

## 4. Network Architecture

```
Input(12) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(3)
```

~10,000 tham số. Double DQN + soft target update + Huber loss.

## 5. Hyperparameters

### Training

| Param | Value | Giải thích |
|---|---|---|
| `lr` | `5e-5` | Learning rate thấp → ổn định |
| `gamma` | `0.995` | Discount cao → ưu tiên sống lâu |
| `tau` | `0.02` | Soft update chậm → target ổn định |
| `batch_size` | `256` | Batch lớn → gradient ổn định |
| `buffer_capacity` | `200_000` | ~600 episodes gần nhất |
| `learn_start` | `5_000` | Đợi đủ experience mới học |
| `learn_every` | `2` | Học mỗi 2 steps |
| `grad_clip` | `1.0` | Gradient clipping |
| `loss` | SmoothL1Loss | Huber loss |

### Exploration

| Param | Value |
|---|---|
| `eps_start` | `1.0` |
| `eps_decay` | `0.997` |
| `eps_end` | `0.02` |

## 6. Game Config (`shared/config.py`)

| Hằng số | Value | Ý nghĩa |
|---|---|---|
| `SCREEN_W` | 1200 | Chiều rộng màn hình |
| `SCREEN_H` | 450 | Chiều cao màn hình |
| `FPS` | 60 | Khung hình/giây |
| `GRAVITY` | 1.1 px/frame² | Trọng lực |
| `JUMP_VEL` | 18.5 px/frame | Vận tốc nhảy |
| `INIT_SPEED` | 6.0 | Tốc độ ban đầu |
| `SPEED_INCREMENT` | 0.005 / frame | Tăng tốc |
| `MAX_SPEED` | 16.0 | Tốc độ tối đa |
| `STATE_SIZE` | 12 | Kích thước state vector |
| `ACTION_SIZE` | 3 | Số action |

### Công thức nhảy

```
T = 2 × JUMP_VEL / GRAVITY = 2 × 18.5 / 1.1 ≈ 34 frame (0.57s)
H = JUMP_VEL² / (2 × GRAVITY) = 18.5² / 2.2 ≈ 155 px
```

## 7. Spawn Logic (`shared/game_env.py`)

### Xương rồng

| Tốc độ | Cluster | Ghi chú |
|---|---|---|
| < 8 | 1-2 cây (70% 1) | Thưa, dễ học |
| 8-11 | 1-2 cây (50% 2) | |
| > 11 | 1-3 cây | Max 3 |

### Chim (Ptera)

| Tốc độ | Tỉ lệ |
|---|---|
| ≤ 8 | 15% |
| 8-12 | 15-25% |
| > 12 | 25% |

Không spawn chim 2 lần liên tiếp. 3 độ cao: sát đất (phải nhảy), giữa, cao (nên cúi).

### Khoảng cách giữa các nhóm

| Tốc độ | Gap |
|---|---|
| ≤ 8 | 500-800 px |
| 8-12 | 400-650 px |
| 12-14 | 350-550 px |
| > 14 | 300-500 px |

## 8. Sơ đồ

```
┌──────────┐    state(12)     ┌───────────────┐    action(3)    ┌──────────┐
│   GAME   │ ───────────────► │   Q-NETWORK   │ ──────────────► │   DINO   │
│   ENV    │                  │  12→128→64→3  │                 │  J/D/R   │
│          │ ◄────────────────│               │                 │          │
└──────────┘   (s,a,r,s',done)└───────────────┘                 └──────────┘
      │                               ▲
      │         ┌───────────┐         │
      └────────►│  REPLAY   │─────────┘
                │  BUFFER   │  batch 256
                │  200K     │  learn /2
                └───────────┘
                      │
                      ▼
                ┌───────────────┐
                │  DOUBLE DQN   │
                │  target net   │  tau=0.02
                │  Huber loss   │
                └───────────────┘
```

## 9. Kết quả tốt nhất

| Metric | Value |
|---|---|
| Train best score | **1029** |
| Eval mean | 94 |
| Eval max | 178 |

(Cấu hình hiện tại là bản đã đạt 1029, giữ lại toàn bộ cải tiến state 12D + jump_safety + bird reward phân loại)
