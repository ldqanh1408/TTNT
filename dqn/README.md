# DQN Dino AI — Cấu hình & Thiết kế

Deep Q-Network điều khiển Chrome Dino tự động né chướng ngại vật.

## 1. State Vector (12D)

Mỗi frame, AI nhận vector 12 chiều mô tả trạng thái game:

| Index | Tên | Công thức | Ý nghĩa |
|---|---|---|---|
| 0 | **dist** | `(cluster_x - dino.x) / 1200` | Khoảng cách đến cụm vật cản (0=sát mặt, 1=xa) |
| 1 | **width** | `cluster_width / 200` | Tổng độ rộng cụm (các vật cách nhau ≤100px gộp lại) |
| 2 | **max_h** | `max_height / 120` | Chiều cao cây cao nhất trong cụm |
| 3 | **has_bird** | `0` hoặc `1` | Có chim trong cụm không? |
| 4 | **bird_y** | `bird.y / ground_y` | Vị trí dọc của chim (0 nếu không có chim) |
| 5 | **bird_x** | `bird.x / 1200` | Tọa độ x của chim (0 nếu không có chim) |
| 6 | **jump_safety** | `(34 - dist/speed) / 34` | Nhảy bây giờ có clear được không? (1=an toàn, 0=chưa nên) |
| 7 | **speed** | `game_speed / 16` | Tốc độ game hiện tại |
| 8 | **dino_y** | `dino.y / ground_y` | Vị trí dọc của dino |
| 9 | **dino_vel** | `dino.vel_y / 18.5` | Vận tốc dọc (âm=lên, dương=xuống) |
| 10 | **is_jumping** | `0` hoặc `1` | Dino đang nhảy? |
| 11 | **is_ducking** | `0` hoặc `1` | Dino đang cúi? |

### Cụm obstacle (cluster)

Các obstacle cách nhau ≤ 100px được gộp thành 1 cụm. State lấy:
- **dist**: khoảng cách đến vật ĐẦU TIÊN trong cụm
- **width**: tổng độ rộng từ vật đầu đến vật cuối
- **max_h**: chiều cao vật CAO NHẤT trong cụm
- **has_bird / bird_x / bird_y**: thông tin chim nếu có trong cụm

```
Ví dụ: [xương rồng x=400 h=35] [xương rồng x=440 h=70]
→ cụm: dist=400/1200, width=74/200, max_h=70/120, has_bird=0
```

### Jump Safety

Dựa trên vật lý nhảy: `jump_duration = 2 × 18.5 / 1.1 ≈ 34 frames`

```
jump_safety = (34 - distance/speed) / 34   (clip 0→1)

  dist=600, speed=10 → time=60 → safety=0.0  (quá xa, chưa nhảy)
  dist=300, speed=10 → time=30 → safety=0.12 (có thể nhảy)
  dist=170, speed=10 → time=17 → safety=0.5  (thời điểm lý tưởng)
  dist=50,  speed=10 → time=5  → safety=0.85 (hơi trễ)
```

Nếu `max_h > 148px` → vật quá cao, jump_safety = 0.

## 2. Action Space (3 actions)

| Action | Hành động |
|---|---|
| 0 | **DUCK** — cúi xuống |
| 1 | **JUMP** — nhảy lên |
| 2 | **RUN** — chạy thẳng |

## 3. Reward Function

| Sự kiện | Reward | Ghi chú |
|---|---|---|
| **Chết** | `-1.0` | Va chạm vật cản |
| **Sống** | `+0.1` / frame | Khuyến khích sống lâu |
| **Vượt xương rồng** | `+10.0` | Nhảy qua cactus |
| **Vượt chim thấp** (<40px ground) | `+25.0` | Phải nhảy, không còn cách khác |
| **Vượt chim giữa** (40-80px) | `+15.0` | Có thể nhảy hoặc cúi |
| **Vượt chim cao** (>80px) + cúi | `+30.0` | Hành động tối ưu: cúi né |
| **Vượt chim cao** (>80px) + nhảy | `+5.0` | Vẫn được nhưng ít điểm |

## 4. Network Architecture

```
Input(12) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(3)
```

~10,000 tham số. Double DQN với soft target update.

## 5. Hyperparameters

### Training

| Param | Value | Giải thích |
|---|---|---|
| `lr` | `5e-5` | Learning rate thấp → hội tụ ổn định, tránh catastrophic forgetting |
| `gamma` | `0.995` | Discount cao → ưu tiên sống lâu dài |
| `tau` | `0.02` | Soft update chậm → target network ổn định |
| `batch_size` | `256` | Batch lớn → gradient ổn định |
| `buffer_capacity` | `200_000` | Chứa ~600 episodes gần nhất, rác cũ bị đẩy ra |
| `learn_start` | `5_000` | Đợi đủ experience trước khi bắt đầu học |
| `learn_every` | `2` | Học mỗi 2 steps |
| `grad_clip` | `1.0` | Gradient clipping |
| `loss` | SmoothL1Loss | Huber loss — ổn định với outlier |

### Exploration

| Param | Value | Giải thích |
|---|---|---|
| `eps_start` | `1.0` | 100% exploration lúc đầu |
| `eps_decay` | `0.997` | Giảm dần, về eps_end sau ~1300 episodes |
| `eps_end` | `0.02` | 2% exploration vĩnh viễn |

## 6. Game Physics

| Hằng số | Value | Ý nghĩa |
|---|---|---|
| `SCREEN_W` | 1200 | Chiều rộng màn hình |
| `SCREEN_H` | 450 | Chiều cao màn hình |
| `FPS` | 60 | Khung hình/giây |
| `GRAVITY` | 1.1 px/frame² | Trọng lực |
| `JUMP_VEL` | 18.5 px/frame | Vận tốc nhảy ban đầu |
| `INIT_SPEED` | 6.0 | Tốc độ game ban đầu |
| `SPEED_INCREMENT` | 0.005 / frame | Tốc độ tăng dần |
| `MAX_SPEED` | 16.0 | Tốc độ tối đa |
| `DINO_SCALE` | 0.45 | Tỉ lệ thu nhỏ dino |
| `OBSTACLE_SCALE` | 0.80 | Tỉ lệ thu nhỏ xương rồng |
| `BIRD_SCALE` | 0.70 | Tỉ lệ thu nhỏ chim |

### Công thức nhảy

```
Thời gian nhảy:    T = 2 × JUMP_VEL / GRAVITY = 2 × 18.5 / 1.1 ≈ 34 frames (0.57s)
Độ cao tối đa:     H = JUMP_VEL² / (2 × GRAVITY) = 18.5² / 2.2 ≈ 155 px
```

## 7. Spawn Logic

### Xương rồng

| Tốc độ | Cluster size |
|---|---|
| < 8 | 1-2 cây (70% 1 cây) |
| 8-11 | 1-2 cây (50% 2 cây) |
| > 11 | 1-3 cây |

### Chim (Ptera)

| Tốc độ | Tỉ lệ spawn |
|---|---|
| ≤ 8 | 15% |
| 8-12 | 15-25% |
| > 12 | 25% |

Chim không spawn 2 lần liên tiếp. Có 3 độ cao: sát đất (phải nhảy), giữa, cao (nên cúi).

## 8. Sơ đồ hoạt động

```
┌──────────┐    state(12)     ┌──────────────┐    action(3)    ┌──────────┐
│   GAME   │ ───────────────► │   Q-NETWORK  │ ──────────────► │   DINO   │
│   ENV    │                  │  12→128→64→3 │                 │  JUMP/   │
│          │ ◄────────────────│              │                 │  DUCK/RUN│
└──────────┘   (s,a,r,s',done)└──────────────┘                 └──────────┘
      │                              ▲
      │         ┌──────────┐         │
      └────────►│ REPLAY   │─────────┘
                │ BUFFER   │  batch 256
                │ 200K     │  learn every 2
                └──────────┘
                       │
                       ▼
                ┌──────────────┐
                │  DOUBLE DQN  │
                │  target net  │
                │  tau=0.02    │
                └──────────────┘
```
