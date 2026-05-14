# Chrome Dino AI — Đồ án AI

Dự án game Chrome Dino kết hợp AI điều khiển khủng long tự động vượt chướng ngại vật.

## Cấu trúc thư mục

```
PythonProject/
├── main.py                  # Điểm vào chính
├── template_ai.py           # Template cho AI mới
├── README.md
├── requirements.txt
│
├── shared/                  # PHẦN CHUNG
│   ├── config.py            # Hằng số game, state size, action size
│   ├── base_ai.py           # Lớp abstract BaseDinoAI
│   ├── game_env.py          # Game environment + spawn logic + reward
│   ├── renderer.py          # Load & scale sprite
│   ├── evaluator.py         # Đánh giá & so sánh AI
│   └── manual_play.py       # Chơi tay
│
├── dqn/                     # Deep Q-Network
│   ├── dqn_ai.py            # QNetwork, ReplayBuffer, DQNDinoAI
│   ├── train_dqn.py         # Script train + dashboard
│   └── README.md            # Chi tiết DQN
│
└── model/                   # Model đã train
    └── dqn_best.pkl
```

## Cài đặt

```bash
pip install pygame numpy matplotlib pillow torch
# hoặc: pip install -r requirements.txt
```

## Cách chạy

```bash
python dqn/train_dqn.py              # Train DQN từ đầu (2000 episodes)
python dqn/train_dqn.py --resume     # Tiếp tục train từ checkpoint
python dqn/train_dqn.py --eval       # Đánh giá (10 runs, không render)
python dqn/train_dqn.py --watch      # Xem AI chơi (5 ván)
```

---

## Game Config (`shared/config.py`)

| Hằng số | Value | Ý nghĩa |
|---|---|---|
| `SCREEN_W` | 1200 | Chiều rộng màn hình |
| `SCREEN_H` | 450 | Chiều cao màn hình |
| `FPS` | 60 | Khung hình/giây |
| `GROUND_Y_OFFSET` | 20 | Khoảng cách ground đến đáy màn hình |
| `GRAVITY` | 1.1 px/frame² | Trọng lực |
| `JUMP_VEL` | 18.5 px/frame | Vận tốc nhảy ban đầu |
| `INIT_SPEED` | 6.0 | Tốc độ game ban đầu |
| `SPEED_INCREMENT` | 0.005 / frame | Tốc độ tăng dần |
| `MAX_SPEED` | 16.0 | Tốc độ tối đa |
| `STATE_SIZE` | 13 | Kích thước state vector |
| `ACTION_SIZE` | 3 | Số action (duck/jump/run) |
| `DINO_SCALE` | 0.45 | Tỉ lệ thu nhỏ dino |
| `OBSTACLE_SCALE` | 0.80 | Tỉ lệ thu nhỏ xương rồng |
| `BIRD_SCALE` | 0.70 | Tỉ lệ thu nhỏ chim |

### Công thức nhảy

```
T = 2 × JUMP_VEL / GRAVITY = 2 × 18.5 / 1.1 ≈ 34 frames (0.57s)
H = JUMP_VEL² / (2 × GRAVITY) = 18.5² / 2.2 ≈ 155 px
```

---

## State Vector (13D)

Mỗi frame, AI nhận vector 13 chiều:

| Index | Tên | Công thức | Ý nghĩa |
|---|---|---|---|
| 0 | **dist** | `(cluster_x - dino.x) / 1200` | Khoảng cách đến cụm vật cản |
| 1 | **width** | `cluster_width / 200` | Tổng độ rộng cụm |
| 2 | **max_h** | `max_height / 120` | Chiều cao cây cao nhất trong cụm |
| 3 | **has_bird** | `0` hoặc `1` | Có chim trong cụm? |
| 4 | **bird_y** | `bird.y / ground_y` | Vị trí dọc của chim |
| 5 | **bird_x** | `bird.x / 1200` | Tọa độ x của chim |
| 6 | **jump_safety** | `(34 - dist/speed) / 34` | Nhảy bây giờ có clear? (1=an toàn) |
| 7 | **bird_high** | `0` hoặc `1` | Chim cao → nên CÚI |
| 8 | **speed** | `game_speed / 16` | Tốc độ game |
| 9 | **dino_y** | `dino.y / ground_y` | Vị trí dọc dino |
| 10 | **dino_vel** | `dino.vel_y / 18.5` | Vận tốc dọc (âm=lên) |
| 11 | **is_jumping** | `0` hoặc `1` | Đang nhảy? |
| 12 | **is_ducking** | `0` hoặc `1` | Đang cúi? |

### Cụm obstacle (cluster)

Vật cản cách nhau ≤ 100px được gộp thành 1 cụm:
- **dist**: khoảng cách đến vật đầu tiên trong cụm
- **width**: tổng độ rộng từ đầu đến cuối cụm
- **max_h**: chiều cao vật cao nhất
- **has_bird / bird_x / bird_y / bird_high**: thông tin chim nếu có

### Jump Safety (state[6])

Dựa trên vật lý nhảy: `jump_duration ≈ 34 frames`

```
jump_safety = (34 - distance/speed) / 34   (clip 0→1)

  dist=300, speed=10 → time=30 → 0.12 (có thể nhảy)
  dist=170, speed=10 → time=17 → 0.50 (lý tưởng)
  dist=50,  speed=10 → time=5  → 0.85 (hơi trễ)
```

> Nếu `max_h > 155px` → jump_safety = 0 (vật quá cao, không nhảy qua được)

### Bird High (state[7])

```
bird_high = 1.0 nếu chim cách ground > 80px (chim cao → cúi đầu)
            0.0 nếu không có chim hoặc chim thấp (phải nhảy)
```

Tín hiệu tường minh — AI không cần suy luận từ bird_y.

---

## Action Space

| Action | Hành động |
|---|---|
| 0 | **DUCK** |
| 1 | **JUMP** |
| 2 | **RUN** |

---

## Reward Function

| Sự kiện | Reward |
|---|---|
| Chết | **-1.0** |
| Sống mỗi frame | **+0.1** |
| Vượt xương rồng | **+10.0** |
| Vượt chim thấp (<40px ground) — phải nhảy | **+25.0** |
| Vượt chim giữa (40-80px) | **+15.0** |
| Vượt chim cao (>80px) — cúi | **+30.0** |
| Vượt chim cao (>80px) — nhảy | **+5.0** |

---

## Spawn Logic

### Xương rồng

| Tốc độ | Cluster size |
|---|---|
| < 8 | 1-2 cây (70% 1 cây) |
| 8-11 | 1-2 cây (50% 2 cây) |
| > 11 | 1-3 cây |

Khoảng cách giữa các cây trong cụm: 10-25px.

### Chim (Ptera)

| Tốc độ | Tỉ lệ spawn |
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

---

## Kiến trúc DQN

```
Input(13) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(3)
```

~10,000 tham số. Double DQN + soft target update + Huber loss.

### Hyperparameters

| Param | Value | Giải thích |
|---|---|---|
| `lr` | `5e-5` | Learning rate thấp → ổn định |
| `gamma` | `0.995` | Discount cao → ưu tiên sống lâu |
| `tau` | `0.02` | Soft target update chậm → ổn định |
| `batch_size` | `256` | Batch lớn → gradient ổn định |
| `buffer_capacity` | `200_000` | ~600 episodes gần nhất |
| `learn_start` | `5_000` | Đợi đủ experience mới học |
| `learn_every` | `2` | Học mỗi 2 steps |
| `grad_clip` | `1.0` | Gradient clipping |
| `loss` | SmoothL1Loss | Huber loss |
| `eps_start` | `1.0` | Epsilon ban đầu |
| `eps_decay` | `0.997` | Decay mỗi episode |
| `eps_end` | `0.02` | Epsilon tối thiểu |

### Sơ đồ

```
┌──────────┐    state(13)     ┌───────────────┐    action(3)
│   GAME   │ ───────────────► │   Q-NETWORK   │ ──────────────► DINO
│   ENV    │                  │  13→128→64→3  │
│          │ ◄────────────────│               │
└──────────┘   (s,a,r,s',done)└───────────────┘
      │                               ▲
      │         ┌───────────┐         │
      └────────►│  REPLAY   │─────────┘
                │  BUFFER   │  batch 256
                │  200K     │
                └───────────┘
```

---

## Kết quả tốt nhất (DQN)

| Metric | Value |
|---|---|
| Train best | **1372** |
| Eval mean | 182 |
| Eval max | 326 |

---

## Cách thêm AI mới

### 1. Copy template

```bash
cp template_ai.py my_ai.py
```

### 2. Implement 3 hàm bắt buộc

```python
from shared.base_ai import BaseDinoAI
import numpy as np

class MyAI(BaseDinoAI):
    def __init__(self):
        super().__init__(name="MyAI")

    def predict(self, state: np.ndarray) -> int:
        """state: 13D vector → action: 0=duck, 1=jump, 2=run"""
        return 2

    def train(self, **kwargs):
        pass

    def save_model(self, path: str):
        pass

    def load_model(self, path: str):
        pass
```

### 3. Test

```python
if __name__ == "__main__":
    from shared.evaluator import evaluate, watch_ai
    ai = MyAI()
    evaluate(ai, n_runs=10)   # Đánh giá 10 lần
    watch_ai(ai, n_games=3)   # Xem AI chơi
```
