# Chrome Dino AI — Đồ án AI

Dự án game Chrome Dino kết hợp AI: 3 thành viên, mỗi người cài đặt 1 thuật toán AI khác nhau (Genetic Algorithm, Particle Swarm Optimization, Deep Q-Network) để điều khiển khủng long tự động vượt chướng ngại vật.

## Cấu trúc thư mục

```
PythonProject/
├── main.py                  # Điểm vào chính: so sánh AI, xem AI chơi, chơi tay
├── template_ai.py           # FILE MẪU — mỗi thành viên copy & đổi tên để viết AI
├── README.md                # File này
├── requirements.txt         # Thư viện cần cài
│
├── shared/                  # PHẦN CHUNG — không cần sửa
│   ├── config.py            # Hằng số: màn hình, tốc độ, jump physics, state size
│   ├── base_ai.py           # Lớp abstract BaseDinoAI — mọi AI phải kế thừa
│   ├── game_env.py          # Môi trường game: khủng long, chướng ngại vật, logic
│   ├── renderer.py          # Load & scale sprite từ sprite sheet
│   ├── evaluator.py         # Công cụ đánh giá & so sánh các AI
│   ├── manual_play.py       # Chế độ chơi tay bằng bàn phím
│   └── templates/           # Sprite sheet ảnh (dino, cactus, ptera...)
│
├── dqn/                     # DQN — Deep Q-Network
│   ├── dqn_ai.py            # QNetwork, ReplayBuffer, DQNDinoAI
│   ├── train_dqn.py         # Script huấn luyện + dashboard
│   └── colab_train.ipynb    # Notebook train trên Kaggle/Colab
│
└── models/                  # Model đã train (tự tạo)
    └── dqn_best.pkl
```

## Cài đặt

```bash
# Python 3.10+ + GPU (tuỳ chọn, cho DQN)

# Cài thư viện
pip install pygame numpy matplotlib pillow torch

# Hoặc:
pip install -r requirements.txt
```

## Cách chạy

```bash
# === DQN ===
python dqn/train_dqn.py              # Huấn luyện từ đầu (2000 episodes)
python dqn/train_dqn.py --resume     # Tiếp tục train từ checkpoint
python dqn/train_dqn.py --eval       # Đánh giá không render (20 lần)
python dqn/train_dqn.py --watch      # Xem AI đã train chơi (5 ván)

# === Tất cả AI ===
python main.py --mode compare        # So sánh 3 AI (headless)
python main.py --mode watch          # Xem AI tốt nhất chơi
python main.py --mode manual         # Tự chơi tay
python main.py --mode train_all      # Huấn luyện tất cả
```

## State vector (16 chiều)

DQN dùng state 16 chiều — 2 obstacle gần nhất + game speed + dino self-state:

| Index | Feature | Công thức |
|-------|---------|-----------|
| **Obstacle 1 (gần nhất)** | | |
| 0 | Khoảng cách | `(ob.x - 80) / SCREEN_W` |
| 1 | Chiều cao obstacle | `(ground_y - ob.y - ob.h) / ground_y` |
| 2 | Chiều rộng | `ob.w / 60` |
| 3 | Là chim? | `1.0` nếu ptera, `0.0` nếu cactus |
| 4 | Độ cao chim | `(ground_y - ob.y) / ground_y` (chỉ khi is_bird) |
| 5 | Tốc độ game | `game_speed / MAX_SPEED` |
| **Obstacle 2** | | |
| 6-10 | Tương tự obstacle 1 | |
| **Dino self-state** | | |
| 11 | Độ cao dino | `y / ground_y` |
| 12 | Vận tốc dọc | `vel_y / jump_vel` (âm = đang bay lên) |
| 13 | Đang nhảy? | `1.0` nếu `is_jumping` |
| 14 | Đang cúi? | `1.0` nếu `is_ducking` |
| 15 | Padding | `0.0` (dự phòng) |

**Action space (3 actions):**
- `0` = Duck (cúi)
- `1` = Jump (nhảy)
- `2` = Run (chạy)

## Kiến trúc DQN

```
State (16) → Linear(512) → ReLU → Dropout(0.1)
           → Linear(512) → ReLU → Dropout(0.1)
           → Linear(256) → ReLU → Dropout(0.1)
           → Linear(3)   → Q(s,a)
```

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `lr` | 3e-5 | Learning rate (Adam) |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Soft update rate cho target network |
| `eps_start` | 1.0 | Epsilon ban đầu (100% random) |
| `eps_decay` | 0.998 | Decay mỗi episode |
| `eps_end` | 0.02 | Epsilon tối thiểu (2% explore) |
| `buffer_capacity` | 200,000 | Replay buffer size |
| `batch_size` | 64 | Batch size mỗi lần học |
| `learn_start` | 5,000 | Số experience tối thiểu trước khi học |
| `learn_every` | 4 | Học mỗi 4 steps |
| `dropout` | 0.1 | Regularization |
| `grad_clip` | 1.0 | Gradient clipping max norm |

**Loss function:** SmoothL1Loss (Huber) — ổn định hơn MSE cho DQN.

**Thuật toán:** Double DQN với soft target update — online network chọn action, target network đánh giá Q-value.

## Reward function

```
reward = env_reward + 0.05                    (survival bonus nhẹ)

env_reward = -50.0                            (nếu chết)
env_reward = 1.0 + len(cleared_obstacles) × 10  (nếu sống)
```

Mỗi obstacle vượt qua được thưởng +10 điểm — đây là tín hiệu chính để agent học cách né.

## Huấn luyện DQN

### Local (CPU/GPU)

```bash
python dqn/train_dqn.py
```

- 2000 episodes, mỗi ep tối đa 10,000 steps
- Model tốt nhất tự động lưu vào `model/dqn_best.pkl`
- Dashboard lưu vào `model/training_curve.png`
- In log mỗi 50 episodes

### Kaggle (GPU T4 x2, free)

1. Mở notebook `dqn/colab_train.ipynb` trên Kaggle
2. Accelerator → GPU T4 x2
3. Upload 3 folder `shared/`, `dqn/`, `templates/` vào `/kaggle/working/`
4. Run all

| Thời gian | Episodes | Chất lượng |
|-----------|----------|------------|
| ~20 phút | 300 | Khá |
| ~1 giờ | 800 | Tốt |
| ~2-3 giờ | 2000 | Rất tốt |

## Cách thêm AI của bạn

### Bước 1: Copy template

```bash
cp template_ai.py member1_ga_ai.py    # Thành viên 1: GA
cp template_ai.py member2_pso_ai.py   # Thành viên 2: PSO
cp template_ai.py member3_dqn_ai.py   # Thành viên 3: tham khảo dqn/
```

### Bước 2: Implement 3 hàm bắt buộc

```python
from shared.base_ai import BaseDinoAI
import numpy as np

class MyAI(BaseDinoAI):

    def __init__(self):
        super().__init__(name="Tên AI của tôi")

    def predict(self, state: np.ndarray) -> int:
        """
        Nhận state vector (16 chiều) → trả về action.
        Action: 0 = duck, 1 = jump, 2 = run
        """
        return 2  # TODO: forward pass của model bạn

    def train(self, **kwargs):
        """Huấn luyện model (GA/PSO/DQN)."""
        pass

    def save_model(self, path: str):
        """Lưu model ra file."""
        pass

    def load_model(self, path: str):
        """(Tuỳ chọn) Tải model từ file."""
        pass
```

### Bước 3: Đăng ký AI vào main.py

```python
from member1_ga_ai  import GeneticAlgorithmAI
from member2_pso_ai import PSODinoAI
from dqn.dqn_ai     import DQNDinoAI

ais = [GeneticAlgorithmAI(), PSODinoAI(), DQNDinoAI()]
```

## Gợi ý triển khai từng thuật toán

### Genetic Algorithm (GA)
- Mỗi cá thể = 1 bộ trọng số neural network (w1, b1, w2, b2)
- Fitness = `fitness_single()` chạy 3 lần lấy trung bình
- Selection: tournament hoặc roulette wheel
- Crossover: uniform hoặc single-point
- Mutation: thêm nhiễu Gaussian nhỏ

### Particle Swarm Optimization (PSO)
- Mỗi particle = 1 bộ trọng số
- Velocity cập nhật theo `p_best` và `g_best`
- Fitness = `fitness_single()`
- Nên chạy ít nhất 50-100 thế hệ

### Deep Q-Network (DQN)
- Đã cài đặt sẵn trong `dqn/` — xem `dqn/dqn_ai.py`
- Double DQN + Soft Target Update + Huber Loss
- Có thể điều chỉnh tham số trong `DQN_CONFIG`

## Testing AI của bạn

```bash
# Chạy file AI riêng để test nhanh
python member1_ga_ai.py

# Hoặc trong code:
if __name__ == "__main__":
    from shared.evaluator import evaluate, watch_ai
    ai = MyAI()
    ai.load_model("models/my_model.npz")
    evaluate(ai, n_runs=10)   # đánh giá 10 lần
    watch_ai(ai, n_games=3)   # xem AI chơi 3 ván
```

## Cấu hình game

Chỉnh trong `shared/config.py`:

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `SCREEN_W/H` | 1200×450 | Kích thước màn hình |
| `FPS` | 60 | Khung hình/giây |
| `INIT_SPEED` | 6.0 | Tốc độ ban đầu |
| `MAX_SPEED` | 16.0 | Tốc độ tối đa |
| `SPEED_INCREMENT` | 0.005 | Tăng tốc mỗi frame |
| `GRAVITY` | 1.1 | Trọng lực (px/frame²) |
| `JUMP_VEL` | 18.5 | Vận tốc nhảy (px/frame) |
| `DINO_SCALE` | 0.45 | Tỉ lệ khủng long |
| `OBSTACLE_SCALE` | 0.80 | Tỉ lệ xương rồng |
| `BIRD_SCALE` | 0.70 | Tỉ lệ chim |

## Lưu ý

- **Không sửa file trong `shared/`** khi chưa thống nhất với nhóm
- File AI của mỗi người implement `predict()`, `train()`, `save_model()`
- Model lưu vào thư mục `models/` (tự tạo)
- DQN cần PyTorch — cài thêm: `pip install torch`
- Train trên Kaggle GPU T4 miễn phí nếu máy không có GPU
