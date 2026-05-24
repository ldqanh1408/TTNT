# Nghiên cứu so sánh Học tăng cường sâu và Thuật toán Tiến hoá cho bài toán điều khiển Agent tự động trong môi trường Chrome Dinosaur: Dueling Double DQN, Proximal Policy Optimization, và Genetic Algorithm Neural Network

> **Đồ án nghiên cứu so sánh ba mô hình AI** trên cùng một môi trường benchmark (Chrome Dinosaur game), đối chiếu ba trường phái Trí tuệ nhân tạo chính trong giải quyết bài toán điều khiển Agent tự động.

## Tóm tắt (Abstract)

Đồ án xây dựng và đánh giá định lượng ba paradigm AI khác nhau cho cùng bài toán điều khiển khủng long tự động vượt chướng ngại vật trong môi trường Chrome Dinosaur:

| # | Mô hình | Họ thuật toán | Đặc trưng cốt lõi |
|---|---|---|---|
| 1 | **Dueling Double DQN + PER** | Học tăng cường — Value-based, Off-policy | Phương trình Bellman, Replay Buffer ưu tiên TD-error, Dueling Heads |
| 2 | **PPO (Actor-Critic Clipped)** | Học tăng cường — Policy-based, On-policy | Clipped Surrogate Objective, GAE, Entropy Regularization |
| 3 | **GA-NN (Genetic Algorithm + Neural Network)** | Tiến hoá quần thể — Gradient-free | Tournament Selection, Uniform Crossover, Gaussian Mutation, Elitism |

Cả ba mô hình chia sẻ cùng một thiết lập MDP đặc trưng 15-chiều, cùng không gian hành động 3-chiều (DUCK/JUMP/RUN), cùng hàm phần thưởng và cùng chính sách sinh môi trường (`AdaptiveSpawnPolicy`). Điều này cho phép so sánh trực tiếp ba thuật toán trên các tiêu chí: *sample efficiency*, *policy stability*, *wall-clock convergence time*, và *generalization* qua các cấu hình spawn chưa thấy trong huấn luyện.

## Cấu trúc thư mục

```
PythonProject/
├── main.py                          # Điểm vào chính (chọn AI để demo)
├── template_ai.py                   # Template tạo AI mới
├── README.md                        # Tài liệu tổng thể (file này)
├── requirements.txt
│
├── shared/                          # PHẦN CHUNG GIỮA CẢ 3 MÔ HÌNH
│   ├── config.py                    # Hằng số game, state size, action size
│   ├── base_ai.py                   # Lớp abstract BaseDinoAI
│   ├── game_env.py                  # DinoEnv: physics + spawn + reward
│   ├── spawn_policy.py              # SpawnPolicy / AdaptiveSpawnPolicy
│   ├── renderer.py                  # Load & scale sprite
│   ├── evaluator.py                 # Đánh giá & so sánh AI
│   └── manual_play.py               # Chơi tay (baseline)
│
├── dqn/                             # MÔ HÌNH 1 — Dueling DDQN + PER
│   ├── dqn_ai.py                    # DuelingQNetwork + PER Buffer + DQNDinoAI
│   ├── train_dqn.py                 # Script train + dashboard
│   ├── dqn_academic_report.md       # Báo cáo học thuật DQN
│   └── README.md                    # Chi tiết DQN
│
├── ppo/                             # MÔ HÌNH 2 — PPO (Actor-Critic)
│   ├── ppo_ai.py                    # ActorCritic + Rollout Buffer + PPODinoAI
│   ├── train_ppo.py                 # Script train + dashboard
│   ├── ppo_academic_report.md       # Báo cáo học thuật PPO
│   └── README.md                    # Chi tiết PPO
│
├── ga/                              # MÔ HÌNH 3 — Genetic Algorithm + NN
│   │                                # (nằm trên nhánh `training/ga`)
│   ├── ga_ai.py                     # GAConfig + GenomeIndividual + GADinoAI
│   ├── neural_network.py            # DinoNet (NumPy MLP, không dùng autograd)
│   ├── train_ga.py                  # Script train + dashboard
│   ├── REPORT_GA_NN.md              # Báo cáo học thuật GA-NN
│   └── README_GA.md                 # Chi tiết GA
│
└── model/                           # Model đã train
    ├── dqn_best.pkl
    ├── ppo_best.pkl
    └── ga_best.pkl
```

## Cài đặt

```bash
pip install pygame numpy matplotlib pillow torch
# hoặc: pip install -r requirements.txt
```

## Cách chạy

### Mô hình 1 — DQN (Dueling Double DQN + PER)
```bash
python -m dqn.train_dqn              # Train từ đầu
python -m dqn.train_dqn --resume     # Tiếp tục từ checkpoint
python -m dqn.train_dqn --eval       # Đánh giá (không render)
python -m dqn.train_dqn --watch      # Xem AI chơi
```

### Mô hình 2 — PPO (Proximal Policy Optimization)
```bash
python -m ppo.train_ppo              # Train từ đầu (2000 episodes)
python -m ppo.train_ppo --resume     # Tiếp tục từ model đã lưu
python -m ppo.train_ppo --eval       # Đánh giá 10 runs
python -m ppo.train_ppo --watch      # Xem AI chơi 5 ván
```

### Mô hình 3 — GA-NN (Genetic Algorithm + Neural Network)
```bash
# Lưu ý: code GA hiện đang ở nhánh `training/ga`
git checkout training/ga
python -m ga.train_ga                # Train tiến hoá quần thể
python -m ga.train_ga --eval         # Đánh giá best individual
python -m ga.train_ga --watch        # Xem best individual chơi
```

### Tài liệu học thuật chi tiết theo mô hình
*   [`dqn/dqn_academic_report.md`](dqn/dqn_academic_report.md) — Báo cáo Dueling DDQN + PER
*   [`ppo/ppo_academic_report.md`](ppo/ppo_academic_report.md) — Báo cáo PPO + GAE
*   `ga/REPORT_GA_NN.md` — Báo cáo GA-NN *(trên nhánh `training/ga`)*

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

## Kiến trúc các mô hình (Tổng quan)

Cả ba mô hình đều nhận đầu vào State Vector 15-D và xuất 1 trong 3 action (DUCK/JUMP/RUN), nhưng có cấu trúc và cơ chế học khác nhau.

### Mô hình 1 — Dueling DDQN + PER
```
Input(15) → Linear(256)+LayerNorm+ReLU → Linear(128)+LayerNorm+ReLU
                                                         ↓
                          ┌──────────────────────────────┴──────────────────────────────┐
              Value Head: Linear(128→64)+LN+ReLU+Linear(64→1) → V(s)
              Advantage:  Linear(128→64)+LN+ReLU+Linear(64→3) → A(s, a)
                                                         ↓
              Q(s, a) = V(s) + (A(s, a) - mean_a' A(s, a'))
```
- **Cập nhật**: Phương trình Bellman với Double Q-Learning (target = $r + \gamma Q(s', \arg\max_a Q(s', a; \theta); \theta^-)$)
- **Replay**: Prioritized Experience Replay (200K, $\alpha=0.6$, $\beta: 0.4→1.0$)
- **Loss**: Huber + IS weights; **Optimizer**: Adam + Polyak soft target ($\tau=0.003$)
- Chi tiết: [`dqn/dqn_academic_report.md`](dqn/dqn_academic_report.md)

### Mô hình 2 — PPO (Actor-Critic Clipped)
```
Actor:   Input(15) → Linear(256)+LN+Tanh → Linear(128)+LN+Tanh → Linear(3) → Categorical π(a|s)
Critic:  Input(15) → Linear(256)+LN+Tanh → Linear(128)+LN+Tanh → Linear(1) → V(s)
```
- **Cập nhật**: Clipped Surrogate Objective $L^{\text{CLIP}} = E[\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)]$, $\epsilon=0.2$
- **Advantage**: Generalized Advantage Estimation (GAE), $\lambda=0.95$
- **Rollout**: 4096 steps → 10 epochs × 16 mini-batch (256) → clear buffer
- **Loss tổng**: $-L^{\text{CLIP}} + 0.5 L^{\text{VF}} - 0.01 H(\pi)$
- Chi tiết: [`ppo/ppo_academic_report.md`](ppo/ppo_academic_report.md)

### Mô hình 3 — GA-NN (Tiến hoá Quần thể)
```
Quần thể 80 cá thể × DinoNet(15→256→128→3 với ReLU+Softmax)
   ↓ Đánh giá fitness (chạy 5 ván, lấy điểm trung bình)
   ↓ Tournament Selection (k=5) → chọn bố mẹ
   ↓ Uniform Crossover (rate=0.80) → con
   ↓ Gaussian Mutation (rate=0.08, σ=0.10) → đột biến
   ↓ Elitism (giữ 8 cá thể tốt nhất, age-penalized) → quần thể mới
```
- **Không dùng gradient**: trọng số được tối ưu bằng thao tác sinh học trên gen
- **Hàm fitness**: trung bình điểm số 5 episode (tránh nhiễu single-run)
- **Backbone NN**: NumPy thuần (không PyTorch), forward duy nhất, không backward
- Chi tiết: `ga/REPORT_GA_NN.md` *(trên nhánh `training/ga`)*

---

## Bảng so sánh ba mô hình

| Tiêu chí | DQN | PPO | GA-NN |
|---|---|---|---|
| Trường phái | Value-based RL | Policy-based RL | Evolutionary Computation |
| On/Off-policy | Off-policy | On-policy | Không áp dụng (gradient-free) |
| Cơ chế học | Bellman + TD-error | Policy Gradient + Clip | Sinh học (selection/crossover/mutation) |
| Bộ nhớ | PER Buffer (200K) | Rollout Buffer (4096) | Quần thể (80 cá thể) |
| Exploration | $\epsilon$-greedy decay | Stochastic policy + entropy | Mutation $\sigma=0.10$ |
| Activation | ReLU | Tanh | ReLU + Softmax |
| Khởi tạo | Kaiming Normal | Orthogonal (std tuỳ lớp) | Xavier |
| Framework | PyTorch (autograd) | PyTorch (autograd) | NumPy (no autograd) |
| Sample efficiency | Cao (tái sử dụng) | Trung bình | Thấp (nhiều fitness eval) |
| Ổn định | Phụ thuộc PER/τ | Cao (clip + KL implicit) | Cao (elitism đảm bảo monotonic) |
| Song song hoá | Khó | Khó | Dễ (đánh giá quần thể đồng thời) |

---

## Tiêu chí đánh giá đồ án

Cả ba mô hình được benchmark trên các tiêu chí định lượng sau:
1. **Best Score**: điểm cao nhất đạt được trong huấn luyện
2. **Avg Eval Score**: điểm trung bình trên 10 episode đánh giá (không exploration)
3. **Wall-clock Convergence**: thời gian thực để đạt ngưỡng điểm mục tiêu
4. **Variance giữa seed**: độ ổn định giữa các lần chạy
5. **Generalization**: hiệu năng trên cấu hình spawn ngoài dải huấn luyện

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
