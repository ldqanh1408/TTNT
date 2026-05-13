# DQN – Deep Q-Network cho Chrome Dino

Thư mục chứa toàn bộ code liên quan đến thuật toán **Deep Q-Network (DQN)** để điều khiển AI chơi game Chrome Dino.

---

## Các file

| File | Mô tả |
|---|---|
| `dqn_ai.py` | Triển khai DQN: mạng nơ-ron, replay buffer, vòng lặp train |
| `train_dqn.py` | Script huấn luyện, đánh giá và xem AI chơi |

---

## Kiến trúc DQN

Mạng DQN gồm **input → hidden layers → output Q-values** cho 3 action.

```
Input (6 chiều)
  ├── dist_to_obs   – khoảng cách đến chướng ngại vật (normalized)
  ├── obs_height    – độ cao chướng ngại vật (normalized)
  ├── obs_width     – chiều rộng chướng ngại vật (normalized)
  ├── is_bird       – 1.0 nếu là chim, 0.0 nếu xương rồng
  ├── bird_height   – chiều cao chim (normalized, chỉ dùng khi is_bird=1)
  └── speed_ratio   – game_speed / MAX_SPEED
        ↓
  Linear(6 → 128) → ReLU()
  Linear(128 → 128) → ReLU()
  Linear(128 → 3)          ← Q(s,0), Q(s,1), Q(s,2)
```

**Action:**
- `0` – cúi (duck)
- `1` – nhảy (jump)
- `2` – chạy (run)

---

## Các thành phần chính

### Double DQN
Dùng **online network** để chọn action tốt nhất ở state tiếp theo, **target network** để ước lượng giá trị. Giảm overestimation bias so với DQN thường.

### Experience Replay
Lưu transition `(state, action, reward, next_state, done)` vào deque có giới hạn 50.000. Mỗi bước học lấy ngẫu nhiên 64 mẫu — phá vỡ correlation giữa các samples liên tiếp.

### Soft Update
Thay vì copy toàn bộ weights cứng sau mỗi N bước, target network được cập nhật mềm mỗi bước:
```
θ_target ← τ * θ_online + (1 - τ) * θ_target
```
với `τ = 0.005`. Ổn định hơn so với hard update.

### Epsilon-Greedy Exploration
- Bắt đầu `ε = 1.0` (100% khám phá)
- Giảm theo `ε *= 0.9995` sau mỗi episode
- Tối thiểu `ε = 0.05`

---

## Cách sử dụng

### Huấn luyện từ đầu

```bash
python dqn/train_dqn.py
```

Mặc định: 800 episodes, lưu checkpoint tốt nhất vào `models/dqn_best.pkl`, vẽ đồ thị vào `models/dqn_training_curve.png`.

### Tiếp tục train từ checkpoint

```bash
python dqn/train_dqn.py --resume
```

### Xem AI đã train chơi

```bash
python dqn/train_dqn.py --watch
```

### Chỉ đánh giá (không render)

```bash
python dqn/train_dqn.py --eval
```

---

## Cấu hình

Chỉnh trong `dqn_ai.py`, dict `DQN_CONFIG`:

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `hidden_sizes` | `[128, 128]` | Kích thước lớp ẩn |
| `buffer_capacity` | `50_000` | Dung lượng replay buffer |
| `batch_size` | `64` | Số mẫu mỗi lần học |
| `lr` | `1e-3` | Learning rate |
| `gamma` | `0.97` | Discount factor |
| `tau` | `0.005` | Soft-update coefficient |
| `eps_decay` | `0.9995` | Tốc độ giảm epsilon |
| `learn_start` | `1_000` | Bắt đầu học sau N transition |
| `learn_every` | `4` | Học sau mỗi N bước |

---

## Tại sao dùng ReLU?

ReLU (Rectified Linear Unit) `f(x) = max(0, x)` là activation function phổ biến nhất cho hidden layers vì 3 lý do:

**1. Tính toán nhanh** — chỉ cần 1 phép so sánh `x > 0`. Không có exp/log như sigmoid/tanh, rất cheap khi mạng có hàng triệu tham số.

**2. Giảm vanishing gradient** — sigmoid và tanh có đạo hàm ≤ 0.25 ở mọi nơi, nên gradient suy giảm exponential khi lan ngược qua nhiều layers. ReLU có gradient = 1 khi x > 0, nên signal truyền thẳng không bị co lại.

**3. Sparse activation** — với input ngẫu nhiên, ~50% neurons đầu ra sẽ = 0 (die). Điều này buộc mạng học các features khác nhau ở mỗi neuron thay vì cùng học 1 feature, tăng khả năng tổng quát hoá.

**Lưu ý:** ReLU không dùng ở output layer vì Q-value có thể âm (DQN dùng MSE loss). Output layer để **tuyến tính** để Q-value có thể nhận mọi giá trị thực.

---

## Load model trong code

```python
from dqn.dqn_ai import DQNDinoAI

ai = DQNDinoAI()
ai.load_model("models/dqn_best.pkl")
ai.epsilon = 0.0   # greedy khi đánh giá

# Lấy action
action = ai.predict(state)   # state là np.ndarray 6 chiều
```
