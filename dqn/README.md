# DQN – Deep Q-Network cho Chrome Dino

Triển khai thuật toán **Double DQN with Experience Replay** để điều khiển AI chơi Chrome Dino. Mục tiêu: tối đa hoá điểm số bằng cách học chính sách nhảy/cúi/chạy tối ưu.

---

## Mục lục

1. [Cấu trúc file](#1-cấu-trúc-file)
2. [Kiến trúc mạng](#2-kiến-trúc-mạng)
3. [State vector (12 chiều)](#3-state-vector-12-chiều)
4. [Thuật toán](#4-thuật-toán)
5. [Reward shaping](#5-reward-shaping)
6. [Cấu hình](#6-cấu-hình)
7. [Cách sử dụng](#7-cách-sử-dụng)
8. [Huấn luyện trên Colab / Kaggle](#8-huấn-luyện-trên-colab--kaggle)
9. [Load model vào code](#9-load-model-vào-code)
10. [Giải thích các thành phần](#10-giải-thích-các-thành-phần)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Cấu trúc file

| File | Mô tả |
|---|---|
| `dqn_ai.py` | Triển khai DQN: mạng nơ-ron, replay buffer, vòng lặp train |
| `train_dqn.py` | Script huấn luyện, đánh giá, xem AI chơi |
| `colab_train.ipynb` | Notebook Google Colab / Kaggle (GPU miễn phí) |
| `models/` | Thư mục lưu checkpoint (`dqn_best.pkl`) |

---

## 2. Kiến trúc mạng

```
Input (12 chiều)
  ├── Obstacle 1: [dist, height, width, is_bird, bird_height]
  ├── Obstacle 2: [dist, height, width, is_bird, bird_height]
  └── speed_ratio
        │
        ▼
  Linear(12 → 128) → ReLU()
  Linear(128 → 128) → ReLU()
  Linear(128 → 3)             ← Q(s,0), Q(s,1), Q(s,2)
```

Output: 3 Q-value — chọn action có Q-value cao nhất.

**Activation function:**

| Layer | Activation | Lý do |
|---|---|---|
| Hidden 1-2 | ReLU | Tránh vanishing gradient, tính nhanh |
| Output | Tuyến tính | Q-value có thể âm (dùng MSE loss) |

---

## 3. State vector (12 chiều)

```
Index  Feature          Mô tả                           Normalize
────────────────────────────────────────────────────────────────
[0]    dist_to_obs1     Khoảng cách đến vật gần nhất    ÷ SCREEN_W
[1]    obs_height1      Độ cao đáy vật thứ 1            ÷ ground_y
[2]    obs_width1       Chiều rộng vật thứ 1            ÷ 60
[3]    is_bird1         1.0 nếu là chim, 0.0 nếu cactus ÷ 1
[4]    bird_height1     Độ cao đỉnh chim (chỉ khi bird)  ÷ ground_y
[5]    speed_ratio      game_speed / MAX_SPEED            ÷ 1
[6-10] (obstacle 2)    Lặp lại 5 features cho vật thứ 2
[11]   (padding)       Luôn = 0.0
```

**Lưu ý:**
- Obstacles được **sort theo x** (nhỏ nhất = xa nhất = nguy hiểm nhất) trước khi điền vào state
- Nếu chỉ có 1 obstacle, obstacle thứ 2 = vector 0
- `dist` lấy `max(0, ob.x - 80)` — offset 80px để dino có phản ứng sớm hơn
- `is_bird=1` thì `bird_height` được set, ngược lại = 0
- Tất cả clip vào `[0, 1]` để tránh giá trị ngoài range

**Action:**

| Value | Action | Dùng khi |
|---|---|---|
| `0` | Cúi (duck) | Bird bay thấp, cactus thấp |
| `1` | Nhảy (jump) | Cactus cao, bird bay cao |
| `2` | Chạy (run) | Giữ nguyên tư thế, tốc độ ổn định |

---

## 4. Thuật toán

### Double DQN
Dùng **online network** chọn action ở next state, **target network** ước lượng giá trị. Giảm overestimation bias.

```
target_Q = r + γ * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)
```

### Experience Replay
Lưu transition `(state, action, reward, next_state, done)` vào deque 50.000. Mỗi bước học lấy ngẫu nhiên 64 mẫu — phá vỡ temporal correlation.

### Soft Update
Target network cập nhật mềm mỗi bước:
```
θ_target ← τ * θ_online + (1 - τ) * θ_target
```
`τ = 0.005`. Ổn định hơn hard update (copy cứng sau N bước).

### Epsilon-Greedy
```
ε = max(0.05, ε × 0.995)
```
Bắt đầu 1.0, về ~0.05 sau ~600 episodes. Exploration giảm dần để exploitation tăng dần.

### Gradient Clipping
`max_norm = 1.0` ngăn gradient explode — đặc biệt quan trọng khi reward shaping tạo gradient lớn.

---

## 5. Reward shaping

Reward được tính lại trước khi lưu vào buffer:

```
if done:          shaped_reward = -10.0
elif reward > 0:  shaped_reward = 1.0 + game_speed × 0.05
else:             shaped_reward = 0.0
```

| Trường hợp | Reward gốc | Shaped | Giải thích |
|---|---|---|---|
| Chết | -50 | **-10** | Giảm tỉ lệ âm: 50:1 → 5:1 |
| Sống + speed cao | +1 | **1.0–1.8** | Thưởng tốc độ cao (sống lâu hơn) |
| Sống + speed thấp | +1 | **1.0–1.1** | Ít thưởng hơn |
| Vượt qua obstacle | +10 | **+10** | Thưởng từ env |

**Tại sao -10 thay vì -50?**
- `-50` tạo gradient quá mạnh → mạng sợ chết quá mức, trở nên bảo thủ
- `-10` vẫa đủ để phân biệt chết vs sống, không làm gradient dominate

---

## 6. Cấu hình

Chỉnh trong `dqn_ai.py`, dict `DQN_CONFIG`:

### Kiến trúc mạng

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `hidden_sizes` | `[128, 128]` | 2 lớp ẩn, mỗi lớp 128 neurons |
| `state_size` | `12` | Kích thước state vector |
| `action_size` | `3` | 3 actions: duck, jump, run |

### Replay buffer

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `buffer_capacity` | `50_000` | Số transition tối đa lưu trong buffer |
| `batch_size` | `64` | Số mẫu mỗi lần học |

### Học tập

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `lr` | `1e-3` | Learning rate — tốc độ cập nhật weights |
| `gamma` | `0.97` | Discount factor — mức độ ưu tiên reward tương lai |
| `tau` | `0.005` | Soft-update coefficient — tốc độ cập nhật target network |
| `learn_start` | `1_000` | Bắt đầu học sau N transitions (đợi buffer đầy đủ) |
| `learn_every` | `4` | Học sau mỗi N bước (tiết kiệm compute) |

### Epsilon-greedy

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `eps_start` | `1.0` | Epsilon ban đầu — 100% random |
| `eps_end` | `0.05` | Epsilon tối thiểu — 5% random, 95% greedy |
| `eps_decay` | `0.995` | Decay rate — về `eps_end` sau ~600 episodes |

### Target network

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `use_soft_update` | `True` | Soft update mỗi bước (bật = ổn định hơn) |
| `target_update_freq` | `200` | Tần suất hard update (chỉ dùng khi soft_update=False) |

### Hướng dẫn tuning

```
Muốn AI học CHẬM hơn nhưng ổn định:  giảm lr (1e-4), tăng tau (0.01)
Muốn AI học NHANH hơn:                tăng lr (2e-3), giảm eps_decay (0.990)
Muốn AI thăm dò nhiều hơn:           giảm eps_end (0.01), giảm eps_decay (0.990)
Muốn AI chơi an toàn:                tăng gamma (0.99), giảm lr (5e-4)
Muốn AI chơi tấn công:              giảm gamma (0.95), tăng lr (2e-3)
```

---

## 7. Cách sử dụng

### Huấn luyện từ đầu

```bash
python dqn/train_dqn.py
```

Mặc định: 800 episodes, tự động lưu checkpoint tốt nhất vào `model/dqn_best.pkl`.

### Tăng số episodes

Sửa `train_dqn.py`, hàm `train_from_scratch()`:

```python
scores = ai.train(
    n_episodes       = 2000,   # ← tăng ở đây
    max_steps_per_ep = 10_000,  # ← tăng nếu muốn episode dài hơn
    ...
)
```

### Tiếp tục train từ checkpoint

```bash
python dqn/train_dqn.py --resume
```

Model đã train 800 ep có thể train tiếp thêm 400 ep để cải thiện.

### Xem AI đã train chơi

```bash
python dqn/train_dqn.py --watch
```

Cần có `model/dqn_best.pkl`. Đặt `epsilon=0` để AI luôn chọn action tốt nhất.

### Chỉ đánh giá (không render)

```bash
python dqn/train_dqn.py --eval
```

Chạy 20 lần, in mean/max/min/std score.

---

## 8. Huấn luyện trên Colab / Kaggle

GPU miễn phí, không cần cài đặt local.

### Google Colab

1. Mở `colab_train.ipynb` trên Google Colab
2. **Runtime > Change runtime type > GPU (T4)**
3. Upload 3 folder `shared/`, `dqn/`, `templates/` vào panel bên trái (kéo thả)
4. Mount Google Drive (model tự lưu vào Drive, không mất khi Colab reset)
5. **Runtime > Run all**

### Kaggle

1. Tạo dataset từ 3 folder trên Kaggle
2. Tạo notebook mới → **Add Data** → gắn dataset
3. Enable GPU: panel bên phải → **Settings > Accelerator > GPU T4**
4. Copy nội dung `colab_train.ipynb` (bỏ bước mount Drive)
5. Đổi đường dẫn `/content/` thành `/kaggle/working/`

### So sánh

| | Colab | Kaggle |
|---|---|---|
| GPU | T4 (free, 8h/phiên) | T4 (free, 30h/tuần) |
| Lưu model | Google Drive | Download thủ công |
| Setup | Upload files tay | Gắn dataset dễ hơn |

---

## 9. Load model vào code

```python
from dqn.dqn_ai import DQNDinoAI

ai = DQNDinoAI()
ai.load_model("model/dqn_best.pkl")
ai.epsilon = 0.0   # greedy — luôn chọn action tốt nhất

# Lấy action từ state (np.ndarray 12 chiều)
action = ai.predict(state)   # trả về 0, 1, hoặc 2
```

**Lưu ý:** Model train với `state_size=12`. Dùng model cũ train với `state_size=6` sẽ crash vì kích thước layer không khớp.

---

## 10. Giải thích các thành phần

### Tại sao dùng ReLU ở hidden layers?

1. **Tính toán nhanh** — chỉ 1 phép so sánh `max(0, x)`. Không có exp/log như sigmoid/tanh.
2. **Giảm vanishing gradient** — ReLU có gradient = 1 khi x > 0, không suy giảm khi lan ngược nhiều layers.
3. **Sparse activation** — ~50% neurons die khi input ngẫu nhiên → mỗi neuron học features khác nhau, tăng tổng quát hoá.

### Tại sao output layer dùng linear (không ReLU)?

Q-value có thể âm (action chọn sai gây thất bại). ReLU clip về 0 → không thể represent negative Q-values → mất thông tin.

### Tại sao dùng Double DQN?

DQN thường overestimate Q-values vì cùng một mạng vừa chọn vừa đánh giá action. Double DQN dùng 2 mạng riêng biệt (online + target) để giảm bias này.

### Tại sao Experience Replay?

Các transition liên tiếp có correlation cao (state t+1 gần state t). Nếu học theo thứ tự, mạng có thể overfit vào một region nhất định. Replay shuffle giữa các transition để mạng học đều từ mọi phần của không gian state.

### Tại sao clip gradient?

Gradient clipping `max_norm=1.0` ngăn gradient explode (bùng nổ) khi reward shaping tạo signal mạnh bất thường. Đặc biệt quan trọng với `-10` death penalty và `+10` pass reward.

---

## 11. Troubleshooting

### Crash ngay khi chạy

```
ModuleNotFoundError: No module named 'torch'
```
```bash
pip install pygame numpy torch matplotlib pillow
```

### State size mismatch

```
RuntimeError: size mismatch
```
Model checkpoint cũ train với `state_size=6` không tương thích với code mới (`state_size=12`). Xoá checkpoint cũ và train lại:

```bash
del models\dqn_best.pkl
python dqn/train_dqn.py
```

### GPU không được dùng

Kiểm tra device trong log khởi tạo:
```
[DQN-Dino] Khởi tạo xong. Device: cuda   ← GPU đang dùng
[DQN-Dino] Khởi tạo xong. Device: cpu    ← GPU không khả dụng
```

Thử:
```python
import torch
print(torch.cuda.is_available())  # True = GPU khả dụng
```

### Loss = NaN

Thường do:
1. **Gradient explode** → tăng `clip_grad_norm` lên 0.5 hoặc giảm `lr`
2. **State có giá trị Inf/NaN** → kiểm tra `_build_state()` có clip đúng
3. **Reward quá lớn** → giảm pass reward từ 10 xuống 5

### AI không học (score = 0 sau 500 episodes)

1. `eps_decay` quá chậm → thử `0.990` thay vì `0.995`
2. `lr` quá thấp → thử `2e-3`
3. `gamma` quá thấp → thử `0.99`
4. Buffer chưa đầy → tăng `learn_start` và đợi

### AI chỉ nhảy hoặc chỉ cúi

1. Epsilon chưa giảm đủ → train thêm episodes
2. Reward shaping chưa tốt → thử tăng pass reward
3. State không đủ thông tin → kiểm tra `is_bird` feature có đúng

---

## Minh hoạ thuật toán (pseudo-code)

```
Initialize Q_online(s, a) và Q_target(s, a) = Q_online
Initialize replay buffer R (capacity = 50,000)

FOR episode = 1 to n_episodes:
    state = env.reset()
    FOR step = 1 to max_steps:
        # Epsilon-greedy
        IF random() < epsilon:
            action = random(0, 1, 2)
        ELSE:
            action = argmax_a Q_online(state, a)

        next_state, reward, done, info = env.step(action)

        # Reward shaping
        IF done:     shaped = -10
        ELSE IF reward > 0: shaped = 1 + speed * 0.05
        ELSE:         shaped = 0

        R.push(state, action, shaped, next_state, done)
        state = next_state

        # Học mỗi learn_every bước
        IF step % learn_every == 0 AND len(R) >= learn_start:
            batch = R.sample(batch_size)

            # Double DQN
            best_a = argmax_a Q_online(next_state, a)
            target = reward + gamma * Q_target(next_state, best_a) * (1 - done)

            loss = MSE(Q_online(state, action), target)
            optimize Q_online with gradient descent
            gradient_clip(1.0)

            # Soft update target network
            Q_target = tau * Q_online + (1 - tau) * Q_target

        IF done: BREAK

    epsilon = max(eps_end, epsilon * eps_decay)
```
