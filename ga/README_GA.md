# Genetic Algorithm (GA) – Tài liệu hệ thống Chrome Dino AI

> **Mục tiêu:** Huấn luyện Dino tự động chơi game Chrome Dino bằng thuật toán Di truyền (Genetic Algorithm).
> **Người đọc:** Lập trình viên, nghiên cứu AI, sinh viên học thuật toán tiến hoá.

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Luồng dữ liệu](#3-luồng-dữ-liệu)
4. [Kiến trúc Neural Network](#4-kiến-trúc-neural-network)
5. [Thuật toán Di truyền chi tiết](#5-thuật-toán-di-truyền-chi-tiết)
6. [Tham số cấu hình](#6-tham-số-cấu-hình)
7. [Lệnh chạy](#7-lệnh-chạy)
8. [Cách đọc biểu đồ training](#8-cách-đọc-biểu-đồ-training)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Tổng quan hệ thống

### 1.1 Combo tối ưu cố định

Thuật toán GA sử dụng bộ ba phương pháp sinh học tối ưu, **cố định không thay đổi**:

| Giai đoạn | Phương pháp | Mô tả |
|---|---|---|
| **Selection** | Tournament + Elitism | Chọn cá thể tốt nhất trong nhóm nhỏ, giữ lại elite |
| **Crossover** | Uniform Crossover | Mỗi gen chọn ngẫu nhiên từ bố hoặc mẹ |
| **Mutation** | Gaussian Perturbation | Cộng nhiễu nhỏ Gaussian vào gen |

### 1.2 Sơ đồ luồng tiến hoá

```
Khởi tạo quần thể ngẫu nhiên (50 cá thể, Xavier init)
    ↓
┌─── Đánh giá fitness (chạy game) ────┐
│  Mỗi cá thể = 1 DinoNet             │
│  Fitness = điểm số trung bình       │
└───────────────────────────────────────┘
    ↓
Chọn lọc (Tournament) → Bố mẹ
    ↓
Lai ghép (Uniform Crossover) → Con
    ↓
Đột biến (Gaussian, σ=0.15) → Con mới
    ↓
Elitism → Giữ lại top 2 cá thể
    ↓
Lặp lại qua N thế hệ
    ↓
Cá thể tốt nhất mọi thời đại ← kết quả
```

---

## 2. Cấu trúc thư mục

```
ga/
├── neural_network.py   ← DinoNet (MLP, cố định Xavier/ReLU/Softmax)
│                        DenseLayer, ReLULayer, SoftmaxLayer
│                        get_flat_weights(), set_flat_weights()
│
├── ga_ai.py           ← GeneticAlgorithm + GenomeIndividual
│                        Tournament Selection, Uniform Crossover
│                        Gaussian Mutation, Elitism
│
├── train_ga.py        ← GADinoAI(BaseDinoAI)
│                        Vòng lặp evolve(), biểu đồ training curve
│
└── README_GA.md       ← Tài liệu này
```

### Quan hệ giữa các lớp

```
BaseDinoAI (shared/base_ai.py)
    └── GADinoAI (train_ga.py)
            ├── GeneticAlgorithm (ga_ai.py)
            │       ├── GenomeIndividual (ga_ai.py)
            │       │       └── DinoNet (neural_network.py)
            │       │               ├── DenseLayer (Xavier)
            │       │               ├── ReLULayer
            │       │               └── SoftmaxLayer
            │       ├── Tournament Selection
            │       ├── Uniform Crossover
            │       └── Gaussian Mutation
            │
            └── run_episode() (shared/evaluator.py)
                    └── DinoEnv (shared/game_env.py)
```

---

## 3. Luồng dữ liệu

### 3.1 Mỗi cá thể được đánh giá như thế nào?

```
1 episode = 1 lần chơi game từ đầu đến khi chết

DinoNet.predict(state) → action (0/1/2)
    ↓
DinoEnv.step_single(dino, action) → next_state, reward, done
    ↓
Lặp cho đến khi done=True hoặc đạt MAX_STEPS
    ↓
fitness = điểm số cuối cùng (score)
```

### 3.2 State vector (đầu vào) – 13 chiều

| Index | Tên | Ý nghĩa |
|---|---|---|
| 0 | `dist` | Khoảng cách đến obstacle gần nhất |
| 1 | `cluster_width` | Bề rộng cụm obstacle |
| 2 | `max_height` | Chiều cao lớn nhất trong cụm |
| 3 | `has_bird` | Có chim trong cụm không? |
| 4 | `bird_y` | Vị trí Y của chim |
| 5 | `bird_x` | Vị trí X của chim |
| 6 | `jump_safety` | Tín hiệu an toàn khi nhảy |
| 7 | `bird_high` | Chim đủ cao để cúi không? |
| 8 | `game_speed` | Tốc độ game hiện tại |
| 9 | `dino_y` | Vị trí Y của Dino |
| 10 | `dino_vel_y` | Vận tốc Y của Dino |
| 11 | `is_jumping` | Đang nhảy? |
| 12 | `is_ducking` | Đang cúi? |

### 3.3 Action (đầu ra)

| Giá trị | Hành động | Khi nào dùng |
|---|---|---|
| `0` | **CÚI (Duck)** | Chim bay cao, cần hạ thấp |
| `1` | **NHẢY (Jump)** | Xương rồng hoặc chim bay thấp |
| `2` | **CHẠY (Run)** | Không có chướng ngại, chạy tiếp |

---

## 4. Kiến trúc Neural Network

### 4.1 Sơ đồ mạng

```
Input(13) → Dense(256) → ReLU → Dense(128) → ReLU → Dense(3) → Softmax → argmax → action
```

### 4.2 Số lượng tham số

| Layer | Trọng số (W) | Bias (b) | Tổng |
|---|---|---|---|
| Dense 1: 13 → 256 | 3,328 | 256 | 3,584 |
| Dense 2: 256 → 128 | 32,768 | 128 | 32,896 |
| Dense 3: 128 → 3 | 384 | 3 | 387 |
| **Tổng** | | | **36,867 params** |

### 4.3 Flat weights – Chromosome

Toàn bộ **36,867 tham số** được duỗi phẳng thành 1 vector 1D. Đây chính là **chromosome** mà GA thao tác:

```
Chromosome = [w1_1, ..., w1_3328, b1_1, ..., b1_256,
              w2_1, ..., b2_1, ..., b2_128,
              w3_1, ..., b3_1, b3_2, b3_3]  ← 36,867 số thực
```

GA crossover hai chromosome bằng Uniform Crossover, mutate bằng Gaussian Perturbation.

---

## 5. Thuật toán Di truyền chi tiết

### 5.1 Khởi tạo quần thể (Initialization)

```
POPULATION_SIZE = 50 cá thể (mặc định)

Mỗi cá thể = 1 DinoNet với trọng số ngẫu nhiên (Xavier init)
→ Tạo ra 50 "bộ não" khác nhau
```

### 5.2 Đánh giá Fitness (Evaluation)

```
Với mỗi cá thể:
    fitness = mean(score_1, score_2, ..., score_FITNESS_EVALS)
    (mặc định: chạy 3 lần, lấy trung bình)

→ Fitness càng cao = cá thể càng giỏi chơi game
```

### 5.3 Chọn lọc – Tournament Selection + Elitism

```
Bước 1: Chọn ngẫu nhiên k cá thể (TOURNAMENT_SIZE=5)
Bước 2: So sánh fitness, chọn cá thể tốt nhất
Bước 3: Lặp lại cho đủ danh sách bố mẹ

Đồng thời:
    - ELITISM_COUNT=2 cá thể tốt nhất được copy thẳng sang thế hệ mới
    - Đảm bảo cá thể tốt nhất không bị mất đi
```

### 5.4 Lai ghép – Uniform Crossover

```
Chromosome của bố (P1):  [0.1, -0.2, 0.3, 0.4, 0.5, 0.6]
Chromosome của mẹ (P2):  [0.2, 0.3, -0.1, 0.7, 0.8, 0.9]

Mask ngẫu nhiên:         [T, F, T, F, T, F]

Con nhận gen từ bố nếu mask=True, từ mẹ nếu mask=False:
Con:                       [0.1, 0.3, 0.3, 0.7, 0.5, 0.9]

CROSSOVER_RATE = 0.80 (80% cơ hội crossover, 20% con = bản sao p1)
```

### 5.5 Đột biến – Gaussian Perturbation

```
Mỗi gen có 10% cơ hội bị đột biến (MUTATION_RATE=0.10)

Nếu bị đột biến:
    new_weight = old_weight + N(0, σ²)
    (σ = MUTATION_STRENGTH = 0.15)

→ Tạo sự ngẫu nhiên, giúp thoát khỏi local optimum
```

### 5.6 Vòng đời một thế hệ

```
┌─────────────────────────────────────────────────────────┐
│  Thế hệ t                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │ Ind #1   │   │ Ind #2   │   │ Ind #50  │  Fitness  │
│  │ f=42    │   │ f=128    │   │ f=67     │            │
│  └──────────┘   └──────────┘   └──────────┘            │
│       │                ↑               │                │
│       │          best fitness          │                │
│       ▼                │               ▼                │
│  Selection ───────────────────────────────────→ Parents│
│       │                                                  │
│       ▼                                                  │
│  Crossover (80%) + Mutation (10%)                        │
│       │                                                  │
│       ▼                                                  │
│  Elitism (top 2 copy thẳng) ──────────────────────────  │
│       │                                                  │
│       ▼                                                  │
│  Thế hệ t+1 ──→ Đánh giá fitness ──→ Next gen         │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Tham số cấu hình

### 6.1 GAConfig

| Tham số | Mặc định | Ý nghĩa | Ảnh hưởng |
|---|---|---|---|
| `POPULATION_SIZE` | `50` | Số cá thể | Lớn → đa dạng cao, chậm. Nhỏ → nhanh, dễ kém |
| `ELITISM_COUNT` | `2` | Cá thể tốt nhất giữ nguyên | Lớn → hội tụ nhanh, dễ stuck. Nhỏ → đa dạng hơn |
| `TOURNAMENT_SIZE` | `5` | Kích thước tournament | Lớn → áp lực chọn lọc mạnh hơn |
| `CROSSOVER_RATE` | `0.80` | Xác suất lai ghép | Lớn → tìm kiếm rộng, nhỏ → khai thác nhanh |
| `MUTATION_RATE` | `0.10` | Xác suất một gen đột biến | Lớn → đa dạng cao, chậm hội tụ. Nhỏ → hội tụ nhanh, dễ stuck |
| `MUTATION_STRENGTH` | `0.10` | Cường độ đột biến (σ) | Lớn → thay đổi mạnh. Nhỏ → fine-tuning |
| `FITNESS_EVALS` | `3` | Số lần chạy tính fitness/cá thể | Lớn → fitness chính xác, rất chậm |
| `RANDOM_SEED` | `None` | Seed ngẫu nhiên | Đặt số → reproduce được kết quả |

### 6.2 NNConfig (cố định)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `INPUT_SIZE` | `13` | State vector |
| `HIDDEN_SIZES` | `[256, 128]` | 2 hidden layers |
| `OUTPUT_SIZE` | `3` | 3 actions |
| Init | **Xavier** | Cố định |
| Activation | **ReLU** | Cố định |
| Output activation | **Softmax** | Cố định |

### 6.3 Gợi ý điều chỉnh

```
┌─────────────────┬───────────┬───────────┬───────────┐
│ Mục tiêu        │ Pop Size  │ Mutation  │ Elitism   │
├─────────────────┼───────────┼───────────┼───────────┤
│ Tìm nhanh       │ 20-30     │ 0.20-0.30 │ 1         │
│ Chất lượng cao  │ 80-100    │ 0.05-0.10 │ 3-5       │
│ Fine-tune       │ 20-30     │ 0.02-0.05 │ 5-10      │
│ Khám phá rộng   │ 100+      │ 0.30+     │ 0-1       │
└─────────────────┴───────────┴───────────┴───────────┘
```

---

## 7. Lệnh chạy

### 7.1 Huấn luyện từ đầu

```bash
# Mặc định: 300 thế hệ, quần thể 50
python ga/train_ga.py

# Tùy chỉnh
python ga/train_ga.py --gen 500
python ga/train_ga.py --gen 500 --pop 80
python ga/train_ga.py --gen 100 --pop 20
```

### 7.2 Tinh chỉnh siêu tham số

```bash
# Tăng mutation rate (khám phá nhiều hơn)
python ga/train_ga.py --gen 300 --mutation-rate 0.25

# Giảm mutation rate (fine-tune)
python ga/train_ga.py --gen 300 --mutation-rate 0.05

# Tăng crossover rate
python ga/train_ga.py --gen 300 --crossover-rate 0.90

# Tăng tournament size (áp lực chọn lọc mạnh hơn)
python ga/train_ga.py --gen 300 --tournament-size 7

# Cố định seed (reproduce được)
python ga/train_ga.py --gen 300 --seed 42

# Tăng fitness evals (chính xác hơn, chậm hơn)
python ga/train_ga.py --gen 300 --fitness-evals 5
```

### 7.3 Xem AI chơi

```bash
python ga/train_ga.py --watch
python ga/train_ga.py --watch --games 10
python ga/train_ga.py --watch --save model/best_dino_ga.pkl
```

### 7.4 Tiếp tục train từ checkpoint

```bash
python ga/train_ga.py --resume --gen 200
```

### 7.5 Đánh giá (không render)

```bash
python ga/train_ga.py --eval
python ga/train_ga.py --eval --eval-runs 50
python ga/train_ga.py --eval --save model/best_dino_ga.pkl --eval-runs 30

### 7.6 Checkpoint định kỳ

```bash
# Lưu checkpoint mỗi 50 thế hệ (mặc định)
python ga/train_ga.py --gen 500 --checkpoint-every 50

# Tắt checkpoint
python ga/train_ga.py --gen 500 --checkpoint-every 0
```

Checkpoint được lưu tại `model/best_dino_ga_genN.pkl` mỗi N thế hệ. Khi train bị crash, có thể resume từ checkpoint gần nhất.
```

### 7.6 Kết hợp tất cả

```bash
python ga/train_ga.py \
    --gen 400 \
    --pop 60 \
    --mutation-rate 0.12 \
    --crossover-rate 0.85 \
    --seed 12345 \
    --save model/my_ga_dino.pkl
```

---

## 8. Cách đọc biểu đồ Training

Sau khi train xong, `model/ga_training_curve.png` gồm 4 panel:

### Panel 1: Best & Avg Fitness

```
Score
  │
  │                    ╭── Best
  │               ╭────╯
  │          ╭────╯  ← Avg
  │     ╭────╯
  │╭────╯
  │╰─────────────── Thế hệ (0 → N)
  │
  └── Peak: điểm cao nhất mọi thời đại
```

**Đọc:** Đường Best càng lên cao → GA tìm được cá thể tốt hơn. Đường Avg theo Best → quần thể đang tiến hoá tốt.

### Panel 2: Best Fitness Smoothed

Đường Best đã được làm mượt (moving average) để thấy xu hướng rõ hơn, bỏ qua nhiễu.

### Panel 3: Fitness Range (Best/Avg/Worst)

```
Score
  │
  │███████████████ Best
  │░░░░░░░░░░░░░░ Avg
  │▒▒▒▒▒▒▒▒▒▒▒▒▒ Worst
  └──────────────── Thế hệ
```

**Đọc:** Vùng giữa Best và Worst cho biết độ đa dạng của quần thể. Khoảng cách lớn → quần thể không đồng nhất (bình thường ở đầu training).

### Panel 4: Histogram phân bố fitness (thế hệ cuối)

**Đọc:** Phân bố fitness của 50 cá thể cuối cùng. Dồn về phải → tiến hoá tốt. Dồn về trái → GA chưa hội tụ.

---

## 9. Troubleshooting

| Vấn đề | Nguyên nhân | Giải pháp |
|---|---|---|
| Fitness không tăng sau 50 gen | Population quá nhỏ hoặc mutation rate quá thấp | Tăng pop lên 80+, tăng mutation rate |
| Best fitness nhảy từng đợt lớn | Elitism quá cao | Giảm ELITISM_COUNT xuống 1-2 |
| Điểm số rất thấp (<10) ở mọi cá thể | Network chưa học được gì | Tăng số thế hệ, tăng fitness_evals |
| Training rất chậm | Fitness_evals quá cao hoặc pop quá lớn | Giảm fitness_evals xuống 1-2 |
| Điểm số best >> avg rất nhiều | Quần thể mất đa dạng | Tăng mutation rate, tăng pop |
| Best fitness tăng rồi giảm đột ngột | Crossover quá mạnh phá vỡ best genome | Giảm crossover rate, tăng elitism |

### Tái sử dụng best model

```python
from ga.ga_ai import GeneticAlgorithm
from ga.train_ga import GADinoAI
from shared.evaluator import watch_ai

# Tải best dino đã train
net, meta = GeneticAlgorithm.load_best("model/best_dino_ga.pkl")

# Tạo AI mới và gán best network
ai = GADinoAI()
ai.best_network = net
ai.best_score   = meta["fitness"]

# Xem AI chơi
watch_ai(ai, n_games=5)
```
