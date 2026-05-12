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
└── shared/                  # PHẦN CHUNG — không cần sửa
    ├── config.py            # Hằng số: kích thước màn hình, tốc độ, jump physics...
    ├── base_ai.py           # Lớp abstract BaseDinoAI — mọi AI phải kế thừa
    ├── game_env.py          # Môi trường game: khủng long, chướng ngại vật, logic
    ├── renderer.py          # Load & scale sprite từ sprite sheet
    ├── evaluator.py         # Công cụ đánh giá & so sánh các AI
    ├── manual_play.py       # Chế độ chơi tay bằng bàn phím
    └── templates/           # Sprite sheet ảnh (dino, cactus, ptera...)
```

## Cài đặt

```bash
# 1. Cài Python 3.10+
# 2. Cài thư viện
pip install pygame numpy Pillow

# Hoặc:
pip install -r requirements.txt
```

## Cách chạy

```bash
# So sánh 3 AI (chạy headless, in bảng điểm)
python main.py --mode compare

# Xem AI tốt nhất chơi (có giao diện)
python main.py --mode watch

# Tự chơi tay
python main.py --mode manual

# Huấn luyện (mỗi thành viên tự train file riêng)
python main.py --mode train_all
```

## Cách thêm AI của bạn

### Bước 1: Copy template

```bash
cp template_ai.py member1_ga_ai.py    # Thành viên 1: GA
cp template_ai.py member2_pso_ai.py   # Thành viên 2: PSO
cp template_ai.py member3_dqn_ai.py   # Thành viên 3: DQN
```

### Bước 2: Implement 3 hàm bắt buộc

Mở file của bạn, class kế thừa `BaseDinoAI` (trong `shared/base_ai.py`). Phải implement **3 hàm**:

```python
from shared.base_ai import BaseDinoAI
import numpy as np

class MyAI(BaseDinoAI):

    def __init__(self):
        super().__init__(name="Tên AI của tôi")
        # Khởi tạo model / neural network / genome ở đây

    def predict(self, state: np.ndarray) -> int:
        """
        Nhận state vector 6 chiều → trả về action.
        Action: 0 = cúi (duck)
                1 = nhảy (jump)
                2 = chạy (run)
        """
        # TODO: forward pass của model bạn
        return 2  # tạm thời luôn chạy

    def train(self, **kwargs):
        """Huấn luyện model (GA/PSO/DQN)."""
        # TODO: vòng lặp huấn luyện
        # Gợi ý: dùng fitness_single() từ evaluator
        pass

    def save_model(self, path: str):
        """Lưu model ra file."""
        # TODO: np.savez / pickle / torch.save...
        pass

    def load_model(self, path: str):
        """(Tuỳ chọn) Tải model từ file."""
        pass
```

### State vector (6 chiều)

| Index | Ý nghĩa | Công thức |
|-------|---------|-----------|
| 0 | Khoảng cách đến obstacle gần nhất | `(ob.x - 80) / SCREEN_W` |
| 1 | Chiều cao obstacle | `(ground_y - ob.y - ob.h) / ground_y` |
| 2 | Chiều rộng obstacle | `ob.w / 60` |
| 3 | Là chim? | `1.0` nếu ptera, `0.0` nếu cactus |
| 4 | Độ cao chim | `(ground_y - ob.y) / ground_y` |
| 5 | Tốc độ game hiện tại | `game_speed / MAX_SPEED` |

### Fitness / Điểm

Dùng hàm `fitness_single()` trong `shared/evaluator.py` để đánh giá AI:

```python
from shared.evaluator import fitness_single

score = fitness_single(my_ai, n_avg=3)  # chạy 3 lần, lấy điểm trung bình
```

### Bước 3: Đăng ký AI vào main.py

Sau khi viết xong, mở `main.py`, bỏ comment dòng import và thay TemplateAI:

```python
from member1_ga_ai  import GeneticAlgorithmAI
from member2_pso_ai import PSODinoAI
from member3_dqn_ai import DQNDinoAI

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
- Dùng PyTorch hoặc TensorFlow
- State (6,) → vài lớp Dense + ReLU → Q-values (3,)
- Replay buffer lưu (state, action, reward, next_state, done)
- Epsilon-greedy exploration
- Target network cập nhật định kỳ

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

## Lưu ý

- **Không sửa file trong thư mục `shared/`** — đó là phần chung của cả nhóm
- File AI của mỗi người chỉ cần implement `predict()`, `train()`, `save_model()`
- Nên dùng `requirements.txt` để cài thêm thư viện nếu cần (VD: `torch`, `tensorflow`)
- Model lưu vào thư mục `models/` (tự tạo)
