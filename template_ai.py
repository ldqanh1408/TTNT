# ============================================================
#  template_ai.py  –  TEMPLATE cho từng thành viên
#  Mỗi người copy file này, đổi tên, rồi tự viết phần model
# ============================================================
import numpy as np
import os
from shared.base_ai import BaseDinoAI
from shared.config import STATE_SIZE, ACTION_SIZE


class TemplateAI(BaseDinoAI):
    """
    ĐÂY LÀ FILE MẪU – copy rồi đổi tên thành:
      - member1_ga_ai.py   (thành viên 1: Genetic Algorithm)
      - member2_pso_ai.py  (thành viên 2: Particle Swarm Optimization)
      - member3_dqn_ai.py  (thành viên 3: Deep Q-Network)
    """

    # ─── 1. Khởi tạo model / trọng số ───────────────────────
    def __init__(self):
        super().__init__(name="TemplateAI")   # ← đổi tên
        # TODO: khởi tạo mạng nơ-ron / genome / particle ở đây
        # Ví dụ mạng đơn giản: 6 → 12 → 3
        self.w1 = np.random.randn(STATE_SIZE, 12) * 0.5
        self.b1 = np.zeros(12)
        self.w2 = np.random.randn(12, ACTION_SIZE) * 0.5
        self.b2 = np.zeros(ACTION_SIZE)

    # ─── 2. Dự đoán action từ state ──────────────────────────
    def predict(self, state: np.ndarray) -> int:
        """
        state: numpy array shape (6,)
        return: int  0=cúi  1=nhảy  2=chạy
        """
        # TODO: thay bằng forward pass của model thật
        h = np.tanh(state @ self.w1 + self.b1)
        out = h @ self.w2 + self.b2
        return int(np.argmax(out))

    # ─── 3. Huấn luyện ───────────────────────────────────────
    def train(self, env=None, n_generations=100, **kwargs):
        """
        Implement thuật toán của bạn ở đây.

        Gợi ý gọi evaluator:
            from evaluator import fitness_single
            score = fitness_single(self, n_avg=3)
        """
        # TODO: viết vòng lặp train (GA / PSO / DQN)
        raise NotImplementedError("Bạn chưa implement train()!")

    # ─── 4. Lưu model ────────────────────────────────────────
    def save_model(self, path: str = "models/template.npz"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        print(f"[{self.name}] Đã lưu model → {path}")

    # ─── 5. Tải model (tuỳ chọn) ─────────────────────────────
    def load_model(self, path: str = "models/template.npz"):
        data     = np.load(path)
        self.w1  = data["w1"]
        self.b1  = data["b1"]
        self.w2  = data["w2"]
        self.b2  = data["b2"]
        print(f"[{self.name}] Đã tải model ← {path}")


# ─────── Chạy thử nhanh ──────────────────────────────────────
if __name__ == "__main__":
    from evaluator import evaluate, watch_ai

    ai = TemplateAI()
    # Nếu đã train xong:
    # ai.load_model("models/template.npz")

    # Đánh giá 5 lần
    evaluate(ai, n_runs=5)

    # Xem AI chơi 2 ván
    watch_ai(ai, n_games=2)