# ============================================================
#  base_ai.py  –  Lớp trừu tượng chung (PHẦN CHUNG – không sửa)
# ============================================================
import numpy as np
from abc import ABC, abstractmethod


class BaseDinoAI(ABC):
    """
    Mọi thành viên đều kế thừa lớp này và implement 3 hàm bắt buộc.

    Template:
    ---------
    class MyAI(BaseDinoAI):
        def __init__(self):
            super().__init__(name="Tên AI của tôi")
            # Khởi tạo model / trọng số ở đây

        def predict(self, state: np.ndarray) -> int:
            # state shape: (6,)  → action: 0, 1, hoặc 2
            ...

        def train(self, **kwargs):
            # Huấn luyện (GA/PSO/DQN tuỳ thành viên)
            ...

        def save_model(self, path: str):
            # Lưu model ra file
            ...

        def load_model(self, path: str):
            # Tải model từ file (tuỳ chọn nhưng nên có)
            ...
    """

    def __init__(self, name: str):
        self.name       = name
        self.best_score = 0
        self.generation = 0          # dùng cho GA/PSO

    # ─────────────── BẮT BUỘC IMPLEMENT ───────────────

    @abstractmethod
    def predict(self, state: np.ndarray) -> int:
        """
        Nhận state (numpy array shape (6,)), trả về action:
          0 = cúi (duck)
          1 = nhảy (jump)
          2 = chạy (run / không làm gì)
        """

    @abstractmethod
    def train(self, **kwargs):
        """
        Huấn luyện model.
        - GA/PSO: truyền vào env, n_episodes, v.v.
        - DQN   : truyền vào env, n_steps, replay_buffer, v.v.
        Kwargs tự định nghĩa theo thuật toán.
        """

    @abstractmethod
    def save_model(self, path: str):
        """Lưu trọng số / genome ra file (numpy / json / pickle)."""

    # ─────────────── TUỲ CHỌN (có thể override) ───────────────

    def load_model(self, path: str):
        """Tải lại model đã lưu. Override nếu cần."""
        raise NotImplementedError(f"[{self.name}] chưa implement load_model()")

    def get_info(self) -> dict:
        """Thông tin tóm tắt về AI (dùng cho Evaluator)."""
        return {
            "name"      : self.name,
            "best_score": self.best_score,
            "generation": self.generation,
        }

    def __repr__(self):
        return (f"<DinoAI name={self.name!r} "
                f"best={self.best_score} gen={self.generation}>")