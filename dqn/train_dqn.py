# ============================================================
#  train_dqn.py  –  Script huấn luyện & đánh giá DQN
#
#  Cách chạy:
#    python train_dqn.py              # huấn luyện từ đầu
#    python train_dqn.py --watch      # xem AI đã train chơi
#    python train_dqn.py --resume     # tiếp tục train từ checkpoint
#    python train_dqn.py --eval       # đánh giá không render
# ============================================================

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # không cần cửa sổ GUI khi vẽ chart
import matplotlib.pyplot as plt

from dqn_ai import DQNDinoAI
from shared.evaluator import watch_ai, evaluate


MODEL_PATH = "models/dqn_best.pkl"
CHART_PATH = "models/dqn_training_curve.png"


# ──────────────────────────────────────────────────────────
#  Vẽ biểu đồ điểm theo episode
# ──────────────────────────────────────────────────────────
def plot_scores(scores: list, save_path: str = CHART_PATH):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(scores, alpha=0.4, color="steelblue", label="Score mỗi episode")

    # Đường trung bình trượt (window=50)
    if len(scores) >= 50:
        window = 50
        moving_avg = np.convolve(scores, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(scores)), moving_avg,
                color="orange", linewidth=2, label=f"Avg({window} ep)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("DQN – Quá trình học")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  📊 Biểu đồ đã lưu: {save_path}")


# ──────────────────────────────────────────────────────────
#  Train từ đầu
# ──────────────────────────────────────────────────────────
def train_from_scratch():
    ai = DQNDinoAI()

    scores = ai.train(
        n_episodes       = 800,    # ← tăng lên 1500-2000 nếu muốn kết quả tốt hơn
        max_steps_per_ep = 8_000,
        verbose_every    = 50,
        save_path        = MODEL_PATH,
    )

    plot_scores(scores)

    print("\n─── Đánh giá sau khi train ───")
    ai.epsilon = 0.0   # tắt exploration khi đánh giá
    evaluate(ai, n_runs=10, verbose=True)


# ──────────────────────────────────────────────────────────
#  Tiếp tục train từ checkpoint
# ──────────────────────────────────────────────────────────
def resume_training():
    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ Không tìm thấy {MODEL_PATH}. Chạy train_from_scratch() trước.")
        return

    ai = DQNDinoAI()
    ai.load_model(MODEL_PATH)

    # Giảm epsilon vì model đã biết khá nhiều
    ai.epsilon = max(0.1, ai.epsilon)

    scores = ai.train(
        n_episodes       = 400,
        max_steps_per_ep = 8_000,
        verbose_every    = 50,
        save_path        = MODEL_PATH,
    )

    plot_scores(scores, CHART_PATH.replace(".png", "_resumed.png"))


# ──────────────────────────────────────────────────────────
#  Xem AI chơi (có cửa sổ render)
# ──────────────────────────────────────────────────────────
def watch():
    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ Không tìm thấy {MODEL_PATH}.")
        return

    ai = DQNDinoAI()
    ai.load_model(MODEL_PATH)
    ai.epsilon = 0.0   # hoàn toàn greedy khi xem

    watch_ai(ai, n_games=5)


# ──────────────────────────────────────────────────────────
#  Đánh giá (không render, nhiều lần, lấy thống kê)
# ──────────────────────────────────────────────────────────
def eval_only():
    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ Không tìm thấy {MODEL_PATH}.")
        return

    ai = DQNDinoAI()
    ai.load_model(MODEL_PATH)
    ai.epsilon = 0.0

    stats = evaluate(ai, n_runs=20, verbose=True)
    print(f"\n  Tổng kết: mean={stats['mean']:.0f}  "
          f"max={stats['max']}  std={stats['std']:.0f}")


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Dino Trainer")
    parser.add_argument("--watch",  action="store_true", help="Xem AI chơi")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục train")
    parser.add_argument("--eval",   action="store_true", help="Chỉ đánh giá")
    args = parser.parse_args()

    if args.watch:
        watch()
    elif args.resume:
        resume_training()
    elif args.eval:
        eval_only()
    else:
        train_from_scratch()