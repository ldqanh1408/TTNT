import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ppo.ppo_ai import PPODinoAI, PPO_CONFIG
from shared.evaluator import watch_ai, evaluate

MODEL_PATH = "model/ppo_best.pkl"
CHART_PATH = "model/training_curve_ppo.png"

def plot_scores(scores: list, save_path: str = CHART_PATH):
    n_ep = len(scores)
    window = min(50, n_ep)
    if n_ep >= window:
        moving_avg = np.convolve(scores, np.ones(window) / window, mode="valid")
        avg_x      = range(window - 1, n_ep)
    else:
        moving_avg = []
        avg_x      = []

    fig = plt.figure(figsize=(12, 6), facecolor="#0f1117")
    fig.suptitle("PPO Chrome Dino – Training Dashboard", fontsize=16, fontweight="bold", color="white", y=0.98)

    ax_score = fig.add_subplot(121)
    ax_avg   = fig.add_subplot(122)

    BG, GRID, TEXT = "#1a1d27", "#2a2d3a", "#c8ccd8"

    for ax in [ax_score, ax_avg]:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.7)

    ax_score.plot(scores, color="#4a9eff", alpha=0.5, linewidth=0.8, label="Score")
    if len(moving_avg):
        ax_score.plot(avg_x, moving_avg, color="#ff9f43", linewidth=2, label=f"Avg({window})")
    
    best_ep = int(np.argmax(scores)) if scores else 0
    if scores:
        ax_score.scatter(best_ep, scores[best_ep], color="#ff4757", zorder=5, s=60, label=f"Best={scores[best_ep]}")
    
    ax_score.set_title("Score per episode")
    ax_score.set_xlabel("Episode")
    ax_score.set_ylabel("Score")
    ax_score.legend(fontsize=8, facecolor=BG, labelcolor=TEXT, framealpha=0.8)

    if len(moving_avg):
        ax_avg.plot(avg_x, moving_avg, color="#ff9f43", linewidth=2)
        ax_avg.fill_between(avg_x, moving_avg, alpha=0.15, color="#ff9f43")
        
    ax_avg.set_title(f"Moving Average (window={window})")
    ax_avg.set_xlabel("Episode")
    ax_avg.set_ylabel("Avg Score")

    stats_text = (
        f"Episodes: {n_ep}  |  "
        f"Best: {max(scores) if scores else 0}  |  "
        f"Final avg({window}): {np.mean(scores[-window:]) if len(scores) >= window else 0:.1f}"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9, color="#74b9ff", bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=GRID))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Biểu đồ đã lưu: {save_path}")

def train_from_scratch():
    ai = PPODinoAI()
    scores = ai.train(n_episodes=2000, save_path=MODEL_PATH)
    plot_scores(scores)

def train_resume():
    ai = PPODinoAI()
    ai.load_model(MODEL_PATH)
    scores = ai.train(n_episodes=1000, save_path=MODEL_PATH)
    plot_scores(scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Huấn luyện / Đánh giá")
    parser.add_argument("--eval",  action="store_true", help="Đánh giá mô hình đã lưu")
    parser.add_argument("--watch", action="store_true", help="Xem AI chơi (render)")
    parser.add_argument("--resume",action="store_true", help="Tiếp tục huấn luyện từ model đã lưu")
    args = parser.parse_args()

    # Tạo thư mục model nếu chưa có
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if args.watch:
        ai = PPODinoAI()
        ai.load_model(MODEL_PATH)
        watch_ai(ai, n_games=5)
    elif args.eval:
        ai = PPODinoAI()
        ai.load_model(MODEL_PATH)
        evaluate(ai, n_runs=10, render=False)
    elif args.resume:
        train_resume()
    else:
        train_from_scratch()
