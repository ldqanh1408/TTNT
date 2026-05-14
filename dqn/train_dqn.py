# ============================================================
#  train_dqn.py  –  Script huấn luyện & đánh giá DQN
#
#  Cách chạy:
#    python train_dqn.py              # huấn luyện từ đầu
#    python train_dqn.py --watch      # xem AI đã train chơi
#    python train_dqn.py --resume     # tiếp tục train từ checkpoint
#    python train_dqn.py --eval       # đánh giá không render
# ============================================================

import sys
import os

# Force UTF-8 output tren Windows (trước mọi print)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Them project root vao sys.path de import tu shared/ va dqn/ deu work
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dqn.dqn_ai import DQNDinoAI, DQN_CONFIG
from shared.evaluator import watch_ai, evaluate


MODEL_PATH = "model/dqn_best.pkl"
CHART_PATH = "model/training_curve.png"


# ──────────────────────────────────────────────────────────
#  Vẽ biểu đồ 4 panel
# ──────────────────────────────────────────────────────────
def plot_scores(scores: list, losses: list = None,
                save_path: str = CHART_PATH):
    """
    Vẽ dashboard 4 panel:
      [0] Score mỗi episode (raw)
      [1] Moving average score (window=50)
      [2] Epsilon decay theo episode
      [3] Loss theo bước học (nếu có)
    """
    eps_start = DQN_CONFIG["eps_start"]
    eps_end   = DQN_CONFIG["eps_end"]
    eps_decay = DQN_CONFIG["eps_decay"]

    n_ep = len(scores)

    # Tính lại epsilon tại mỗi episode (từ eps_start)
    eps_curve = [max(eps_end, eps_start * (eps_decay ** i)) for i in range(n_ep)]

    # Moving average (window=50)
    window = min(50, n_ep)
    if n_ep >= window:
        moving_avg = np.convolve(scores, np.ones(window) / window, mode="valid")
        avg_x      = range(window - 1, n_ep)
    else:
        moving_avg = []
        avg_x      = []

    # ── Layout ──────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10), facecolor="#0f1117")
    fig.suptitle("DQN Chrome Dino – Training Dashboard",
                 fontsize=16, fontweight="bold",
                 color="white", y=0.98)

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.95,
                           top=0.92, bottom=0.08)

    ax_score  = fig.add_subplot(gs[0, 0])
    ax_avg    = fig.add_subplot(gs[0, 1])
    ax_eps    = fig.add_subplot(gs[1, 0])
    ax_loss   = fig.add_subplot(gs[1, 1])

    BG    = "#1a1d27"
    GRID  = "#2a2d3a"
    TEXT  = "#c8ccd8"

    for ax in [ax_score, ax_avg, ax_eps, ax_loss]:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.7)

    # ── Panel 1: Score raw ──────────────────────────────────
    ax_score.plot(scores, color="#4a9eff", alpha=0.5, linewidth=0.8,
                  label="Score")
    if len(moving_avg):
        ax_score.plot(avg_x, moving_avg, color="#ff9f43",
                      linewidth=2, label=f"Avg({window})")
    best_ep = int(np.argmax(scores))
    ax_score.scatter(best_ep, scores[best_ep], color="#ff4757",
                     zorder=5, s=60, label=f"Best={scores[best_ep]}")
    ax_score.set_title("Score mỗi episode")
    ax_score.set_xlabel("Episode")
    ax_score.set_ylabel("Score")
    ax_score.legend(fontsize=8, facecolor=BG, labelcolor=TEXT,
                    framealpha=0.8)

    # ── Panel 2: Moving average zoom ───────────────────────
    if len(moving_avg):
        ax_avg.plot(avg_x, moving_avg, color="#ff9f43",
                    linewidth=2)
        ax_avg.fill_between(avg_x, moving_avg, alpha=0.15,
                            color="#ff9f43")
        p25 = np.percentile(moving_avg, 25)
        p75 = np.percentile(moving_avg, 75)
        ax_avg.axhline(p25, color="#a29bfe", linewidth=1,
                       linestyle=":", label=f"P25={p25:.0f}")
        ax_avg.axhline(p75, color="#55efc4", linewidth=1,
                       linestyle=":", label=f"P75={p75:.0f}")
        ax_avg.legend(fontsize=8, facecolor=BG, labelcolor=TEXT,
                      framealpha=0.8)
    ax_avg.set_title(f"Moving Average (window={window})")
    ax_avg.set_xlabel("Episode")
    ax_avg.set_ylabel("Avg Score")

    # ── Panel 3: Epsilon decay ─────────────────────────────
    ax_eps.plot(eps_curve, color="#a29bfe", linewidth=2)
    ax_eps.fill_between(range(n_ep), eps_curve, alpha=0.15,
                        color="#a29bfe")
    # Đánh dấu điểm epsilon xuống dưới 0.1
    thresh = 0.1
    cross  = next((i for i, e in enumerate(eps_curve) if e <= thresh), None)
    if cross:
        ax_eps.axvline(cross, color="#fd79a8", linewidth=1.2,
                       linestyle="--",
                       label=f"ε≤{thresh} tại ep {cross}")
        ax_eps.legend(fontsize=8, facecolor=BG, labelcolor=TEXT,
                      framealpha=0.8)
    ax_eps.set_title("Epsilon (exploration rate)")
    ax_eps.set_xlabel("Episode")
    ax_eps.set_ylabel("Epsilon")
    ax_eps.set_ylim(0, 1.05)

    # ── Panel 4: Loss ───────────────────────────────────────
    if losses and len(losses) > 0:
        # Downsample nếu quá nhiều điểm
        step = max(1, len(losses) // 2000)
        loss_ds = losses[::step]
        ax_loss.plot(loss_ds, color="#55efc4", alpha=0.5,
                     linewidth=0.6, label="Loss")
        w2 = min(200, len(loss_ds))
        if len(loss_ds) >= w2:
            loss_ma = np.convolve(loss_ds,
                                  np.ones(w2) / w2, mode="valid")
            ax_loss.plot(range(w2 - 1, len(loss_ds)), loss_ma,
                         color="#fd79a8", linewidth=1.8,
                         label=f"Avg({w2})")
        ax_loss.legend(fontsize=8, facecolor=BG, labelcolor=TEXT,
                       framealpha=0.8)
    else:
        ax_loss.text(0.5, 0.5, "Chưa có dữ liệu loss",
                     ha="center", va="center",
                     transform=ax_loss.transAxes, color=TEXT)
    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("Bước học")
    ax_loss.set_ylabel("MSE Loss")

    # ── Thống kê tổng quan ──────────────────────────────────
    stats_text = (
        f"Episodes: {n_ep}  |  "
        f"Best: {max(scores) if scores else 0}  |  "
        f"Final avg({window}): "
        f"{np.mean(scores[-window:]):.1f}  |  "
        f"Final ε: {eps_curve[-1]:.3f}"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9,
             color="#74b9ff",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                       edgecolor=GRID))

    plt.savefig(save_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Biểu đồ đã lưu: {save_path}")


# ──────────────────────────────────────────────────────────
#  Train từ đầu
# ──────────────────────────────────────────────────────────
def train_from_scratch():
    ai = DQNDinoAI()

    scores = ai.train(
        n_episodes       = 2000,
        max_steps_per_ep = 10_000,
        verbose_every    = 50,
        save_path        = MODEL_PATH,
    )

    plot_scores(scores, losses=ai.losses)

    print("\n─── Đánh giá sau khi train ───")
    ai.epsilon = 0.0
    evaluate(ai, n_runs=10, verbose=True)


# ──────────────────────────────────────────────────────────
#  Tiếp tục train từ checkpoint
# ──────────────────────────────────────────────────────────
def resume_training():
    if not os.path.exists(MODEL_PATH):
        print(f"  X Không tìm thấy {MODEL_PATH}.")
        return

    ai = DQNDinoAI()
    ai.load_model(MODEL_PATH)
    ai.epsilon = max(0.1, ai.epsilon)

    scores = ai.train(
        n_episodes       = 400,
        max_steps_per_ep = 8_000,
        verbose_every    = 50,
        save_path        = MODEL_PATH,
    )

    plot_scores(scores, losses=ai.losses,
                save_path=CHART_PATH.replace(".png", "_resumed.png"))


# ──────────────────────────────────────────────────────────
#  Xem AI chơi
# ──────────────────────────────────────────────────────────
def watch():
    if not os.path.exists(MODEL_PATH):
        print(f"  X Không tìm thấy {MODEL_PATH}.")
        return

    ai = DQNDinoAI()
    ai.load_model(MODEL_PATH)
    ai.epsilon = 0.0
    watch_ai(ai, n_games=5)


# ──────────────────────────────────────────────────────────
#  Đánh giá (không render)
# ──────────────────────────────────────────────────────────
def eval_only():
    if not os.path.exists(MODEL_PATH):
        print(f"  X Không tìm thấy {MODEL_PATH}.")
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