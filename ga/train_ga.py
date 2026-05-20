# train_ga.py  –  Huấn luyện Genetic Algorithm cho Chrome Dino
# Cách chạy:
#   python ga/train_ga.py               # train từ đầu (mặc định)
#   python ga/train_ga.py --watch       # xem AI đã train chơi
#   python ga/train_ga.py --resume      # tiếp tục train từ checkpoint
#   python ga/train_ga.py --eval        # đánh giá không render
#   python ga/train_ga.py --gen 300     # train N thế hệ
#   python ga/train_ga.py --pop 50      # quần thể N cá thể

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
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ga.neural_network import DinoNet, NNConfig
from ga.ga_ai          import GeneticAlgorithm, GAConfig, GenomeIndividual
from shared.base_ai    import BaseDinoAI
from shared.evaluator  import run_episode, evaluate, watch_ai
from shared.config     import STATE_SIZE, ACTION_SIZE, MAX_STEPS_PER_EPISODE


MODEL_PATH = "model/best_dino_ga.pkl"
CHART_PATH = "model/ga_training_curve.png"


class GADinoAI(BaseDinoAI):
    def __init__(self,
                 ga_cfg:  GAConfig | None = None,
                 nn_cfg:  NNConfig | None = None,
                 name:    str = "GA-Dino"):
        super().__init__(name=name)
        self.ga_cfg = ga_cfg or GAConfig()
        self.nn_cfg = nn_cfg or NNConfig()
        self.ga: GeneticAlgorithm | None = None
        self.best_network: DinoNet | None = None
        GenomeIndividual._reset_counter()

    def predict(self, state: np.ndarray) -> int:
        if self.best_network is None:
            return np.random.randint(0, ACTION_SIZE)
        return self.best_network.predict(state)

    def train(self, n_generations: int = 300,
              fitness_evals: int = 3,
              verbose_every: int = 10,
              save_path: str = MODEL_PATH,
              **kwargs):
        print(f"\n{'='*60}")
        print(f"  🧬 KHỞI TẠO GA – {self.name}")
        print(f"{'='*60}")
        print(f"  Quần thể   : {self.ga_cfg.POPULATION_SIZE} cá thể")
        print(f"  Generations: {n_generations}")
        print(f"  Fitness/Eval: {fitness_evals} lần chạy / cá thể")
        print(f"  Crossover  : {self.ga_cfg.CROSSOVER_RATE*100:.0f}%  (uniform)")
        print(f"  Mutation   : {self.ga_cfg.MUTATION_RATE*100:.0f}%  "
              f"(gaussian, σ={self.ga_cfg.MUTATION_STRENGTH})")
        print(f"  Elitism    : {self.ga_cfg.ELITISM_COUNT} cá thể")
        print(f"  Selection  : tournament (k={self.ga_cfg.TOURNAMENT_SIZE})")
        print(f"  Network    : {self.nn_cfg.INPUT_SIZE}"
              f" → {self.nn_cfg.HIDDEN_SIZES}"
              f" → {self.nn_cfg.OUTPUT_SIZE}")
        print(f"  Params/Ind : {DinoNet(self.nn_cfg).num_params}")
        print(f"  Save path  : {save_path}")
        print(f"{'='*60}\n")

        self.ga = GeneticAlgorithm(ga_cfg=self.ga_cfg, nn_cfg=self.nn_cfg)
        self.ga.initialize_population()

        class _AIFitWrapper:
            __slots__ = ("network",)

            def __init__(self, individual: GenomeIndividual):
                self.network = individual.network

            def predict(self, state):
                return self.network.predict(state)

        def fitness_fn(individual: GenomeIndividual) -> float:
            ai = _AIFitWrapper(individual)
            result = run_episode(ai, render=False, max_steps=MAX_STEPS_PER_EPISODE)
            return float(result["score"])

        t_start = time.time()

        best = self.ga.evolve(
            n_generations=n_generations,
            fitness_fn=fitness_fn,
            verbose_every=verbose_every,
            save_path=save_path,
            checkpoint_every=kwargs.get("checkpoint_every", 50),
        )

        t_elapsed = time.time() - t_start
        self.best_network = best.network.copy()
        self.best_score  = best.fitness
        self.generation  = best.age

        self._plot_history(save_path.replace(".pkl", "_curve.png"))
        self._print_final_stats(t_elapsed)

    def save_model(self, path: str = MODEL_PATH):
        if self.best_network is None:
            print(f"  [!] Chưa train – không có gì để lưu.")
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.best_network.save(path)
        print(f"  [{self.name}] Đã lưu model → {path}  "
              f"(fitness={self.best_score:.2f}, gen={self.generation})")

    def load_model(self, path: str = MODEL_PATH):
        if not os.path.exists(path):
            print(f"  [!] Không tìm thấy file: {path}")
            return
        self.best_network = DinoNet.load(path)
        print(f"  [{self.name}] Đã tải model ← {path}")

    def _plot_history(self, save_path: str = CHART_PATH):
        if self.ga is None or not self.ga.history:
            return

        history = self.ga.history
        gens    = [h["generation"]    for h in history]
        bests   = [h["best_fitness"]  for h in history]
        avgs    = [h["avg_fitness"]  for h in history]
        worsts  = [h["worst_fitness"] for h in history]

        n = len(gens)
        window = max(3, n // 20)
        if n >= window:
            bests_ma = np.convolve(bests, np.ones(window)/window, mode="valid")
            avgs_ma  = np.convolve(avgs,  np.ones(window)/window, mode="valid")
            ma_x     = range(window - 1, n)
        else:
            bests_ma = bests
            avgs_ma  = avgs
            ma_x     = list(range(n))

        fig = plt.figure(figsize=(14, 9), facecolor="#0f1117")
        fig.suptitle(
            f"GA Chrome Dino – Training Dashboard  "
            f"(pop={self.ga_cfg.POPULATION_SIZE}, gen={n})",
            fontsize=14, fontweight="bold", color="white", y=0.98,
        )

        gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35,
                               left=0.08, right=0.95,
                               top=0.90, bottom=0.10)

        ax_best  = fig.add_subplot(gs[0, 0])
        ax_avg   = fig.add_subplot(gs[0, 1])
        ax_range = fig.add_subplot(gs[1, 0])
        ax_bar   = fig.add_subplot(gs[1, 1])

        BG   = "#1a1d27"
        GRID = "#2a2d3a"
        TEXT = "#c8ccd8"

        for ax in [ax_best, ax_avg, ax_range, ax_bar]:
            ax.set_facecolor(BG)
            ax.tick_params(colors=TEXT, labelsize=8)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID)
            ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.7)

        ax_best.plot(gens, bests, color="#4a9eff", alpha=0.4, linewidth=0.8, label="Best")
        ax_best.plot(gens, avgs,  color="#ff9f43", alpha=0.4, linewidth=0.8, label="Avg")
        if ma_x:
            ax_best.plot(ma_x, bests_ma, color="#4a9eff", linewidth=2.0, label=f"Best MA({window})")
            ax_best.plot(ma_x, avgs_ma,  color="#ff9f43", linewidth=2.0, label=f"Avg MA({window})")
        best_idx = int(np.argmax(bests))
        ax_best.scatter(gens[best_idx], bests[best_idx],
                       color="#ff4757", s=80, zorder=5, label=f"Peak={bests[best_idx]:.0f}")
        ax_best.set_title("Best & Avg Fitness / Generation")
        ax_best.set_xlabel("Generation")
        ax_best.set_ylabel("Score")
        ax_best.legend(fontsize=7, facecolor=BG, labelcolor=TEXT, framealpha=0.8)

        ax_avg.plot(gens, bests, color="#4a9eff", alpha=0.3, linewidth=0.6)
        if ma_x:
            ax_avg.plot(ma_x, bests_ma, color="#4a9eff", linewidth=2)
            ax_avg.fill_between(ma_x, bests_ma, alpha=0.1, color="#4a9eff")
        ax_avg.set_title("Best Fitness (Smoothed)")
        ax_avg.set_xlabel("Generation")
        ax_avg.set_ylabel("Score")

        ax_range.fill_between(gens, worsts, bests, alpha=0.3, color="#55efc4", label="Range")
        ax_range.plot(gens, bests,  color="#55efc4", linewidth=1, label="Best")
        ax_range.plot(gens, avgs,   color="#fd79a8", linewidth=1.5, linestyle="--", label="Avg")
        ax_range.plot(gens, worsts, color="#ff4757", linewidth=1, label="Worst")
        ax_range.set_title("Fitness Range (Best/Avg/Worst)")
        ax_range.set_xlabel("Generation")
        ax_range.set_ylabel("Score")
        ax_range.legend(fontsize=7, facecolor=BG, labelcolor=TEXT, framealpha=0.8)

        if self.ga.population:
            final_fit = [ind.fitness for ind in self.ga.population]
            ax_bar.hist(final_fit, bins=20, color="#a29bfe", edgecolor="#4a9eff", alpha=0.8)
            ax_bar.axvline(np.mean(final_fit), color="#ff9f43", linewidth=2,
                           linestyle="--", label=f"Mean={np.mean(final_fit):.1f}")
            ax_bar.axvline(np.max(final_fit), color="#55efc4", linewidth=2,
                           linestyle="--", label=f"Best={np.max(final_fit):.1f}")
            ax_bar.set_title("Fitness Distribution (Final Gen)")
            ax_bar.set_xlabel("Score")
            ax_bar.set_ylabel("Count")
            ax_bar.legend(fontsize=7, facecolor=BG, labelcolor=TEXT, framealpha=0.8)

        stats_text = (
            f"Pop={self.ga_cfg.POPULATION_SIZE}  |  "
            f"Gens={n}  |  "
            f"Best={max(bests):.0f}  |  "
            f"Final Avg={avgs[-1]:.1f}  |  "
            f"Mutation={self.ga_cfg.MUTATION_RATE*100:.0f}%  |  "
            f"σ={self.ga_cfg.MUTATION_STRENGTH}"
        )
        fig.text(0.5, 0.01, stats_text, ha="center", fontsize=9, color="#74b9ff",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=GRID))

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=130, facecolor=fig.get_facecolor())
        plt.close()
        print(f"\n  📊 Biểu đồ đã lưu: {save_path}")

    def _print_final_stats(self, elapsed: float):
        if self.ga is None or not self.ga.history:
            return
        h     = self.ga.history
        gens  = [x["generation"]     for x in h]
        bests = [x["best_fitness"]   for x in h]
        avgs  = [x["avg_fitness"]    for x in h]

        print(f"\n{'─'*60}")
        print(f"  TIẾN HOÁ HOÀN TẤT")
        print(f"  {'─'*60}")
        print(f"  Thời gian  : {elapsed/60:.1f} phút  ({elapsed:.1f}s)")
        print(f"  Thế hệ    : {len(gens)}")
        print(f"  Best Score: {max(bests):.0f}  (gen={gens[np.argmax(bests)]})")
        print(f"  Final Avg : {avgs[-1]:.1f}")
        print(f"  Best Model: {MODEL_PATH}")
        print(f"{'─'*60}\n")


# ─── Hàm chạy ────────────────────────────────────────────────

def train_from_scratch(args):
    ga_cfg = GAConfig(
        POPULATION_SIZE   = args.pop,
        ELITISM_COUNT     = max(1, args.pop // 20),
        TOURNAMENT_SIZE   = args.tournament_size,
        CROSSOVER_RATE    = args.crossover_rate,
        MUTATION_RATE     = args.mutation_rate,
        MUTATION_STRENGTH = args.mutation_strength,
        FITNESS_EVALS     = args.fitness_evals,
        RANDOM_SEED       = args.seed,
    )

    ai = GADinoAI(ga_cfg=ga_cfg, name=args.name or "GA-Dino")
    ai.train(
        n_generations=args.gen,
        fitness_evals=args.fitness_evals,
        verbose_every=args.verbose_every,
        save_path=args.save,
        checkpoint_every=args.checkpoint_every,
    )

    print("\n  ── Đánh giá 10 lần chạy ──")
    stats = evaluate(ai, n_runs=10, verbose=True)
    return ai, stats


def resume_training(args):
    if not os.path.exists(args.save):
        print(f"  X Không tìm thấy {args.save}")
        return

    net, meta = GeneticAlgorithm.load_best(args.save)

    ga_cfg = GAConfig(
        POPULATION_SIZE   = args.pop,
        MUTATION_RATE     = args.mutation_rate,
        MUTATION_STRENGTH = args.mutation_strength,
        FITNESS_EVALS     = args.fitness_evals,
        RANDOM_SEED       = args.seed,
    )

    ai = GADinoAI(ga_cfg=ga_cfg)
    ai.best_network = net
    ai.best_score   = meta["fitness"]
    ai.generation   = meta["generation"]

    print(f"\n  Tiếp tục train từ gen={meta['generation']}, "
          f"fitness={meta['fitness']:.2f}")

    ai.train(
        n_generations=args.gen,
        fitness_evals=args.fitness_evals,
        verbose_every=args.verbose_every,
        save_path=args.save,
        checkpoint_every=args.checkpoint_every,
    )

    print("\n  ── Đánh giá ──")
    evaluate(ai, n_runs=10, verbose=True)


def watch_trained(args):
    if not os.path.exists(args.save):
        print(f"  X Không tìm thấy {args.save}")
        return

    ai = GADinoAI()
    ai.load_model(args.save)
    print(f"\n  Đang xem [{ai.name}] chơi {args.games} ván...\n")
    watch_ai(ai, n_games=args.games)


def eval_only(args):
    if not os.path.exists(args.save):
        print(f"  X Không tìm thấy {args.save}")
        return

    ai = GADinoAI()
    ai.load_model(args.save)

    stats = evaluate(ai, n_runs=args.eval_runs, verbose=True)
    print(f"\n  Tổng kết: mean={stats['mean']:.0f}  "
          f"max={stats['max']}  std={stats['std']:.0f}")


# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genetic Algorithm Chrome Dino Trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python ga/train_ga.py                     # train 300 thế hệ, pop=50
  python ga/train_ga.py --gen 500 --pop 80  # 500 thế hệ, quần thể 80
  python ga/train_ga.py --watch             # xem AI đã train chơi
  python ga/train_ga.py --resume            # tiếp tục train
  python ga/train_ga.py --eval              # đánh giá không render
  python ga/train_ga.py --pop 30 --seed 42
        """,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--watch",  action="store_true", help="Xem AI đã train chơi")
    mode_group.add_argument("--resume", action="store_true", help="Tiếp tục train từ checkpoint")
    mode_group.add_argument("--eval",   action="store_true", help="Đánh giá không render")

    parser.add_argument("--gen",           type=int, default=300,
                        help="Số thế hệ (mặc định: 300)")
    parser.add_argument("--pop",           type=int, default=50,
                        help="Kích thước quần thể (mặc định: 50)")
    parser.add_argument("--fitness-evals", type=int, default=3,
                        help="Số lần chạy tính fitness/cá thể (mặc định: 3)")
    parser.add_argument("--crossover-rate",   type=float, default=0.80,
                        help="Xác suất crossover (mặc định: 0.80)")
    parser.add_argument("--mutation-rate",    type=float, default=0.10,
                        help="Tỷ lệ đột biến (mặc định: 0.10)")
    parser.add_argument("--mutation-strength", type=float, default=0.15,
                        help="Cường độ đột biến sigma (mặc định: 0.15)")
    parser.add_argument("--tournament-size",  type=int,   default=5,
                        help="Kích thước tournament (mặc định: 5)")
    parser.add_argument("--seed",      type=int,   default=None,
                        help="Random seed (mặc định: None/ngẫu nhiên)")
    parser.add_argument("--save",      type=str,   default=MODEL_PATH,
                        help=f"Đường dẫn lưu model (mặc định: {MODEL_PATH})")
    parser.add_argument("--verbose-every", type=int, default=10,
                        help="In log mỗi N thế hệ (mặc định: 10)")
    parser.add_argument("--name",  type=str, default="GA-Dino",
                        help="Tên AI (mặc định: GA-Dino)")
    parser.add_argument("--games",     type=int, default=5,
                        help="Số ván xem khi dùng --watch (mặc định: 5)")
    parser.add_argument("--eval-runs", type=int, default=20,
                        help="Số lần chạy khi dùng --eval (mặc định: 20)")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Lưu checkpoint mỗi N thế hệ (mặc định: 50, đặt 0 để tắt)")

    args = parser.parse_args()

    if args.watch:
        watch_trained(args)
    elif args.resume:
        resume_training(args)
    elif args.eval:
        eval_only(args)
    else:
        train_from_scratch(args)
