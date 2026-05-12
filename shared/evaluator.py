# ============================================================
#  evaluator.py  –  Chạy thử & so sánh các AI (PHẦN CHUNG)
# ============================================================
import time
import numpy as np
import pygame
from shared.game_env import DinoEnv, Dinosaur
from shared.base_ai  import BaseDinoAI
from shared.config   import *


# ──────────────────────────────────────────────
#  Chạy 1 episode cho 1 AI (không render)
# ──────────────────────────────────────────────
def run_episode(ai: BaseDinoAI, render: bool = False,
                max_steps: int = MAX_STEPS_PER_EPISODE) -> dict:
    """
    Chạy 1 episode đến khi chết hoặc hết max_steps.

    Returns
    -------
    dict với các key:
        score (int), steps (int), survived (bool)
    """
    env  = DinoEnv(render=render)
    dino = Dinosaur(env.sprites)
    state = env.reset()

    total_reward = 0.0
    hi_score = 0
    final_score = 0
    ai_info = {
        "name": getattr(ai, "name", "AI"),
        "generation": getattr(ai, "generation", 0),
        "last_action": -1,
        "steps": 0,
        "speed": 0.0,
    }
    for _ in range(max_steps):
        if render:
            if not env.handle_events():
                break

        action = ai.predict(state)
        ai_info["last_action"] = action
        state, reward, done, info = env.step_single(dino, action)
        total_reward += reward
        ai_info["steps"] = dino.steps
        ai_info["speed"] = env.game_speed

        if render:
            env.render_frame(dino, hi_score, ai_info)
        if done:
            final_score = info["points"]
            if info["points"] > hi_score:
                hi_score = info["points"]
            break

    env.close()
    return {
        "score"   : final_score,
        "steps"   : dino.steps,
        "survived": not done,
    }


# ──────────────────────────────────────────────
#  Đánh giá AI qua N lần chạy
# ──────────────────────────────────────────────
def evaluate(ai: BaseDinoAI, n_runs: int = 5,
             verbose: bool = True) -> dict:
    """
    Chạy n_runs lần, tính trung bình / max / min.

    Dùng để so sánh các AI với nhau sau khi train.
    """
    scores = []
    for i in range(n_runs):
        result = run_episode(ai, render=False)
        scores.append(result["score"])
        if verbose:
            print(f"  [{ai.name}] Run {i+1}/{n_runs} → score={result['score']}")

    stats = {
        "ai"  : ai.name,
        "mean": float(np.mean(scores)),
        "max" : int(np.max(scores)),
        "min" : int(np.min(scores)),
        "std" : float(np.std(scores)),
    }
    if verbose:
        print(f"\n  ✅ {ai.name}: mean={stats['mean']:.0f}  "
              f"max={stats['max']}  min={stats['min']}  std={stats['std']:.0f}\n")
    return stats


# ──────────────────────────────────────────────
#  So sánh nhiều AI
# ──────────────────────────────────────────────
def compare_ais(ais: list[BaseDinoAI], n_runs: int = 5):
    """
    In bảng so sánh điểm giữa tất cả AI.

    Ví dụ:
        from evaluator import compare_ais
        compare_ais([ga_ai, pso_ai, dqn_ai], n_runs=10)
    """
    print("\n" + "="*55)
    print("  SO SÁNH CÁC AI")
    print("="*55)
    all_stats = []
    for ai in ais:
        stats = evaluate(ai, n_runs=n_runs, verbose=False)
        all_stats.append(stats)

    all_stats.sort(key=lambda s: s["mean"], reverse=True)

    print(f"\n{'Hang':<6}{'Ten AI':<20}{'Mean':>8}{'Max':>8}{'Min':>8}{'Std':>8}")
    print("-"*55)
    for rank, s in enumerate(all_stats, 1):
        print(f"  #{rank:<4}{s['ai']:<20}{s['mean']:>8.0f}"
              f"{s['max']:>8}{s['min']:>8}{s['std']:>8.0f}")
    print("="*55 + "\n")
    return all_stats


# ──────────────────────────────────────────────
#  Demo xem AI chơi (có render)
# ──────────────────────────────────────────────
def watch_ai(ai: BaseDinoAI, n_games: int = 3):
    """Xem AI chơi n_games ván có giao diện."""
    print(f"\nDang xem [{ai.name}] choi {n_games} van...\n")
    for i in range(n_games):
        result = run_episode(ai, render=True)
        print(f"  Van {i+1}: score={result['score']}")
    print()


# ──────────────────────────────────────────────
#  Fitness đơn giản dùng cho GA/PSO
# ──────────────────────────────────────────────
def fitness_single(ai: BaseDinoAI, n_avg: int = 3) -> float:
    """
    Chạy n_avg lần, trả về điểm trung bình (dùng làm fitness).
    Các thành viên GA/PSO gọi hàm này trong vòng lặp huấn luyện.
    """
    scores = [run_episode(ai, render=False)["score"] for _ in range(n_avg)]
    return float(np.mean(scores))
