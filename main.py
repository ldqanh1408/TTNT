# ============================================================
#  main.py  –  Điểm vào chính: train + so sánh 3 AI
# ============================================================
"""
Cách dùng:
    python main.py --mode compare   # So sánh 3 AI (cần load model sẵn)
    python main.py --mode watch     # Xem AI tốt nhất chơi
    python main.py --mode manual    # Tự chơi tay
    python main.py --mode train_all # Huấn luyện cả 3 (mất nhiều thời gian)
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Import 3 AI của 3 thành viên (đổi đường dẫn khi có AI thật) ──
# from member1_ga_ai  import GeneticAlgorithmAI
# from member2_pso_ai import PSODinoAI
# from member3_dqn_ai import DQNDinoAI

# Dùng template để test khi chưa có AI thật
from template_ai       import TemplateAI
from shared.evaluator  import compare_ais, watch_ai
from shared.manual_play import manual_play


def main():
    parser = argparse.ArgumentParser(description="Dino AI Project")
    parser.add_argument("--mode", default="compare",
                        choices=["compare", "watch", "manual", "train_all"])
    args = parser.parse_args()

    # ── Tạo AI instances ──
    ais = [TemplateAI(), TemplateAI(), TemplateAI()]
    ais[0].name = "GA (Thành viên 1)"
    ais[1].name = "PSO (Thành viên 2)"
    ais[2].name = "DQN (Thành viên 3)"

    if args.mode == "compare":
        compare_ais(ais, n_runs=10)

    elif args.mode == "watch":
        best = max(ais, key=lambda a: a.best_score)
        watch_ai(best, n_games=5)

    elif args.mode == "manual":
        manual_play()

    elif args.mode == "train_all":
        print("⚠️  train_all: mỗi thành viên tự train trong file riêng!")
        print("   Chạy: python member1_ga_ai.py")
        print("         python member2_pso_ai.py")
        print("         python member3_dqn_ai.py")


if __name__ == "__main__":
    main()
