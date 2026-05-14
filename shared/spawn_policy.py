# spawn_policy.py  ← file mới, không đụng game_env.py
from __future__ import annotations
import random
from collections import deque
import numpy as np
from dataclasses import dataclass, field
from typing import Literal


PatternName = Literal[
    "single",       # 1 cactus đơn
    "chain2",       # 2 cactus gap sát → nhảy liên tiếp
    "chain3",       # 3 cactus gap sát → nhảy 3 lần liên tiếp
    "jump_duck",    # cactus → chim thấp/giữa  (nhảy rồi cúi)
    "duck_jump",    # chim cao → cactus         (cúi rồi nhảy)
    "sandwich",     # cactus → chim → cactus    (nhảy, cúi, nhảy)
]

CactusSize = Literal["small", "big", "double"]

# ────────────────────────────────────────────────
#  Tham số spawn — AI chỉ được học những thứ này
# ────────────────────────────────────────────────
@dataclass
class SpawnConfig:
    # Xác suất standalone bird (không tính pattern)
    ptera_prob_low:   float = 0.10
    ptera_prob_mid:   float = 0.15
    ptera_prob_high:  float = 0.20
    ptera_prob_vhigh: float = 0.25   # FIX: thêm vào – LearnableSpawnPolicy cần

    # Gap pixel giữa các nhóm (lo, hi)
    gap_slow:  tuple = (500, 800)
    gap_mid:   tuple = (400, 650)
    gap_fast:  tuple = (330, 520)
    gap_vfast: tuple = (260, 450)
    gap_max:   tuple = (200, 360)    # FIX: thêm vào – LearnableSpawnPolicy cần

    # Bird height weights [low, mid, high]
    bird_height_w: list = field(default_factory=lambda: [30, 45, 25])

    # Pattern weights — ưu tiên single, hạn chế pattern phức tạp
    pattern_w_slow:  list = field(default_factory=lambda: [90, 10,  0,  0,  0,  0])
    pattern_w_mid:   list = field(default_factory=lambda: [60, 30,  0, 10,  0,  0])
    pattern_w_fast:  list = field(default_factory=lambda: [40, 35,  5, 15,  5,  0])
    pattern_w_vfast: list = field(default_factory=lambda: [25, 35, 10, 20,  5,  5])

    # FIX: Cactus size weights [small, big, double] theo dải speed
    # Slow → chủ yếu small, dễ nhảy; Fast → big/double nhiều hơn, khó hơn
    cactus_size_w_slow:  list = field(default_factory=lambda: [60, 30, 10])
    cactus_size_w_mid:   list = field(default_factory=lambda: [45, 35, 20])
    cactus_size_w_fast:  list = field(default_factory=lambda: [30, 40, 30])
    cactus_size_w_vfast: list = field(default_factory=lambda: [20, 45, 35])

    # FIX: cluster_w_* cho decide_cactus_cluster (deprecated, dùng pattern system)
    # Giữ lại để không break LearnableSpawnPolicy._apply_curriculum
    cluster_w_slow:  list = field(default_factory=lambda: [60, 40,  0, 0])
    cluster_w_mid:   list = field(default_factory=lambda: [50, 40, 10, 0])
    cluster_w_fast:  list = field(default_factory=lambda: [35, 35, 20, 10])

    # Số frame phản ứng tối thiểu giữa các cactus trong chain pattern
    # (ít frame → khó hơn → nên GIẢM theo speed, không phải tăng)
    chain_react_frames_slow:  int = 7
    chain_react_frames_mid:   int = 5
    chain_react_frames_fast:  int = 4
    chain_react_frames_vfast: int = 3


_PATTERN_NAMES: list[PatternName] = [
    "single", "chain2", "chain3", "jump_duck", "duck_jump", "sandwich"
]
_CACTUS_SIZES: list[CactusSize] = ["small", "big", "double"]


# ────────────────────────────────────────────────
#  Policy — AI học ở đây, không đụng DinoEnv
# ────────────────────────────────────────────────
class SpawnPolicy:
    """
    Override class này để train AI spawn.
    DinoEnv chỉ gọi các method bên dưới.
    """

    def __init__(self, config: SpawnConfig | None = None):
        self.config = config or SpawnConfig()

    # ── 1. Loại obstacle nào spawn tiếp? ──────────────────────
    def decide_type(
        self,
        game_speed: float,
        last_type: str | None,
        points: int,
    ) -> Literal["ptera", "cactus"]:
        cfg = self.config
        # Không spawn 2 chim liên tiếp
        if last_type == "ptera":
            return "cactus"
        if game_speed <= 8:
            p = cfg.ptera_prob_low
        elif game_speed <= 12:
            p = cfg.ptera_prob_mid
        elif game_speed <= 18:
            p = cfg.ptera_prob_high
        else:
            p = cfg.ptera_prob_vhigh
        return "ptera" if random.random() < p else "cactus"

    # ── 2. Pattern nào cho lần spawn cactus này? ──────────────
    def decide_pattern(self, game_speed: float) -> PatternName:
        cfg = self.config
        if game_speed < 8:
            w = cfg.pattern_w_slow
        elif game_speed < 12:
            w = cfg.pattern_w_mid
        elif game_speed < 18:
            w = cfg.pattern_w_fast
        else:
            w = cfg.pattern_w_vfast
        return random.choices(_PATTERN_NAMES, weights=w)[0]

    # ── 3. Interval đến spawn tiếp ────────────────────────────
    def decide_interval(self, game_speed: float) -> int:
        cfg = self.config
        if game_speed <= 8:
            lo, hi = cfg.gap_slow
        elif game_speed <= 12:
            lo, hi = cfg.gap_mid
        elif game_speed <= 18:
            lo, hi = cfg.gap_fast
        elif game_speed <= 24:
            lo, hi = cfg.gap_vfast
        else:
            lo, hi = cfg.gap_max
        return max(15, int(random.randint(lo, hi) / max(game_speed, 0.1)))

    # ── 4. Kích thước cactus theo speed ───────────────────────
    def decide_cactus_size(self, game_speed: float) -> CactusSize:
        """
        FIX: Method mới — tách logic kích thước ra khỏi Obstacle.__init__.
        game_env._spawn_cactus_cluster dùng method này để tạo cactus nhất quán.
        """
        cfg = self.config
        if game_speed < 8:
            w = cfg.cactus_size_w_slow
        elif game_speed < 12:
            w = cfg.cactus_size_w_mid
        elif game_speed < 18:
            w = cfg.cactus_size_w_fast
        else:
            w = cfg.cactus_size_w_vfast
        return random.choices(_CACTUS_SIZES, weights=w)[0]

    # ── 5. Số frame reaction giữa các cactus trong chain ──────
    def decide_chain_react_frames(self, game_speed: float) -> int:
        """
        FIX: react_frames GIẢM theo speed (khó hơn ở tốc độ cao).
        Trước đây bị ngược: slow=2, vfast=7 → chain dễ hơn ở tốc độ cao!
        """
        cfg = self.config
        if game_speed <= 8:
            return cfg.chain_react_frames_slow
        elif game_speed <= 12:
            return cfg.chain_react_frames_mid
        elif game_speed <= 18:
            return cfg.chain_react_frames_fast
        else:
            return cfg.chain_react_frames_vfast

    # ── 6. Độ cao bird ────────────────────────────────────────
    def decide_bird_height(
        self,
        ground_top: int,
        bird_h: int,
        force: Literal["low", "mid", "high", "any"] = "any",
    ) -> int:
        heights = {
            "low":  ground_top - bird_h - 10,
            "mid":  ground_top - bird_h - 55,
            "high": ground_top - bird_h - 110,
        }
        if force != "any":
            return heights[force]
        choice = random.choices(
            ["low", "mid", "high"],
            weights=self.config.bird_height_w,
        )[0]
        return heights[choice]

    # ── 7. (Deprecated) Cluster size ──────────────────────────
    def decide_cactus_cluster(self, game_speed: float) -> int:
        """
        FIX: method giờ dùng cluster_w_* đã có trong SpawnConfig.
        Không còn crash vì thiếu attribute.
        Deprecated: pattern system (decide_pattern) thay thế hoàn toàn.
        """
        cfg = self.config
        if game_speed < 8:
            w = cfg.cluster_w_slow
        elif game_speed < 11:
            w = cfg.cluster_w_mid
        else:
            w = cfg.cluster_w_fast
        sizes = [i + 1 for i, wt in enumerate(w) if wt > 0]
        weights = [wt for wt in w if wt > 0]
        return random.choices(sizes, weights=weights)[0]


# ────────────────────────────────────────────────
#  Ví dụ: AI học bằng CMA-ES / RL đơn giản
#  Chỉ cập nhật SpawnConfig, không đụng DinoEnv
# ────────────────────────────────────────────────
class LearnableSpawnPolicy(SpawnPolicy):
    """
    Curriculum learning: tự động tăng độ khó theo episode.
    Cũng có thể bị RL agent điều khiển qua apply_params().
    """

    PARAM_DIM = 10

    def __init__(self, curriculum: bool = True, max_episodes: int = 2000):
        super().__init__()
        self.curriculum    = curriculum
        self.max_episodes  = max_episodes
        self.episode       = 0
        self._difficulty   = 0.0

        self.params = np.array([
            0.20, 0.28, 0.35,            # bird rates low/mid/high
            200.0, 350.0, 150.0, 260.0,  # gap slow lo/hi + gap vfast lo/hi
            30.0, 40.0, 30.0,            # bird height weights [low, mid, high]
        ], dtype=np.float32)

    # ── Curriculum ────────────────────────────────────────────

    def set_episode(self, ep: int):
        """Gọi đầu mỗi episode — tự động scale difficulty."""
        self.episode = ep
        if self.curriculum:
            self._difficulty = min(1.0, ep / self.max_episodes)
            self._apply_curriculum()

    def _apply_curriculum(self):
        d   = self._difficulty
        cfg = self.config

        # Bird rate: tăng đều theo difficulty
        cfg.ptera_prob_low   = 0.10 + d * 0.20
        cfg.ptera_prob_mid   = 0.15 + d * 0.25
        cfg.ptera_prob_high  = 0.20 + d * 0.25
        cfg.ptera_prob_vhigh = 0.25 + d * 0.25   # FIX: field đã tồn tại

        # Gap: thu hẹp dần theo difficulty
        cfg.gap_slow  = (int(500 - d * 200), int(800 - d * 350))
        cfg.gap_mid   = (int(400 - d * 200), int(650 - d * 300))
        cfg.gap_fast  = (int(330 - d * 150), int(520 - d * 250))
        cfg.gap_vfast = (int(260 - d * 110), int(450 - d * 200))
        cfg.gap_max   = (int(200 - d *  80), int(360 - d * 160))   # FIX: field đã tồn tại

        # Cluster weights (dùng cho decide_cactus_cluster nếu cần)
        if d < 0.2:
            cfg.cluster_w_slow = [60, 40,  0, 0]
            cfg.cluster_w_mid  = [50, 50,  0, 0]
            cfg.cluster_w_fast = [40, 40, 20, 0]
        elif d < 0.5:
            cfg.cluster_w_slow = [40, 40, 20, 0]
            cfg.cluster_w_mid  = [30, 40, 30, 0]
            cfg.cluster_w_fast = [25, 35, 25, 15]
        elif d < 0.8:
            cfg.cluster_w_slow = [30, 40, 30, 0]
            cfg.cluster_w_mid  = [25, 35, 30, 10]
            cfg.cluster_w_fast = [20, 30, 30, 20]
        else:
            cfg.cluster_w_slow = [25, 35, 30, 10]
            cfg.cluster_w_mid  = [20, 30, 30, 20]
            cfg.cluster_w_fast = [15, 25, 35, 25]

        # Cactus size: ép cactus lớn/double nhiều hơn ở difficulty cao
        big_w    = int(30 + d * 20)
        double_w = int(10 + d * 25)
        small_w  = max(5, 100 - big_w - double_w)
        cfg.cactus_size_w_slow  = [small_w + 10, big_w - 5,  max(0, double_w - 5)]
        cfg.cactus_size_w_mid   = [small_w + 5,  big_w,      double_w]
        cfg.cactus_size_w_fast  = [small_w,       big_w + 5,  double_w + 5]
        cfg.cactus_size_w_vfast = [max(5, small_w - 10), big_w + 10, double_w + 10]

        # Bird height: ép cả nhảy (low) lẫn cúi (high)
        low_w  = 30
        high_w = 25 + int(d * 20)
        mid_w  = max(5, 100 - low_w - high_w)
        cfg.bird_height_w = [low_w, mid_w, high_w]

        # Chain react frames: ép tighter theo difficulty
        cfg.chain_react_frames_slow  = max(3, 7 - int(d * 3))
        cfg.chain_react_frames_mid   = max(2, 5 - int(d * 2))
        cfg.chain_react_frames_fast  = max(2, 4 - int(d * 2))
        cfg.chain_react_frames_vfast = max(1, 3 - int(d * 1))

    # ── RL agent interface ────────────────────────────────────

    def apply_params(self, params: np.ndarray):
        """RL agent cập nhật param — override curriculum."""
        p = np.clip(params, 1e-3, None)
        cfg = self.config
        cfg.ptera_prob_low   = float(np.clip(p[0], 0.0, 1.0))
        cfg.ptera_prob_mid   = float(np.clip(p[1], 0.0, 1.0))
        cfg.ptera_prob_high  = float(np.clip(p[2], 0.0, 1.0))
        lo1, hi1 = sorted([int(p[3]), int(p[4])])
        lo2, hi2 = sorted([int(p[5]), int(p[6])])
        cfg.gap_slow  = (max(100, lo1), max(lo1 + 50, hi1))
        cfg.gap_vfast = (max(100, lo2), max(lo2 + 50, hi2))
        bw = p[7:10].tolist()
        cfg.bird_height_w = [max(1, int(x)) for x in bw]
        self.params = np.array(params, dtype=np.float32)

    def get_params(self) -> np.ndarray:
        return self.params.copy()


# ────────────────────────────────────────────────
#  AdaptiveSpawnPolicy – tự điều chỉnh độ khó
#  dựa trên performance thực tế của agent
# ────────────────────────────────────────────────
class AdaptiveSpawnPolicy(SpawnPolicy):
    """
    Chính sách spawn thích ứng:
      - Curriculum cơ bản: power curve (dễ đầu, khó cuối)
      - P-controller: điều chỉnh difficulty dựa trên điểm trung bình gần đây
      - Pattern phức tạp hơn ở difficulty cao (chain3, sandwich, duck_jump)
      - Gap chặt hơn, jitter ngẫu nhiên để tránh memorization

    Dùng trong DQNDinoAI.train():
        policy = AdaptiveSpawnPolicy(max_episodes=2000)
        for ep in range(n_episodes):
            policy.set_episode(ep)          # đầu episode
            ...
            policy.update_performance(score) # cuối episode
    """

    def __init__(self, max_episodes: int = 2000):
        super().__init__()
        self.max_episodes       = max_episodes
        self.episode            = 0
        self._difficulty        = 0.10
        self._target_difficulty = 0.18

        self.score_window = deque(maxlen=20)
        self._difficulty_history = []

    # ── Gọi đầu mỗi episode ──────────────────────────────────

    def set_episode(self, ep: int):
        self.episode = ep
        progress = ep / max(1, self.max_episodes)

        # Base curriculum: power curve 1.3 → dễ ở đầu, tăng tốc ở cuối
        base_difficulty = progress ** 1.3

        # Target không giảm dưới base (curriculum là sàn tối thiểu)
        self._target_difficulty = max(base_difficulty, self._target_difficulty)

        # P-controller: smooth approach to target (hệ số 0.2)
        self._difficulty += 0.2 * (self._target_difficulty - self._difficulty)
        self._difficulty = max(0.0, min(1.0, self._difficulty))

        self._apply_curriculum()
        self._difficulty_history.append(self._difficulty)

    # ── Gọi cuối mỗi episode ─────────────────────────────────

    def update_performance(self, episode_score: int):
        """
        Điều chỉnh target difficulty dựa trên điểm trung bình 20 episode gần nhất.

        Nguyên tắc:
          - Score thấp  → giảm difficulty (agent đang gặp khó)
          - Score cao   → tăng difficulty (agent đã sẵn sàng cho thử thách mới)
          - Score trung bình → giữ nguyên, cho agent thời gian củng cố
        """
        self.score_window.append(episode_score)
        if len(self.score_window) < 3:
            return

        avg = np.mean(self.score_window)

        if avg < 50:
            self._target_difficulty = max(0.05, self._difficulty - 0.10)
        elif avg < 100:
            self._target_difficulty = max(0.10, self._difficulty - 0.05)
        elif avg < 250:
            self._target_difficulty = min(0.55, self._difficulty + 0.015)
        elif avg < 500:
            self._target_difficulty = min(0.75, self._difficulty + 0.025)
        elif avg < 800:
            self._target_difficulty = min(0.90, self._difficulty + 0.035)
        else:
            self._target_difficulty = min(1.0, self._difficulty + 0.05)

    @property
    def difficulty(self) -> float:
        return self._difficulty

    # ── Áp curriculum vào SpawnConfig ────────────────────────

    def _apply_curriculum(self):
        d   = self._difficulty
        cfg = self.config

        # Bird probability — tăng dần, mạnh hơn ở cuối
        cfg.ptera_prob_low   = min(0.50, 0.08 + d * 0.42)
        cfg.ptera_prob_mid   = min(0.55, 0.12 + d * 0.43)
        cfg.ptera_prob_high  = min(0.50, 0.15 + d * 0.35)
        cfg.ptera_prob_vhigh = min(0.55, 0.20 + d * 0.35)

        # Gap — thu hẹp theo power curve (ép mạnh ở cuối)
        # Dùng d**1.2 để gap giảm chậm ở đầu, nhanh ở cuối
        d_gap = d ** 1.2
        cfg.gap_slow  = (int(550 - d_gap * 250), int(850 - d_gap * 400))
        cfg.gap_mid   = (int(450 - d_gap * 200), int(700 - d_gap * 300))
        cfg.gap_fast  = (int(380 - d_gap * 150), int(580 - d_gap * 250))
        cfg.gap_vfast = (int(300 - d_gap * 120), int(500 - d_gap * 200))
        cfg.gap_max   = (int(240 - d_gap * 100), int(400 - d_gap * 180))

        # Pattern weights — mở khóa pattern phức tạp dần
        if d < 0.25:
            cfg.pattern_w_slow  = [95,  5,  0,  0,  0,  0]
            cfg.pattern_w_mid   = [80, 15,  0,  5,  0,  0]
            cfg.pattern_w_fast  = [60, 25,  5,  5,  5,  0]
            cfg.pattern_w_vfast = [45, 30, 10,  5,  5,  5]
        elif d < 0.50:
            cfg.pattern_w_slow  = [75, 20,  5,  0,  0,  0]
            cfg.pattern_w_mid   = [50, 30, 10,  5,  5,  0]
            cfg.pattern_w_fast  = [35, 30, 15,  8,  7,  5]
            cfg.pattern_w_vfast = [25, 30, 20, 10,  8,  7]
        elif d < 0.75:
            cfg.pattern_w_slow  = [55, 28, 10,  5,  2,  0]
            cfg.pattern_w_mid   = [35, 28, 18, 10,  5,  4]
            cfg.pattern_w_fast  = [20, 25, 22, 13, 10, 10]
            cfg.pattern_w_vfast = [12, 22, 24, 15, 13, 14]
        else:
            cfg.pattern_w_slow  = [40, 28, 15, 10,  5,  2]
            cfg.pattern_w_mid   = [25, 25, 22, 13,  8,  7]
            cfg.pattern_w_fast  = [12, 20, 24, 17, 14, 13]
            cfg.pattern_w_vfast = [ 8, 16, 24, 18, 17, 17]

        # Cactus size — ép to hơn ở difficulty cao
        small_w  = max(10, int(60 - d * 40))
        big_w    = int(30 + d * 15)
        double_w = max(5, 100 - small_w - big_w)
        cfg.cactus_size_w_slow  = [small_w + 10, big_w - 3,  max(0, double_w - 7)]
        cfg.cactus_size_w_mid   = [small_w + 5,  big_w,       double_w - 5]
        cfg.cactus_size_w_fast  = [small_w,       big_w + 5,   double_w - 5]
        cfg.cactus_size_w_vfast = [max(5, small_w - 10), big_w + 8, double_w + 2]

        # Bird height — ép cả nhảy (low) lẫn cúi (high)
        low_w  = max(15, 30 - int(d * 15))
        high_w = min(50, 25 + int(d * 25))
        mid_w  = max(5, 100 - low_w - high_w)
        cfg.bird_height_w = [low_w, mid_w, high_w]

        # Chain react frames — chặt hơn ở difficulty cao
        cfg.chain_react_frames_slow  = max(3, 8 - int(d * 4))
        cfg.chain_react_frames_mid   = max(2, 6 - int(d * 3))
        cfg.chain_react_frames_fast  = max(2, 5 - int(d * 3))
        cfg.chain_react_frames_vfast = max(1, 4 - int(d * 2))