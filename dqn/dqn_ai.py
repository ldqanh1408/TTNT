# ============================================================
#  dqn_ai.py  –  Deep Q-Network cho Chrome Dino
#
#  Cấu trúc:
#    SumTree                – cây nhị phân cho Prioritized Replay
#    PrioritizedReplayBuffer – bộ nhớ ưu tiên theo TD-error
#    DuelingQNetwork        – mạng dueling: tách value & advantage
#    DQNDinoAI              – lớp AI chính, kế thừa BaseDinoAI
#
#  State (15 chiều):
#    [0-4]   obs1: time_to_obs, height, width, is_bird, action_hint
#    [5-9]   obs2: time_to_obs, height, width, is_bird, action_hint
#    [10]    game_speed / MAX_SPEED
#    [11]    is_jumping (0/1)
#    [12]    is_ducking (0/1)
#    [13]    remaining_airtime: thời gian còn lại trên không / jump_dur
#    [14]    dino vel_y / JUMP_VEL (âm=lên, dương=xuống)
#
#  width (obs *2): phân biệt cactus_small (w26) vs cactus_double (w54) —
#  2 loại cùng height=56, thiếu width agent nhảy giống nhau → chết ở cây rộng.
#
#  Action:
#    0 = cúi (duck)
#    1 = nhảy (jump)
#    2 = chạy (run)
# ============================================================

import os
import random
import pickle
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from shared.base_ai import BaseDinoAI


# ──────────────────────────────────────────────────────────
#  Cấu hình DQN
# ──────────────────────────────────────────────────────────
DQN_CONFIG = {
    # Mạng: Dueling 15→256→128 → [Value: 64→1, Advantage: 64→3]
    "hidden_sizes"      : [256, 128],
    "advantage_hidden"  : 64,
    "state_size"        : 15,
    "action_size"       : 3,

    # Prioritized Experience Replay
    "buffer_capacity"   : 500_000,    # 200k → 500k: giảm catastrophic forgetting
    "per_alpha"         : 0.6,
    "per_beta_start"    : 0.4,
    "per_beta_end"      : 1.0,
    "per_beta_frames"   : 1_000_000, # 500k → 1M: beta annealing chậm hơn cho buffer lớn
    "per_epsilon"       : 1e-6,

    # ── N-step returns (Rainbow-style multi-step bootstrap) ──
    # PPO dùng GAE λ=0.95 ≈ propagate reward ~20 steps trong 1 update.
    # DQN 1-step → +12 reward chỉ được lan ngược 1 step/learn → credit
    # assignment cực chậm với Chrome Dino (sparse +12 mỗi ~50 frames).
    # n=3 cân bằng bias/variance & cải thiện đáng kể tốc độ hội tụ.
    "n_step"            : 3,

    # Training
    "batch_size"        : 256,       # 512 → 256: lower replay ratio to prevent catastrophic forgetting
    "learn_start"       : 10_000,
    "lr"                : 3e-4,      # 1e-4 → 3e-4: LR cao hơn vì decay đã sửa
    "cosine_T_max"      : 4_000_000, # learn_every=2 → ~3.7M learn steps cho 2500 ep
    "min_lr"            : 5e-5,      # sàn LR để không bao giờ về 0
    "gamma"             : 0.99,
    "tau"               : 0.0015,    # 0.003 → 0.0015: slower target update for better Q-learning stability

    "grad_clip"         : 5.0,       # 10.0 → 5.0: kiểm soát gradient tốt hơn
    "dropout"           : 0.0,

    # Exploration
    "eps_start"         : 1.0,
    "eps_decay"         : 0.9980,    # 0.996 → 0.998: chậm hơn, đạt 0.01 ~ep 2300
    "eps_end"           : 0.01,
    "eps_end_episode"   : 3000,      # 1500 → 3000: không hard-stop epsilon giữa chừng
    "eps_diff_floor"    : 0.03,      # floor = difficulty * 0.03 — chỉ bind khi base ε rất thấp

    # Misc
    "target_update_freq": 2000,
    "use_soft_update"   : True,
    "learn_every"       : 2,         # 4 → 2: gấp đôi replay ratio. PPO làm 160 grad/4096 env
                                     # ≈ 0.04 ratio; DQN giờ 256/2=128 grad transitions/env step.

    # ── Random-speed coverage (an toàn hơn) ───────────────────
    # CŨ: 50% ep random trên [INIT, INIT+(MAX-INIT)*difficulty=30] @ ε=1.0 →
    #     dino chết trong 5-10 frame ở speed cao → PER buffer bị nuốt bởi
    #     "high-speed=die" transitions → Q-net pessimistic ở speed cao.
    # MỚI: 25% ep, cap ở 50% dải tốc độ, chỉ bật sau warmup_episodes để
    #      agent có thời gian học basics ở base speed trước.
    "random_start_prob"        : 0.25,
    "random_start_max_factor"  : 0.50,    # cap = INIT + (MAX-INIT)*0.50*difficulty
    "random_start_warmup_eps"  : 100,     # không random-start trước ep này
}


# ──────────────────────────────────────────────────────────
#  SumTree – cấu trúc dữ liệu cho Prioritized Replay
# ──────────────────────────────────────────────────────────
class SumTree:
    """Cây nhị phân hoàn chỉnh: lá = priority, nút trong = tổng."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.ptr = 0
        self._size = 0

    def add(self, priority: float, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def get_leaf(self, s: float):
        """Tìm lá có tổng tích lũy >= s. Trả về (idx, priority, data)."""
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                break
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total_priority(self) -> float:
        return self.tree[0]

    def __len__(self):
        return self._size


# ──────────────────────────────────────────────────────────
#  Prioritized Experience Replay Buffer
# ──────────────────────────────────────────────────────────
class PrioritizedReplayBuffer:
    """Bộ nhớ ưu tiên: sample theo TD-error magnitude."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        data = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        )
        self.tree.add(self.max_priority ** self.alpha, data)

    def sample(self, batch_size: int, beta: float):
        batch = []
        indices = []
        priorities = []

        total = max(self.tree.total_priority, 1e-10)
        segment = total / batch_size

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get_leaf(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        states, actions, rewards, next_states, dones = zip(*batch)

        # Importance sampling weights
        n = len(self.tree)
        probs = np.array(priorities) / total
        weights = (n * probs) ** (-beta)
        weights /= (weights.max() + 1e-10)

        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: list, td_errors: np.ndarray,
                          epsilon: float = 1e-6):
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self):
        return len(self.tree)


# ──────────────────────────────────────────────────────────
#  Dueling Q-Network
# ──────────────────────────────────────────────────────────
class DuelingQNetwork(nn.Module):
    """
    Kiến trúc Dueling: tách biệt Value V(s) và Advantage A(s,a).
    Q(s,a) = V(s) + A(s,a) - mean(A(s,.))

    Dùng LayerNorm thay vì BatchNorm → ổn định với batch nhỏ,
    không cần switch train/eval khi chọn action.
    """

    def __init__(self, state_size: int, action_size: int,
                 hidden_sizes: list, advantage_hidden: int = 64,
                 dropout: float = 0.0):
        super().__init__()

        # ── Feature extractor ──
        feat_layers = []
        in_size = state_size
        for h in hidden_sizes:
            feat_layers += [
                nn.Linear(in_size, h),
                nn.LayerNorm(h),
                nn.ReLU(),
            ]
            if dropout > 0:
                feat_layers.append(nn.Dropout(dropout))
            in_size = h
        self.features = nn.Sequential(*feat_layers)

        # ── Value head: V(s) ──
        self.value = nn.Sequential(
            nn.Linear(in_size, advantage_hidden),
            nn.LayerNorm(advantage_hidden),
            nn.ReLU(),
            nn.Linear(advantage_hidden, 1),
        )

        # ── Advantage head: A(s,a) ──
        self.advantage = nn.Sequential(
            nn.Linear(in_size, advantage_hidden),
            nn.LayerNorm(advantage_hidden),
            nn.ReLU(),
            nn.Linear(advantage_hidden, action_size),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                       nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        val = self.value(feat)          # (batch, 1)
        adv = self.advantage(feat)      # (batch, action_size)
        return val + adv - adv.mean(dim=1, keepdim=True)


# ──────────────────────────────────────────────────────────
#  DQN AI – kế thừa BaseDinoAI
# ──────────────────────────────────────────────────────────
class DQNDinoAI(BaseDinoAI):
    """
    Deep Q-Network Agent cho Chrome Dino.
    Dùng Dueling architecture + Prioritized Experience Replay.

    Ví dụ sử dụng:
    ---------------
    from dqn_ai import DQNDinoAI
    ai = DQNDinoAI()
    ai.train(n_episodes=2000)
    ai.save_model("models/dqn.pkl")
    """

    def __init__(self, config: dict = None, name: str = "DQN-Dino"):
        super().__init__(name=name)
        self.cfg    = config or DQN_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        s  = self.cfg["state_size"]
        a  = self.cfg["action_size"]
        h  = self.cfg["hidden_sizes"]
        ah = self.cfg.get("advantage_hidden", 64)
        d  = self.cfg.get("dropout", 0.0)

        self.q_net      = DuelingQNetwork(s, a, h, ah, dropout=d).to(self.device)
        self.target_net = DuelingQNetwork(s, a, h, ah, dropout=d).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(),
                                    lr=self.cfg["lr"])
        # PER cần loss từng mẫu riêng → reduction='none'
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        self.buffer = PrioritizedReplayBuffer(
            self.cfg["buffer_capacity"],
            alpha=self.cfg.get("per_alpha", 0.6),
        )

        self.epsilon    = self.cfg["eps_start"]
        self.steps      = 0
        self.losses     = []
        self._loss_ema  = 0.0

        # ── N-step bootstrap state ─────────────────────────────
        # transient queue gom các transition liên tiếp; flush vào PER khi đủ
        # n_step hoặc khi episode kết thúc (xem _push_transition).
        self.n_step      = int(self.cfg.get("n_step", 1))
        self.n_step_buffer = deque(maxlen=self.n_step)
        self._gamma_n    = self.cfg["gamma"] ** self.n_step

        # Rolling-avg model selection — lưu model khi rolling avg 30 ep > best_rolling_avg
        # (thay vì lưu theo 1 ván may rủi gây eval mean=58 dù best=549).
        self._eval_scores = deque(maxlen=30)
        self._best_rolling_avg = 0.0
        self._save_pending = False  # cờ để save model ở cuối episode sau khi đủ dữ liệu

        print(f"[{self.name}] Initialized. Device: {self.device}")
        print(f"  Architecture: Dueling {s} -> {h} -> [V: {ah}->1 | A: {ah}->{a}]")
        print(f"  PER: alpha={self.cfg.get('per_alpha', 0.6)}, "
              f"capacity={self.cfg['buffer_capacity']:,}")
        print(f"  N-step={self.n_step}, gamma^n={self._gamma_n:.4f}, "
              f"learn_every={self.cfg['learn_every']}")

    # ── Dự đoán action ─────────────────────────────────────

    def predict(self, state: np.ndarray) -> int:
        return self._select_action(state, training=False)

    # ── Chọn action (epsilon-greedy) ───────────────────────

    def _select_action(self, state: np.ndarray,
                       training: bool = True) -> int:
        eps = self.epsilon if training else 0.0
        if random.random() < eps:
            return random.randint(0, self.cfg["action_size"] - 1)

        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.q_net(t)
            return int(q.argmax().item())

    # ── N-step transition buffering ────────────────────────
    #
    # Mỗi transition đi vào n_step_buffer (deque maxlen=n). Khi buffer đầy
    # HOẶC gặp terminal, đẩy 1 transition (s_t, a_t, R_t^(n), s_{t+n}, done) vào
    # PER với R_t^(n) = Σ γ^k r_{t+k}. Trong _learn, target dùng γ^n cho bootstrap.
    #
    # Khi terminal (xảy ra giữa chừng), flush hết các shorter windows còn lại —
    # tất cả đều terminal=True nên bootstrap=0, chỉ cần R_t^(k) chính xác là đủ.

    def _compute_n_step_return(self) -> tuple:
        """Tính (rew_accum, next_state, done) từ n_step_buffer hiện tại."""
        rew       = 0.0
        gamma_pow = 1.0
        next_state = None
        done       = False
        gamma_b    = self.cfg["gamma"]
        for _, _, r, ns, d in self.n_step_buffer:
            rew       += gamma_pow * r
            gamma_pow *= gamma_b
            next_state = ns
            done       = d
            if d:
                break
        return rew, next_state, done

    def _push_transition(self, state, action, reward, next_state, done):
        """Đưa 1 transition vào n-step queue, đẩy head vào PER khi cần."""
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Chưa đủ n và chưa terminal → giữ trong queue
        if len(self.n_step_buffer) < self.n_step and not done:
            return

        # Đẩy head
        s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        rew_n, next_n, done_n = self._compute_n_step_return()
        self.buffer.push(s0, a0, rew_n, next_n, done_n)
        self.n_step_buffer.popleft()

        # Nếu terminal, flush tiếp các shorter windows còn lại
        if done:
            while self.n_step_buffer:
                s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
                rew_n, next_n, done_n = self._compute_n_step_return()
                self.buffer.push(s0, a0, rew_n, next_n, done_n)
                self.n_step_buffer.popleft()

    # ── Học từ replay buffer ───────────────────────────────

    def _learn(self):
        if len(self.buffer) < self.cfg["learn_start"]:
            return

        # Beta annealing: beta_start → beta_end theo steps
        beta = self.cfg["per_beta_start"] + \
               (self.cfg["per_beta_end"] - self.cfg["per_beta_start"]) * \
               min(1.0, self.steps / self.cfg["per_beta_frames"])

        states, actions, rewards, next_states, dones, indices, weights = \
            self.buffer.sample(self.cfg["batch_size"], beta)

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        w  = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        current_q = self.q_net(s).gather(1, a)

        # Double DQN + N-step: chọn action bằng q_net, đánh giá bằng target_net.
        # γ^n bootstrap — vì rewards đã được tích lũy n bước trong PER:
        #   R_t^(n) = r_t + γ r_{t+1} + ... + γ^(n-1) r_{t+n-1}
        #   target  = R_t^(n) + γ^n * Q_target(s_{t+n}, argmax) * (1 - done)
        with torch.no_grad():
            best_a   = self.q_net(ns).argmax(1, keepdim=True)
            next_q   = self.target_net(ns).gather(1, best_a)
            target_q = r + self._gamma_n * next_q * (1 - d)

        # PER: loss từng mẫu, nhân với IS weight
        elementwise_loss = self.loss_fn(current_q, target_q)
        loss = (w * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(),
                                max_norm=self.cfg.get("grad_clip", 10.0))
        self.optimizer.step()

        # Cập nhật priorities trong buffer
        td_errors = (target_q - current_q).detach().cpu().numpy().flatten()
        self.buffer.update_priorities(indices, td_errors,
                                     epsilon=self.cfg.get("per_epsilon", 1e-6))

        # Soft update target network
        if self.cfg["use_soft_update"]:
            tau = self.cfg["tau"]
            with torch.no_grad():
                for tp, op in zip(self.target_net.parameters(),
                                  self.q_net.parameters()):
                    tp.data.lerp_(op.data, tau)

        # Cosine LR annealing: lr → min_lr theo cosine curve.
        # Multiplicative decay cũ (lr_decay=0.999999) quá chậm — đến ep 2500
        # LR vẫn ~1.35e-4 → gradient updates mạnh phá policy đã hội tụ.
        # Cosine: lr giảm dần về min_lr, flatten ở cuối → ổn định policy.
        total_steps = self.cfg.get("cosine_T_max", 2_000_000)
        min_lr = self.cfg.get("min_lr", 5e-5)
        progress = min(1.0, self.steps / total_steps)
        lr_current = min_lr + (self.cfg["lr"] - min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_current

        # EMA loss để log mượt hơn
        alpha = 0.95
        self._loss_ema = alpha * self._loss_ema + (1 - alpha) * loss.item()
        self.losses.append(loss.item())

    def _hard_update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ── Vòng lặp train ─────────────────────────────────────

    def train(self, n_episodes: int = 500,
              max_steps_per_ep: int = 5_000,
              verbose_every: int = 50,
              save_path: str = "models/dqn_checkpoint.pkl",
              **kwargs):

        from shared.game_env import DinoEnv, Dinosaur
        from shared.spawn_policy import AdaptiveSpawnPolicy
        from shared.config import INIT_SPEED, MAX_SPEED

        policy = AdaptiveSpawnPolicy(max_episodes=n_episodes)

        print(f"\n{'='*55}")
        print(f"  DQN TRAINING START – {n_episodes} episodes")
        print(f"  Device: {self.device}")
        print(f"  Spawn policy: Adaptive (performance-based)")
        print(f"  PER: alpha={self.cfg.get('per_alpha', 0.6)}, "
              f"beta={self.cfg['per_beta_start']}->{self.cfg['per_beta_end']}")
        print(f"  eps_decay = {self.cfg['eps_decay']}  "
              f"(to {self.cfg['eps_end']} after ~"
              f"{int(-(np.log(self.cfg['eps_end'] / self.cfg['eps_start'])) / (-np.log(self.cfg['eps_decay'])))}"
              f" ep)")
        print(f"  eps floor = difficulty × {self.cfg.get('eps_diff_floor', 0.20):.2f}  "
              f"(exploration ∝ curriculum)")
        print(f"{'='*55}\n")

        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True
        )

        score_history = []
        best_score    = 0

        # ── Random-start hyperparams (từ DQN_CONFIG) ──────────────
        rs_prob   = self.cfg.get("random_start_prob", 0.25)
        rs_factor = self.cfg.get("random_start_max_factor", 0.50)
        rs_warmup = self.cfg.get("random_start_warmup_eps", 100)

        # Path lưu best-raw model — bên cạnh best rolling-avg ở save_path chính.
        # save_path: model selection trên rolling-avg 30 ep (ổn định)
        # raw_save_path: model selection trên best ep_score (PPO-style, agressive)
        # Final evaluation pick max(eval(rolling_avg_model), eval(raw_model))
        if save_path.endswith(".pkl"):
            raw_save_path = save_path[:-4] + "_raw.pkl"
        else:
            raw_save_path = save_path + "_raw"

        for ep in range(1, n_episodes + 1):
            policy.set_episode(ep)
            env   = DinoEnv(render=False, spawn_policy=policy)
            dino  = Dinosaur(env.sprites)

            # ── Speed-range coverage (an toàn hơn version cũ) ──────
            # CŨ: 50% random trên [INIT, MAX*difficulty=30] → ở ε=1.0 dino chết
            #     trong 5-10 frame ở speed cao → PER buffer bị dominate bởi
            #     high-speed-death transitions (PER priority = |TD| lớn) →
            #     Q-net học "high speed = die no matter what" → conservative.
            # MỚI: 25% sau warmup, cap ở 0.5*dải tốc độ → max=18 thay vì 30.
            #     Vẫn phủ được dải tốc độ trung-cao mà không poison buffer.
            #     Score CỦA random-start cũng được dùng cho curriculum (chứ
            #     không discard như cũ) — feedback đầy đủ cho P-controller.
            if ep > rs_warmup and random.random() < rs_prob:
                max_start = INIT_SPEED + (MAX_SPEED - INIT_SPEED) * rs_factor * policy.difficulty
                start_speed = random.uniform(INIT_SPEED, max_start)
                randomized  = True
            else:
                start_speed = None
                randomized  = False
            state = env.reset(dino, start_speed=start_speed)

            # Reset n-step queue đầu episode — tránh leak transition từ ep trước
            self.n_step_buffer.clear()

            ep_reward = 0
            for _ in range(max_steps_per_ep):
                action = self._select_action(state, training=True)
                next_state, env_reward, done, info = env.step_single(dino, action)

                # N-step buffering thay cho push trực tiếp
                self._push_transition(state, action, env_reward, next_state, done)
                state      = next_state
                ep_reward += env_reward
                self.steps += 1

                if self.steps % self.cfg["learn_every"] == 0:
                    self._learn()

                if not self.cfg["use_soft_update"]:
                    if self.steps % self.cfg["target_update_freq"] == 0:
                        self._hard_update_target()

                if done:
                    break

            # Decay epsilon — base decay + difficulty floor
            # Khi curriculum mở khóa pattern mới (difficulty cao), agent cần
            # exploration để học pattern đó. eps_diff_floor đảm bảo epsilon
            # không bao giờ xuống dưới difficulty * 0.03.
            diff_floor = max(0.0, policy.difficulty * self.cfg.get("eps_diff_floor", 0.03))
            if ep <= self.cfg.get("eps_end_episode", 3000):
                base_eps = self.epsilon * self.cfg["eps_decay"]
                self.epsilon = max(self.cfg["eps_end"], base_eps, diff_floor)
            else:
                self.epsilon = max(self.cfg["eps_end"], diff_floor)

            # Late-training stabilization: giảm tau (target net cập nhật chậm hơn)
            # và giữ epsilon tối thiểu 0.01 để tránh greedy collapse.
            progress = ep / max(1, n_episodes)
            if progress > 0.8:
                self.cfg["tau"] = 0.0005
                self.epsilon = max(0.01, self.epsilon)

            ep_score = info["points"]
            self.generation = ep

            # ── Score accounting ──────────────────────────────────
            # score_history & policy.update_performance: nhận TẤT CẢ episode
            #   → P-controller có feedback đầy đủ (random-start vẫn là gameplay
            #   thật, dù điểm thấp giả tạo do bắt đầu giữa game).
            # self._eval_scores (rolling-avg window): CHỈ normal start →
            #   tránh skewing baseline. Mixing random-start sẽ kéo rolling_avg
            #   xuống thấp giả tạo, làm threshold "beat best_rolling_avg" sai.
            score_history.append(ep_score)
            policy.update_performance(ep_score)

            # ── Model saving — DOUBLE TRACK ───────────────────────
            # Skip cả 2 nếu là random-start (điểm bị skewed thấp giả tạo
            # do bắt đầu giữa game).
            # (1) Best raw: PPO-style — ngay khi vượt best_score → save raw.
            #     Catch những moment đỉnh cao mà rolling avg miss.
            # (2) Best rolling-avg: lưu khi rolling_avg 30 ep > prev best.
            #     Đảm bảo model ổn định, không phải may rủi 1 ván.
            if not randomized:
                self._eval_scores.append(ep_score)

                if ep_score > best_score:
                    best_score = ep_score
                    self.best_score = best_score
                    self.save_model(raw_save_path)

                if len(self._eval_scores) >= 10:
                    rolling_avg = np.mean(self._eval_scores)
                    if rolling_avg > self._best_rolling_avg:
                        self._best_rolling_avg = rolling_avg
                        self.save_model(save_path)

            if ep % verbose_every == 0:
                recent    = score_history[-verbose_every:]
                avg_score = np.mean(recent)
                buf_size  = len(self.buffer)
                roll_info = f"roll={self._best_rolling_avg:.0f}" if len(self._eval_scores) >= 10 else "roll=--"
                print(f"  Ep {ep:>5}/{n_episodes} | "
                      f"Score avg={avg_score:>7.1f} best={best_score:>6} | "
                      f"ε={self.epsilon:.3f} | "
                      f"loss={self._loss_ema:.4f} | "
                      f"diff={policy.difficulty:.2f} | "
                      f"buf={buf_size:>6} | "
                      f"{roll_info}")

            env.close()

        print(f"\n  Done! Best score: {best_score}  "
              f"(best rolling-avg: {self._best_rolling_avg:.0f})")
        print(f"  Models saved:")
        print(f"    Rolling-avg → {save_path}")
        print(f"    Best-raw    → {raw_save_path}\n")
        return score_history

    # ── Lưu / Tải model ────────────────────────────────────

    def save_model(self, path: str):
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".",
            exist_ok=True
        )
        data = {
            "q_net_state"     : self.q_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "epsilon"         : self.epsilon,
            "steps"           : self.steps,
            "best_score"      : self.best_score,
            "best_rolling_avg": self._best_rolling_avg,
            "generation"      : self.generation,
            "config"          : self.cfg,
            "model_version"   : 4,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_model(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)

        version = data.get("model_version", 1)

        if version >= 2:
            # Dueling architecture — load trực tiếp
            self.q_net.load_state_dict(data["q_net_state"])
            self.target_net.load_state_dict(data["target_net_state"])
        else:
            # V1 model (old MLP Sequential) – incompatible
            print(f"[{self.name}] WARNING: V1 model incompatible with Dueling. "
                  f"Train from scratch.")
            self.epsilon    = data.get("epsilon",    self.cfg["eps_end"])
            self.steps      = 0
            self.best_score = 0
            self.generation = 0
            return

        self.epsilon    = data.get("epsilon",    self.cfg["eps_end"])
        self.steps      = data.get("steps",      0)
        self.best_score = data.get("best_score", 0)
        self.generation = data.get("generation", 0)
        self._best_rolling_avg = data.get("best_rolling_avg", 0.0)
        self.q_net.eval()
        print(f"[{self.name}] Loaded model from {path}  "
              f"(best={self.best_score}, roll_avg={self._best_rolling_avg:.0f}, "
              f"gen={self.generation}, v{version})")
