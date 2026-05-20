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
    "buffer_capacity"   : 200_000,
    "per_alpha"         : 0.6,
    "per_beta_start"    : 0.4,
    "per_beta_end"      : 1.0,
    "per_beta_frames"   : 500_000,
    "per_epsilon"       : 1e-6,

    # Training
    "batch_size"        : 512,       # 256 → 512: batch lớn hơn = gradient ổn hơn
    "learn_start"       : 10_000,
    "lr"                : 3e-4,      # 1e-4 → 3e-4: LR cao hơn vì decay đã sửa
    "lr_decay"          : 0.9999990, # BUG FIX: 0.9999 → 0.9999990 (sau 200k steps: 0.82x, không về 0)
    "min_lr"            : 5e-5,      # sàn LR để không bao giờ về 0
    "gamma"             : 0.99,
    "tau"               : 0.003,     # 0.005 → 0.003: target update chậm hơn = ổn định hơn
    "grad_clip"         : 5.0,       # 10.0 → 5.0: kiểm soát gradient tốt hơn
    "dropout"           : 0.0,

    # Exploration
    "eps_start"         : 1.0,
    "eps_decay"         : 0.9960,    # 0.995 → 0.9960: chậm hơn, đạt eps_end ~ep 1150
    "eps_end"           : 0.01,      # 0.02 → 0.01: khám phá ít hơn ở giai đoạn cuối
    "eps_end_episode"   : 1500,      # 1200 → 1500: cho thêm thời gian khám phá

    # Misc
    "target_update_freq": 2000,
    "use_soft_update"   : True,
    "learn_every"       : 4,         # 2 → 4: pair với batch_size 512, cùng throughput
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

        print(f"[{self.name}] Initialized. Device: {self.device}")
        print(f"  Architecture: Dueling {s} -> {h} -> [V: {ah}->1 | A: {ah}->{a}]")
        print(f"  PER: alpha={self.cfg.get('per_alpha', 0.6)}, "
              f"capacity={self.cfg['buffer_capacity']:,}")

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

        # Double DQN: chọn action bằng q_net, đánh giá bằng target_net
        with torch.no_grad():
            best_a   = self.q_net(ns).argmax(1, keepdim=True)
            next_q   = self.target_net(ns).gather(1, best_a)
            target_q = r + self.cfg["gamma"] * next_q * (1 - d)

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

        # LR decay theo step, có sàn min_lr để không về 0
        min_lr = self.cfg.get("min_lr", 5e-5)
        lr_scale = max(min_lr / self.cfg["lr"], self.cfg["lr_decay"] ** self.steps)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.cfg["lr"] * lr_scale

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
        print(f"{'='*55}\n")

        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True
        )

        score_history = []
        best_score    = 0

        for ep in range(1, n_episodes + 1):
            policy.set_episode(ep)
            env   = DinoEnv(render=False, spawn_policy=policy)
            dino  = Dinosaur(env.sprites)

            # ── Speed-range coverage ──────────────────────────────
            # Episode bình thường kết thúc khi dino chết ở speed ~12, nên
            # buffer gần như không có transition tốc độ cao → Q-net chưa từng
            # học vùng đó → "phản ứng không kịp khi tốc độ tăng dần".
            # 50% episode bắt đầu ở tốc độ ngẫu nhiên trên toàn dải để nạp
            # thẳng transition tốc độ cao vào buffer. Các episode này chỉ dùng
            # để phủ state-space — KHÔNG tính vào logging/curriculum/best.
            if random.random() < 0.5:
                start_speed = random.uniform(INIT_SPEED, MAX_SPEED)
                randomized  = True
            else:
                start_speed = None
                randomized  = False
            state = env.reset(dino, start_speed=start_speed)

            ep_reward = 0
            for _ in range(max_steps_per_ep):
                action = self._select_action(state, training=True)
                next_state, env_reward, done, info = env.step_single(dino, action)

                self.buffer.push(state, action, env_reward, next_state, done)
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

            # Decay epsilon
            if ep <= self.cfg.get("eps_end_episode", 1200):
                self.epsilon = max(self.cfg["eps_end"],
                                   self.epsilon * self.cfg["eps_decay"])
            else:
                self.epsilon = self.cfg["eps_end"]

            ep_score = info["points"]
            self.generation = ep

            # Episode random-start chỉ để phủ dải tốc độ — điểm của nó thấp
            # giả tạo (bắt đầu giữa game) nên không đưa vào logging, không
            # dùng điều khiển curriculum, và không tính best score.
            if not randomized:
                score_history.append(ep_score)
                policy.update_performance(ep_score)

                if ep_score > best_score:
                    best_score      = ep_score
                    self.best_score = best_score
                    self.save_model(save_path)

            if ep % verbose_every == 0:
                recent    = score_history[-verbose_every:]
                avg_score = np.mean(recent)
                buf_size  = len(self.buffer)
                print(f"  Ep {ep:>5}/{n_episodes} | "
                      f"Score avg={avg_score:>7.1f} best={best_score:>6} | "
                      f"ε={self.epsilon:.3f} | "
                      f"loss={self._loss_ema:.4f} | "
                      f"diff={policy.difficulty:.2f} | "
                      f"buf={buf_size:>6}")

            env.close()

        print(f"\n  Done! Best score: {best_score}")
        print(f"  Model saved to: {save_path}\n")
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
            "generation"      : self.generation,
            "config"          : self.cfg,
            "model_version"   : 2,
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
        self.q_net.eval()
        print(f"[{self.name}] Loaded model from {path}  "
              f"(best={self.best_score}, gen={self.generation}, v{version})")
