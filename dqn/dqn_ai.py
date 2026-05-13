# ============================================================
#  dqn_ai.py  –  Deep Q-Network cho Chrome Dino
#
#  Cấu trúc:
#    ReplayBuffer  – bộ nhớ kinh nghiệm (experience replay)
#    QNetwork      – mạng nơ-ron dự đoán Q-value
#    DQNDinoAI     – lớp AI chính, kế thừa BaseDinoAI
#
#  State (6 chiều):
#    [0] dist_to_obs     – khoảng cách đến chướng ngại / SCREEN_W
#    [1] obs_height      – độ cao đáy chướng ngại / ground_y
#    [2] obs_width       – chiều rộng chướng ngại / 60
#    [3] is_bird         – 1.0 nếu là chim, 0.0 nếu xương rồng
#    [4] bird_height     – chiều cao chim so với mặt đất
#    [5] speed_ratio     – game_speed / MAX_SPEED
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
import torch.optim as optim

from shared.base_ai import BaseDinoAI


# ──────────────────────────────────────────────────────────
#  Cấu hình DQN  (chỉnh ở đây, không cần sửa class bên dưới)
# ──────────────────────────────────────────────────────────
DQN_CONFIG = {
    # Kiến trúc mạng
    "hidden_sizes"   : [128, 128],   # kích thước các lớp ẩn
    "state_size"     : 12,
    "action_size"    : 3,

    # Replay buffer
    "buffer_capacity": 50_000,       # số transition lưu tối đa
    "batch_size"     : 64,

    # Học tập
    "lr"             : 1e-3,         # learning rate
    "gamma"          : 0.97,         # discount factor
    "tau"            : 0.005,        # soft-update target network

    # Epsilon-greedy (khám phá → khai thác)
    "eps_start"      : 1.0,
    "eps_end"        : 0.05,
    "eps_decay"      : 0.9995,       # nhân sau mỗi episode

    # Target network update
    "target_update_freq": 200,       # bước (step) giữa 2 lần hard-update
                                     # (chỉ dùng nếu KHÔNG dùng soft-update)
    "use_soft_update": True,         # True = soft-update mỗi bước, khuyến nghị

    # Training
    "learn_start"    : 1_000,        # bắt đầu học sau khi có đủ mẫu trong buffer
    "learn_every"    : 4,            # học 1 lần sau mỗi N bước
}


# ──────────────────────────────────────────────────────────
#  Replay Buffer
# ──────────────────────────────────────────────────────────
class ReplayBuffer:
    """Bộ nhớ kinh nghiệm cho DQN."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ──────────────────────────────────────────────────────────
#  Q-Network (MLP đơn giản)
# ──────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """Mạng nơ-ron dự đoán Q(s, a) cho mọi action cùng lúc."""

    def __init__(self, state_size: int, action_size: int,
                 hidden_sizes: list):
        super().__init__()
        layers = []
        in_size = state_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        layers.append(nn.Linear(in_size, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────
#  DQN AI  –  kế thừa BaseDinoAI
# ──────────────────────────────────────────────────────────
class DQNDinoAI(BaseDinoAI):
    """
    Deep Q-Network Agent cho Chrome Dino.

    Ví dụ sử dụng:
    ---------------
    from dqn_ai import DQNDinoAI
    ai = DQNDinoAI()
    ai.train(n_episodes=500)
    ai.save_model("models/dqn.pkl")
    """

    def __init__(self, config: dict = None, name: str = "DQN-Dino"):
        super().__init__(name=name)
        self.cfg   = config or DQN_CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        s = self.cfg["state_size"]
        a = self.cfg["action_size"]
        h = self.cfg["hidden_sizes"]

        # Hai mạng: online (học) và target (ổn định)
        self.q_net      = QNetwork(s, a, h).to(self.device)
        self.target_net = QNetwork(s, a, h).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(),
                                    lr=self.cfg["lr"])
        self.loss_fn   = nn.MSELoss()

        self.buffer  = ReplayBuffer(self.cfg["buffer_capacity"])
        self.epsilon = self.cfg["eps_start"]
        self.steps   = 0          # tổng bước đã đi
        self.losses  = []         # lịch sử loss để theo dõi

        print(f"[{self.name}] Khởi tạo xong. Device: {self.device}")
        print(f"  Mạng: {s} → {h} → {a}")

    # ── Dự đoán action (dùng khi play / evaluate) ──────────

    def predict(self, state: np.ndarray) -> int:
        """Greedy: chọn action có Q-value cao nhất (epsilon=0)."""
        return self._select_action(state, training=False)

    # ── Chọn action (epsilon-greedy khi train) ─────────────

    def _select_action(self, state: np.ndarray,
                       training: bool = True) -> int:
        eps = self.epsilon if training else 0.0
        if random.random() < eps:
            return random.randint(0, self.cfg["action_size"] - 1)

        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.q_net(t)
            return int(q.argmax().item())

    # ── Học từ một batch trong replay buffer ───────────────

    def _learn(self):
        if len(self.buffer) < self.cfg["learn_start"]:
            return

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.cfg["batch_size"])

        s  = torch.FloatTensor(states).to(self.device)
        a  = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d  = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q hiện tại
        current_q = self.q_net(s).gather(1, a)

        # Q target (Bellman)
        with torch.no_grad():
            # Double DQN: online chọn action, target tính giá trị
            best_a  = self.q_net(ns).argmax(1, keepdim=True)
            next_q  = self.target_net(ns).gather(1, best_a)
            target_q = r + self.cfg["gamma"] * next_q * (1 - d)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping tránh exploding gradient
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft-update target network
        if self.cfg["use_soft_update"]:
            tau = self.cfg["tau"]
            for tp, op in zip(self.target_net.parameters(),
                              self.q_net.parameters()):
                tp.data.copy_(tau * op.data + (1 - tau) * tp.data)

        self.losses.append(loss.item())

    # ── Hard-update target network ─────────────────────────

    def _hard_update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ── Vòng lặp train ─────────────────────────────────────

    def train(self, n_episodes: int = 500,
              max_steps_per_ep: int = 5_000,
              verbose_every: int = 50,
              save_path: str = "models/dqn_checkpoint.pkl",
              **kwargs):
        """
        Huấn luyện DQN qua n_episodes episode.

        Parameters
        ----------
        n_episodes      : số episode huấn luyện
        max_steps_per_ep: bước tối đa mỗi episode (tránh vòng lặp mãi)
        verbose_every   : in log sau mỗi N episode
        save_path       : đường dẫn lưu checkpoint tốt nhất
        """
        from shared.game_env import DinoEnv, Dinosaur

        print(f"\n{'='*55}")
        print(f"  BẮT ĐẦU HUẤN LUYỆN DQN – {n_episodes} episodes")
        print(f"  Device: {self.device}")
        print(f"{'='*55}\n")

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        score_history = []
        best_score    = 0

        for ep in range(1, n_episodes + 1):
            env   = DinoEnv(render=False)
            dino  = Dinosaur(env.sprites)
            state = env.reset(dino)

            ep_reward = 0
            ep_steps  = 0

            for _ in range(max_steps_per_ep):
                action = self._select_action(state, training=True)
                next_state, reward, done, info = env.step_single(dino, action)

                # Reward shaping: thưởng thêm khi vừa né được chướng ngại
                shaped_reward = reward
                if done:
                    shaped_reward = -100.0   # phạt nặng khi chết
                elif reward > 0:
                    shaped_reward = 1.0 + env.game_speed * 0.1  # thưởng tăng dần

                self.buffer.push(state, action, shaped_reward, next_state, done)
                state      = next_state
                ep_reward += shaped_reward
                ep_steps  += 1
                self.steps += 1

                # Học sau mỗi learn_every bước
                if self.steps % self.cfg["learn_every"] == 0:
                    self._learn()

                # Hard-update (nếu không dùng soft-update)
                if not self.cfg["use_soft_update"]:
                    if self.steps % self.cfg["target_update_freq"] == 0:
                        self._hard_update_target()

                if done:
                    break

            # Giảm epsilon sau mỗi episode
            self.epsilon = max(self.cfg["eps_end"],
                               self.epsilon * self.cfg["eps_decay"])

            ep_score = info["points"]
            score_history.append(ep_score)
            self.generation = ep

            if ep_score > best_score:
                best_score       = ep_score
                self.best_score  = best_score
                self.save_model(save_path)

            # Log
            if ep % verbose_every == 0:
                recent     = score_history[-verbose_every:]
                avg_score  = np.mean(recent)
                avg_loss   = np.mean(self.losses[-200:]) if self.losses else 0
                buf_size   = len(self.buffer)
                print(f"  Ep {ep:>5}/{n_episodes} | "
                      f"Score avg={avg_score:>7.1f} best={best_score:>6} | "
                      f"ε={self.epsilon:.3f} | "
                      f"loss={avg_loss:.4f} | "
                      f"buf={buf_size:>6}")

            env.close()

        print(f"\n  ✅ Xong! Best score: {best_score}")
        print(f"  Model đã lưu tại: {save_path}\n")
        return score_history

    # ── Lưu / Tải model ────────────────────────────────────

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "q_net_state"     : self.q_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "epsilon"         : self.epsilon,
            "steps"           : self.steps,
            "best_score"      : self.best_score,
            "generation"      : self.generation,
            "config"          : self.cfg,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_model(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_net.load_state_dict(data["q_net_state"])
        self.target_net.load_state_dict(data["target_net_state"])
        self.epsilon     = data.get("epsilon",    self.cfg["eps_end"])
        self.steps       = data.get("steps",      0)
        self.best_score  = data.get("best_score", 0)
        self.generation  = data.get("generation", 0)
        self.q_net.eval()
        print(f"[{self.name}] Đã tải model từ {path}  "
              f"(best={self.best_score}, gen={self.generation})")