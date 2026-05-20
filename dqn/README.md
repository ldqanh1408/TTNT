# DQN Dino AI — Deep Q-Network with Dueling + PER

## 1. State Vector (13D)

Each frame, `game_env.py:_build_state()` returns a `np.float32` array of 13 elements.

### 1.1. Full implementation (`game_env.py:752-793`)

```python
def _build_state(self, dino):
    state = [0.0] * STATE_SIZE  # 13

    sorted_obs = sorted(self.obs, key=lambda ob: ob.x)
    ref_x = dino.x if dino else 80

    # [0-3] + [4-7] 2 obstacles: dist, height, is_bird, bird_y
    for i in range(2):
        if i < len(sorted_obs):
            ob = sorted_obs[i]
            base = i * 4
            dist_px = ob.x - ref_x
            state[base + 0] = max(0.0, min(1.0, dist_px / SCREEN_W))    # dist
            state[base + 1] = min(1.0, ob.h / 160.0)                     # height
            state[base + 2] = 1.0 if ob.type_ == "bird" else 0.0          # is_bird
            state[base + 3] = ob.y / max(self.ground_y, 1) if ob.type_ == "bird" else 0.0  # bird_y

    # [8] Tốc độ
    state[8] = self.game_speed / MAX_SPEED

    # [9-10] Dino state
    if dino is not None:
        state[9]  = 1.0 if dino.is_jumping else 0.0
        state[10] = 1.0 if dino.is_ducking else 0.0

    # [11] Jump safety cho obs1
    if len(sorted_obs) >= 1 and self.game_speed > 0.1:
        ob1 = sorted_obs[0]
        dist_px = ob1.x - ref_x
        jump_dur = 2 * JUMP_VEL / GRAVITY
        max_jump = JUMP_VEL**2 / (2 * GRAVITY)
        if ob1.h < max_jump:
            t = dist_px / self.game_speed
            state[11] = max(0.0, min(1.0, (jump_dur - t) / jump_dur))

    # [12] Dino vertical velocity (âm = đang bay lên, dương = đang rơi)
    if dino is not None and dino.is_jumping:
        state[12] = dino._vel_y / JUMP_VEL
    else:
        state[12] = 0.0

    return np.array(state, dtype=np.float32)
```

### 1.2. Bảng tham chiếu index

#### Obstacle features (indices 0-9)

Hai obstacle gần nhất, sắp xếp theo `x` tăng dần. Mỗi obstacle 5 feature:

| Block | Index | Feature | Công thức | Khoảng giá trị | Ý nghĩa |
|---|---|---|---|---|---|
| **obs1** | 0 | `time_to_obs` | `min(1, (dist_px / game_speed) / 60)` | 0.0 … 1.0 | Số frame tới obstacle / 60 (nhận biết theo tốc độ) |
| | 1 | `height` | `min(1.0, ob.h / 160)` | 0.0 … 1.0 | Chiều cao obstacle / 160px |
| | 2 | `width` | `min(1.0, ob.w / 100)` | 0.0 … 1.0 | Chiều **rộng** obstacle / 100px — phân biệt small/double/big |
| | 3 | `is_bird` | `1.0 if ob.type_ == "bird" else 0.0` | 0 hoặc 1 | 1 = ptera (chim), 0 = xương rồng |
| | 4 | `action_hint` | `0.0` / `0.5` / `1.0` | {0, 0.5, 1} | 0 = NHẢY, 0.5 = CÚI, 1 = CHẠY QUA |
| **obs2** | 5-9 | — | Giống obs1 | — | Obstacle thứ hai (toàn 0 nếu không có) |

> `width` là feature thêm vào (state 13D → 15D): `cactus_small` (w≈26) và
> `cactus_double` (w≈54) **cùng height≈56** nên nếu thiếu width agent coi 2 loại
> như nhau, nhảy cùng kiểu → chết ở cây rộng / cụm xương rồng.

#### Game state (indices 10-13)

| Index | Feature | Công thức | Khoảng giá trị | Ý nghĩa |
|---|---|---|---|---|
| 10 | `speed` | `game_speed / MAX_SPEED` | 6/20=0.30 … 1.0 | Tốc độ hiện tại / 20 |
| 11 | `is_jumping` | `1.0 if dino.is_jumping else 0.0` | 0 hoặc 1 | Dino đang trong jump arc |
| 12 | `is_ducking` | `1.0 if dino.is_ducking else 0.0` | 0 hoặc 1 | Dino đang cúi |
| 13 | `remaining_airtime` | `min(1, max(0, t_land / jump_dur))` | 0.0 … 1.0 | Thời gian còn lại trên không / chu kỳ nhảy |

#### Dino vertical velocity (index 14)

| Index | Feature | Công thức | Khoảng giá trị | Ý nghĩa |
|---|---|---|---|---|
| 14 | `vel_y` | `dino._vel_y / JUMP_VEL` | -1.0 … 1.0 | Âm = đang bay lên, dương = đang rơi, 0 = đứng đất |

### 1.3. Jump safety chi tiết

Vật lý nhảy từ `config.py`:
```
JUMP_VEL = 18.5   px/frame — vận tốc ban đầu
GRAVITY  = 1.1    px/frame² — trọng lực

jump_duration = 2 × JUMP_VEL / GRAVITY      ≈ 33.6 frames
max_jump_h    = JUMP_VEL² / (2 × GRAVITY)   ≈ 155.6 px
```

Công thức:
```
t_to_obs   = dist_px / game_speed        — còn bao nhiêu frame đến obstacle
jump_safety = clamp((jump_duration - t_to_obs) / jump_duration, 0, 1)
```

| dist | speed=6 | speed=10 | speed=14 | Ý nghĩa |
|---|---|---|---|---|
| 400px | 0.00 | 0.00 | 0.05 | Quá xa — nhảy sẽ rơi trước khi gặp obstacle |
| 200px | 0.01 | 0.40 | 0.57 | Thời điểm bắt đầu nhảy tốt |
| 100px | 0.50 | 0.70 | 0.79 | Hơi muộn |
| 50px  | 0.75 | 0.85 | 0.89 | Gấp — nhảy ngay |

> Nếu `ob1.h >= max_jump_h` (obstacle quá cao, không nhảy qua được) → `jump_safety = 0.0`.
> Nếu không có obstacle nào trên màn hình → `jump_safety = 0.0`.

### 1.4. Ành xạ toàn bộ 13 chiều

```
State layout:
 ┌──────────┬──────────┬──────┬──────────┬──────┬───────────┬───────────┬───────────┬───────┬──────────┐
 │ 0    1  2│ 3    4  5│ 6   7│ 8        │ 9    │ 10        │ 11        │ 12        │       │          │
 │ dist  h  │ is  bird │ ...  │ speed    │ is   │ is        │ jump      │ vel_y     │       │          │
 │      │   │ _b   _y  │      │          │ jump │ duck      │ safety    │           │       │          │
 │ obs1     │          │ obs2 │ (norm.)  │ (0/1)│ (0/1)     │ (0..1)    │ (-1..1)   │       │          │
 └──────────┴──────────┴──────┴──────────┴──────┴───────────┴───────────┴───────────┴───────┴──────────┘
```

---

## 2. Action Space

| Action | Name | Game behavior |
|---|---|---|
| 0 | **DUCK** | Calls `dino.duck()` — shrinks hitbox, cannot jump while ducking |
| 1 | **JUMP** | Calls `dino.jump()` — starts jump at `JUMP_VEL=18.5` |
| 2 | **RUN**  | Calls `dino.unduck()` — stand/upright, default running state |

---

## 3. Reward Function

Implemented in `game_env.py:step_single()`:

```python
if done:
    reward = -1.0                            # death penalty
else:
    bonus = 0.0
    for ob in cleared:                       # each obstacle cleared this frame
        if ob.type_ == "bird":
            bird_dist = ground_y - ob.y - ob.h
            if bird_dist < 40:               # low bird → must jump
                bonus += 25.0
            elif bird_dist < 80:             # mid bird → jump or duck
                bonus += 15.0
            else:                            # high bird → should duck
                bonus += 30.0 if dino.is_ducking else 5.0
        else:
            bonus += 10.0                    # cactus cleared
    reward = 0.1 + bonus                     # +0.1 survival per frame
```

| Event | Reward | Explanation |
|---|---|---|
| Death (collision) | **-1.0** | Penalty |
| Survive 1 frame | **+0.1** | Encourage longevity |
| Clear cactus | **+10.0** | Jump over cactus |
| Clear low bird (< 40px) | **+25.0** | Must jump — no alternative |
| Clear mid bird (40-80px) | **+15.0** | Jump or duck both work |
| Clear high bird (> 80px) while **DUCKING** | **+30.0** | Optimal action |
| Clear high bird (> 80px) while **JUMPING** | **+5.0** | Survives but suboptimal |

---

## 4. Network Architecture — Dueling Q-Network

```
Input(13) ──> features ──> [Value head] ──> 1
          │                │
          │   Linear(13, 256)          Linear(128, 64)
          │   LayerNorm(256)           LayerNorm(64)
          │   ReLU()                   ReLU()
          │   Linear(256, 128)         Linear(64, 1)  →  V(s)
          │   LayerNorm(128)
          │   ReLU()
          │                │
          │                └──> [Advantage head] ──> 3
          │
          │                     Linear(128, 64)
          │                     LayerNorm(64)
          │                     ReLU()
          │                     Linear(64, 3)  →  A(s,a)
          │
          Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
```

**Why Dueling:** Separates "how good is this state" (Value) from "which action is better" (Advantage). For the Dino game, most states have similar value (running is good), but specific states demand one action (high bird = must duck). The advantage stream specializes in these action-specific decisions.

**Why LayerNorm over BatchNorm:** DQN trains online with small batches (256) from a non-stationary distribution. LayerNorm normalizes per-sample across features — independent of batch statistics — making it more stable for RL.

| Component | Specification |
|---|---|
| Input | 13D state vector |
| Feature extractor | 13 -> 256 -> 128 (LayerNorm + ReLU after each) |
| Value head | 128 -> 64 -> 1 (LayerNorm + ReLU) |
| Advantage head | 128 -> 64 -> 3 (LayerNorm + ReLU) |
| Weight init | Kaiming normal |
| Total parameters | **54,276** |

---

## 5. Algorithm — Double DQN + PER + Soft Target Update

### 5.1. Prioritized Experience Replay

Transitions are sampled proportional to their TD-error magnitude. Critical experiences (surprise deaths, rare pattern clears) are replayed more often instead of being buried in uniform buffer.

```
SumTree: O(log N) sampling, O(log N) priority updates
Priority(i) = (|TD_error_i| + ε) ** α
IS weight(i) = (N · P(i)) ** (-β)  / max_weight
β annealed: 0.4 → 1.0 over 500K steps
```

### 5.2. Double DQN

```python
# Online network selects action, target network evaluates it
best_a   = q_net(next_states).argmax(1)           # action selection
next_q   = target_net(next_states).gather(1, best_a)  # value estimation
target_q = rewards + gamma * next_q * (1 - dones)
```

### 5.3. PER-weighted Huber Loss

```python
elementwise_loss = SmoothL1Loss(current_q, target_q)  # per-sample
loss = (IS_weights * elementwise_loss).mean()           # weighted mean
```

### 5.4. Soft Target Update

```python
for target_param, online_param in zip(target_net.parameters(), q_net.parameters()):
    target_param = tau * online_param + (1 - tau) * target_param  # tau = 0.005
```

### 5.5. Learning Rate Schedule

```python
lr_scale = 0.9999 ** steps
lr = initial_lr * lr_scale    # initial_lr = 1e-4
```

### 5.6. Epsilon-Greedy Exploration

```python
if random() < epsilon:
    action = random.randint(0, 2)     # explore
else:
    action = argmax(q_net(state))     # exploit

# Episode-level decay (not step-level)
epsilon = max(eps_end, epsilon * eps_decay)    # 0.995^episode
```

| Episode | ε |
|---|---|
| 1 | 1.000 |
| 150 | 0.471 |
| 300 | 0.222 |
| 500 | 0.082 |
| 800 | 0.018 (hits eps_end) |
| 1200+ | 0.020 (hard switch to pure exploitation + noise) |

After `eps_end_episode=1200`, epsilon is locked at 0.02 for the remaining episodes.

---

## 6. Hyperparameters

### 6.1. DQN Configuration (`dqn_ai.py:DQN_CONFIG`)

| Category | Parameter | Value | Rationale |
|---|---|---|---|
| **Network** | `hidden_sizes` | `[256, 128]` | Deeper feature extraction |
| | `advantage_hidden` | `64` | Value/Advantage head size |
| | `state_size` | `13` | 13D state vector |
| | `action_size` | `3` | duck/jump/run |
| | `dropout` | `0.0` | Not needed with LayerNorm |
| **PER** | `per_alpha` | `0.6` | Prioritization exponent |
| | `per_beta_start` | `0.4` | IS correction start |
| | `per_beta_end` | `1.0` | Full IS correction |
| | `per_beta_frames` | `500_000` | Steps to anneal beta |
| | `per_epsilon` | `1e-6` | Avoid zero priority |
| **Training** | `batch_size` | `256` | Bigger batch for deeper net |
| | `learn_start` | `10_000` | Prefill buffer before learning |
| | `lr` | `1e-4` | Adam learning rate |
| | `lr_decay` | `0.9999` | Per-step LR decay |
| | `gamma` | `0.99` | Discount factor |
| | `tau` | `0.005` | Slow soft update for stability |
| | `grad_clip` | `10.0` | Relaxed clip for bigger net |
| | `learn_every` | `2` | Learn every N steps |
| **Buffer** | `buffer_capacity` | `200_000` | PER buffer capacity |
| **Explore** | `eps_start` | `1.0` | Full exploration at start |
| | `eps_decay` | `0.995` | Episode-level decay |
| | `eps_end` | `0.02` | Minimum exploration |
| | `eps_end_episode` | `1200` | Hard switch to ε=0.02 |

### 6.2. Game Physics (`shared/config.py`)

| Constant | Value | Meaning |
|---|---|---|
| `SCREEN_W` | 1200 | Screen width (px) |
| `SCREEN_H` | 450 | Screen height (px) |
| `FPS` | 60 | Frames per second |
| `GROUND_Y_OFFSET` | 20 | Ground offset from screen bottom |
| `GRAVITY` | 1.1 | Gravity (px/frame²) |
| `JUMP_VEL` | 18.5 | Initial jump velocity (px/frame) |
| `INIT_SPEED` | 6.0 | Starting game speed |
| `SPEED_INCREMENT` | 0.008 | Speed increase per frame |
| `MAX_SPEED` | 20.0 | Maximum game speed (cap để giữ game vượt được ở score cao) |
| `DINO_SCALE` | 0.45 | Dino sprite scaling |
| `OBSTACLE_SCALE` | 0.80 | Cactus sprite scaling |
| `BIRD_SCALE` | 0.70 | Ptera sprite scaling |

---

## 7. Adaptive Spawn Policy (`shared/spawn_policy.py`)

### 7.1. Core Design

`AdaptiveSpawnPolicy` adjusts difficulty in real-time based on the agent's actual performance — not a fixed linear schedule.

```
Performance feedback loop:
  1. Agent plays episode at current difficulty
  2. Score recorded in sliding window (last 20 episodes)
  3. Target difficulty adjusted:
       avg < 50   → decrease (agent struggling)
       avg 50-100 → decrease slightly
       avg 100-250 → hold / slow increase
       avg 250-500 → increase moderately
       avg 500-800 → increase
       avg > 800  → increase aggressively
  4. P-controller smoothly approaches target (coefficient 0.2)
```

### 7.2. Base Curriculum

Power curve: `difficulty = progress^1.3` — gentle early, aggressive late. The performance feedback can accelerate or decelerate this.

| Episode | progress | Base difficulty | Typical difficulty |
|---|---|---|---|
| 1 | 0.001 | 0.00 | ~0.10-0.15 |
| 100 | 0.05 | 0.02 | ~0.25-0.50 |
| 500 | 0.25 | 0.16 | ~0.60-0.90 |
| 1000 | 0.50 | 0.41 | ~0.80-1.00 |
| 1500 | 0.75 | 0.69 | ~0.95-1.00 |
| 2000 | 1.00 | 1.00 | ~1.00 |

### 7.3. Spawn Patterns by Difficulty

| Difficulty | Patterns unlocked | Bird rate |
|---|---|---|
| < 0.25 | single (95%), chain2 (5%) | 8-18% |
| 0.25 - 0.50 | + chain3 (5%), jump_duck (5%) | 18-34% |
| 0.50 - 0.75 | + sandwich (5-10%), + duck_jump (5-13%) | 34-45% |
| > 0.75 | All patterns at high weights | 45-55% |

**Pattern catalog:**
| Pattern | Sequence | Skills tested |
|---|---|---|
| `single` | 1 cactus | Basic jump |
| `chain2` | 2 cactus with gap | Double jump timing |
| `chain3` | 3 cactus with gaps | Triple jump timing |
| `jump_duck` | cactus -> mid/high bird | Jump then duck transition |
| `duck_jump` | high bird -> cactus | Duck then unduck + jump |
| `sandwich` | cactus -> bird -> cactus | Jump, duck, jump sequence |

### 7.4. Gap Scaling

Gaps shrink with difficulty using a power curve `d^1.2` — slow initial change, rapid late-change:

| Speed | Difficulty 0.2 | Difficulty 0.5 | Difficulty 1.0 |
|---|---|---|---|
| Slow (< 8) | 480-730 px | 370-560 px | 200-360 px |
| Mid (8-12) | 380-580 px | 290-440 px | 150-270 px |
| Fast (12-18) | 320-500 px | 240-380 px | 130-230 px |
| VFast (18-24) | 260-410 px | 200-320 px | 110-190 px |
| Max (> 24) | 200-330 px | 160-260 px | 90-180 px |

### 7.5. Other Spawn Features

- **No consecutive birds:** `last_type == "ptera"` always forces next to be cactus
- **Cactus size scales with speed/difficulty:** more big/double cactus at high speed
- **Bird height distribution:** `[low, mid, high]` weights shift toward `high` at higher difficulties — forcing more ducking decisions
- **Chain react frames:** Minimum frames between chain obstacles decreases from 8 (slow) to 3-4 (fast), tightening the execution window

---

## 8. Training Loop

```python
ai = DQNDinoAI()
policy = AdaptiveSpawnPolicy(max_episodes=2000)

for ep in range(1, 2001):
    policy.set_episode(ep)                  # Set difficulty for this episode
    env   = DinoEnv(render=False, spawn_policy=policy)
    dino  = Dinosaur(env.sprites)
    state = env.reset(dino)

    for step in range(max_steps):
        action = ai._select_action(state, training=True)  # epsilon-greedy
        next_state, reward, done, info = env.step_single(dino, action)

        ai.buffer.push(state, action, reward, next_state, done)
        state = next_state

        if ai.steps % 2 == 0:               # Learn every 2 steps
            ai._learn()                     # PER sampling + Dueling update

        if done: break

    ai.epsilon = max(eps_end, ai.epsilon * eps_decay)
    policy.update_performance(info["points"])  # Adaptive feedback

    if info["points"] > best_score:
        ai.save_model("dqn_best.pkl")
```

---

## 9. Save / Load Model

### Format (`dqn_best.pkl`)

```python
data = {
    "q_net_state":      OrderedDict,   # Online dueling network weights
    "target_net_state": OrderedDict,   # Target dueling network weights
    "epsilon":          float,         # Current epsilon
    "steps":            int,           # Total training steps
    "best_score":       int,           # Best episode score
    "generation":       int,           # Current episode
    "config":           dict,          # Full DQN_CONFIG
    "model_version":    2,             # v2 = Dueling + PER (v1 = old MLP)
}
pickle.dump(data, f)
```

### Usage

```python
ai = DQNDinoAI()
ai.load_model("model/dqn_best.pkl")      # Load trained model
ai.train(n_episodes=500)                 # Continue training
ai.save_model("model/dqn_v3.pkl")        # Save new checkpoint
```

**Note:** v1 models (old MLP Sequential) are incompatible with v2 (Dueling). Train from scratch.

---

## 10. How to Run

```bash
python dqn/train_dqn.py              # Train from scratch (2000 episodes)
python dqn/train_dqn.py --resume     # Resume from model/dqn_best.pkl
python dqn/train_dqn.py --eval       # Evaluate (20 runs, no render, epsilon=0)
python dqn/train_dqn.py --watch      # Watch AI play (5 games, render, epsilon=0)
```

Requires: `numpy`, `torch`, `pygame`, `matplotlib`

---

## 11. Results

| Metric | Target | Notes |
|---|---|---|
| Train best score | 3000+ | Single best episode during training |
| Eval mean (10 runs) | 500+ | Average with epsilon=0 |
| Eval min | 150+ | Worst run — adaptive curriculum ensures survivable patterns |
| Eval std | Under 2x mean | Lower relative variance |

The combination of Dueling architecture, Prioritized Experience Replay, and adaptive spawn curriculum targets more consistent performance across diverse obstacle patterns compared to the v1 MLP baseline (which had eval mean=182, std=103, best=1372).
