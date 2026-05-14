# DQN Dino AI — Deep Q-Network Chi tiết

## 1. State Vector (13D)

| Index | Tên | Công thức | Ý nghĩa |
|---|---|---|---|
| 0 | **dist** | `(cluster_x - dino.x) / 1200` | Khoảng cách đến cụm (0=sát, 1=xa) |
| 1 | **width** | `cluster_width / 200` | Tổng độ rộng cụm |
| 2 | **max_h** | `max_height / 120` | Chiều cao nhất trong cụm |
| 3 | **has_bird** | `0`/`1` | Có chim trong cụm? |
| 4 | **bird_y** | `bird.y / ground_y` | Vị trí dọc chim |
| 5 | **bird_x** | `bird.x / 1200` | Tọa độ x chim |
| 6 | **jump_safety** | `(34 - dist/speed) / 34` | Nhảy bây giờ có clear? (1=an toàn) |
| 7 | **bird_high** | `0`/`1` | Chim cao → nên CÚI |
| 8 | **speed** | `game_speed / 16` | Tốc độ game |
| 9 | **dino_y** | `dino.y / ground_y` | Vị trí dọc dino |
| 10 | **dino_vel** | `dino.vel_y / 18.5` | Vận tốc dọc (âm=lên) |
| 11 | **is_jumping** | `0`/`1` | Đang nhảy? |
| 12 | **is_ducking** | `0`/`1` | Đang cúi? |

### Cụm obstacle (cluster)

Vật cản cách nhau ≤ 100px → gộp 1 cụm:
- **dist**: đến vật đầu tiên
- **width**: tổng rộng từ đầu đến cuối
- **max_h**: chiều cao nhất
- **bird_x/y/high**: thông tin chim nếu có

### Jump Safety

```
jump_duration = 2 × 18.5 / 1.1 ≈ 34 frames
jump_safety  = (34 - distance/speed) / 34   (clip 0→1)

  dist=300, speed=10 → time=30 → 0.12 (có thể nhảy)
  dist=170, speed=10 → time=17 → 0.50 (lý tưởng)
  dist=50,  speed=10 → time=5  → 0.85 (hơi trễ)
```

> max_h > 155px → jump_safety = 0 (không nhảy qua được)

### Bird High

```
bird_high = 1.0 nếu chim cách ground > 80px (chim cao → nên cúi)
            0.0 nếu không có chim hoặc chim thấp (phải nhảy)
```

Tín hiệu tường minh giúp AI biết ngay "gặp chim cao thì cúi".

## 2. Action Space

| Action | Hành động |
|---|---|
| 0 | **DUCK** |
| 1 | **JUMP** |
| 2 | **RUN** |

## 3. Reward Function

| Sự kiện | Reward |
|---|---|
| Chết | **-1.0** |
| Sống mỗi frame | **+0.1** |
| Vượt xương rồng | **+10.0** |
| Vượt chim thấp (<40px ground) | **+25.0** |
| Vượt chim giữa (40-80px) | **+15.0** |
| Vượt chim cao (>80px) + cúi | **+30.0** |
| Vượt chim cao (>80px) + nhảy | **+5.0** |

## 4. Network Architecture

```
Input(13) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(3)
```

~10K tham số. Double DQN + soft target update + Huber loss.

## 5. Hyperparameters

### Training

| Param | Value |
|---|---|
| `lr` | `5e-5` |
| `gamma` | `0.995` |
| `tau` | `0.02` |
| `batch_size` | `256` |
| `buffer_capacity` | `200_000` |
| `learn_start` | `5_000` |
| `learn_every` | `2` |
| `grad_clip` | `1.0` |
| `loss` | SmoothL1Loss |

### Exploration

| Param | Value |
|---|---|
| `eps_start` | `1.0` |
| `eps_decay` | `0.997` |
| `eps_end` | `0.02` |

## 6. Spawn Logic

### Xương rồng

| Tốc độ | Cluster |
|---|---|
| < 8 | 1-2 cây (70% 1) |
| 8-11 | 1-2 cây (50% 2) |
| > 11 | 1-3 cây |

### Chim (Ptera)

| Tốc độ | Tỉ lệ |
|---|---|
| ≤ 8 | 15% |
| 8-12 | 15-25% |
| > 12 | 25% |

Không spawn 2 lần liên tiếp. 3 độ cao: sát đất, giữa, cao.

## 7. Game Physics

| Hằng số | Value |
|---|---|
| `SCREEN_W` | 1200 |
| `SCREEN_H` | 450 |
| `GRAVITY` | 1.1 px/frame² |
| `JUMP_VEL` | 18.5 px/frame |
| `INIT_SPEED` | 6.0 |
| `MAX_SPEED` | 16.0 |
| `SPEED_INCREMENT` | 0.005/frame |

```
T_jump = 2 × 18.5 / 1.1 ≈ 34 frames
H_jump = 18.5² / 2.2 ≈ 155 px
```

## 8. Kết quả

| Metric | Value |
|---|---|
| Train best | **1372** |
| Eval mean | 182 |
| Eval max | 326 |

## 9. Cách chạy

```bash
python dqn/train_dqn.py              # Train từ đầu (2000 ep)
python dqn/train_dqn.py --resume     # Tiếp tục train
python dqn/train_dqn.py --eval       # Đánh giá (10 runs)
python dqn/train_dqn.py --watch      # Xem AI chơi (5 ván)
```
