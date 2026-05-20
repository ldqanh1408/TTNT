# Proximal Policy Optimization (PPO) Dino AI

Kiến trúc PPO được tạo ra để thử nghiệm/so sánh với thuật toán DQN hiện có, dùng môi trường thiết lập sẵn từ dự án.

## 1. Mạng Nơ-ron (Actor-Critic)

```
Input(15) ──> Actor (Policy) ──> [128 -> 128] ──> logits(3)
          │
          └──> Critic (Value) ──> [128 -> 128] ──> V(s) (1)
```

## 2. Đặc điểm PPO
PPO kết hợp giữa Policy Gradient và Q-Learning (tính giá trị trạng thái để giảm variance).
Các tính năng bao gồm:
- **GAE (Generalized Advantage Estimation):** Dùng để tính toán lợi thế (Advantages) ổn định bằng tham số `gae_lambda=0.95`.
- **Clip Surrogate Objective:** Gradient clipping bằng `clip_epsilon=0.2` giúp ngăn policy thay đổi quá lớn ở mỗi cập nhật.
- **Entropy Bonus:** Giúp khám phá thêm trong giai đoạn đầu (`entropy_coef=0.01`).

## 3. Cách chạy
- Train từ đầu (2000 episodes):
  `python ppo/train_ppo.py`
- Tiếp tục train từ checkpoint (thêm 1000 episodes):
  `python ppo/train_ppo.py --resume`
- Xem AI đã train chơi:
  `python ppo/train_ppo.py --watch`
- Chạy hệ thống đánh giá ẩn (tốc độ cao):
  `python ppo/train_ppo.py --eval`
