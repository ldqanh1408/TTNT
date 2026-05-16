import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from shared.base_ai import BaseDinoAI

PPO_CONFIG = {
    "state_size": 15,
    "action_size": 3,
    "hidden_sizes": [256, 128],
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "n_rollout_steps": 4096,
    "batch_size": 256,
    "n_epochs": 10,
}

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes):
        super(ActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_size, hidden_sizes[0])),
            nn.LayerNorm(hidden_sizes[0]),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.LayerNorm(hidden_sizes[1]),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[1], action_size), std=0.01)
        )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_size, hidden_sizes[0])),
            nn.LayerNorm(hidden_sizes[0]),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            nn.LayerNorm(hidden_sizes[1]),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[1], 1), std=1.0)
        )

    def forward(self, x):
        pass # typically we call get_action or get_value

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

class PPODinoAI(BaseDinoAI):
    def __init__(self, config=None):
        super().__init__(name="PPO_AI")
        self.cfg = config if config else PPO_CONFIG
        self.device = torch.device("cpu")
        
        self.network = ActorCritic(
            self.cfg["state_size"], 
            self.cfg["action_size"], 
            self.cfg["hidden_sizes"]
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.cfg["learning_rate"], eps=1e-5)
        self.buffer = PPOBuffer()

        self.steps_done = 0
        self.best_score = 0

    def predict(self, state: np.ndarray) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _, _ = self.network.get_action_and_value(state_t)
        return action.item()

    def update(self, next_state, next_done):
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.buffer.actions)).to(self.device)
        logprobs = torch.FloatTensor(np.array(self.buffer.logprobs)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.buffer.rewards)).to(self.device)
        values = torch.FloatTensor(np.array(self.buffer.values)).to(self.device).squeeze(-1)
        dones = torch.FloatTensor(np.array(self.buffer.dones)).to(self.device)

        # Calculate advantages using GAE
        with torch.no_grad():
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            next_value = self.network.get_value(next_state_t).squeeze(-1)
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
                delta = rewards[t] + self.cfg["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + self.cfg["gamma"] * self.cfg["gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_inds = np.arange(self.cfg["n_rollout_steps"])
        clipfracs = []

        for epoch in range(self.cfg["n_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, self.cfg["n_rollout_steps"], self.cfg["batch_size"]):
                end = start + self.cfg["batch_size"]
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.network.get_action_and_value(states[mb_inds], actions[mb_inds])
                logratio = newlogprob - logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.cfg["clip_epsilon"]).float().mean().item()]

                mb_advantages = advantages[mb_inds]
                # Normalizing advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.cfg["clip_epsilon"], 1 + self.cfg["clip_epsilon"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.cfg["entropy_coef"] * entropy_loss + v_loss * self.cfg["value_coef"]

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg["max_grad_norm"])
                self.optimizer.apply_gradients() if hasattr(self.optimizer, 'apply_gradients') else self.optimizer.step()

        self.buffer.clear()
        
    def train(self, n_episodes: int = 500, max_steps_per_ep: int = 5_000, 
              save_path: str = "models/ppo_checkpoint.pkl", **kwargs):
              
        from shared.game_env import DinoEnv, Dinosaur
        from shared.spawn_policy import AdaptiveSpawnPolicy
        from shared.config import INIT_SPEED, MAX_SPEED

        policy = AdaptiveSpawnPolicy(max_episodes=n_episodes)

        print(f"\n{'='*55}")
        print(f"  PPO TRAINING START – {n_episodes} episodes")
        print(f"  Network: {self.cfg['hidden_sizes']}")
        print(f"  Batch/Rollout: {self.cfg['batch_size']}/{self.cfg['n_rollout_steps']}")
        print(f"  Device: {self.device}")
        print(f"{'='*55}\n")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        env = DinoEnv(render=False, spawn_policy=policy)
        dino = Dinosaur(env.sprites)

        scores_history = []
        episode = 0
        state = env.reset(dino)
        done = False
        
        while episode < n_episodes:
            for step in range(self.cfg["n_rollout_steps"]):
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, logprob, _, value = self.network.get_action_and_value(state_t)
                
                action_item = action.item()
                value_item = value.item()
                logprob_item = logprob.item()

                next_state, reward, done, _ = env.step_single(dino, action_item)

                self.buffer.states.append(state)
                self.buffer.actions.append(action_item)
                self.buffer.logprobs.append(logprob_item)
                self.buffer.rewards.append(reward)
                self.buffer.values.append(value_item)
                self.buffer.dones.append(done)

                state = next_state
                
                if done:
                    scores_history.append(dino.score)
                    if dino.score > self.best_score:
                        self.best_score = dino.score
                        self.save_model(save_path.replace('.pkl', '_best.pkl'))

                    if (episode + 1) % 50 == 0:
                        avg_sc = np.mean(scores_history[-50:])
                        print(f"Episode {episode + 1:4d}/{n_episodes} | Avg Score: {avg_sc:5.1f} | Best: {self.best_score:5.0f}")

                    episode += 1
                    if episode >= n_episodes:
                        break
                        
                    rnd_speed = random.uniform(INIT_SPEED, min(INIT_SPEED + 4.0, MAX_SPEED))
                    state = env.reset(dino, start_speed=rnd_speed)
                    done = False

            if episode < n_episodes:
                self.update(state, float(done))

        # Final save
        self.save_model(save_path)
        print(f"\n[PPO] DONE - Best score: {self.best_score}")
        return scores_history

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.network.state_dict(), path)

    def load_model(self, path: str):
        if os.path.exists(path):
            self.network.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print(f"Warning: {path} not found.")

