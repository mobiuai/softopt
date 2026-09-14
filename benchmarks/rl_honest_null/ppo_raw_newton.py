#!/usr/bin/env python3
"""
================================================================================
PPO LunarLander: real torch.optim.Adam vs SoftOpt (standalone)
================================================================================
Same structure and fairness protocol as the original ppo_lunarlander.py
customer script -- identical PPO algorithm, identical hyperparameters,
identical RNG-state-matching between the two arms. The only change is
the optimizer.

IMPORTANT CONTEXT (from today's extensive RL investigation): full PPO
has a continuously-evolving policy, which is EXACTLY the property that
eleven distinct SoftJet-Newton/SoftOpt mechanisms were shown NOT to help
with earlier today (the moving-target problem -- see project notes).
Expect this to likely replicate that null pattern rather than show an
advantage; running it is the honest way to confirm rather than assume.
A quick smoke test (4096 timesteps, 1 seed) already showed this pattern:
Adam -148.7 vs SoftOpt -562.0 -- Adam clearly ahead, consistent with
today's null findings, not a bug.

TIMING: measured directly at 4096 timesteps/1 seed/both arms: ~17s.
The full customer-matching config (100,000 timesteps x 30 seeds) scales
to roughly 3-3.5 HOURS. Consider reducing NUM_SEEDS (e.g. to 5) for a
faster, still-informative check before committing to the full run.
================================================================================
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical
from scipy.stats import wilcoxon
import os, json, sys
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
sys.path.insert(0, '.')
from softopt_torch import SoftOpt

ENV_NAME = "LunarLander-v3"
TOTAL_TIMESTEPS = 100_000
STEPS_PER_UPDATE = 2048
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
LR = 3e-4
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
NUM_SEEDS = 30


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)

    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs); self.actions.append(action); self.log_probs.append(log_prob)
        self.rewards.append(reward); self.dones.append(done); self.values.append(value)

    def clear(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        advantages, returns = [], []
        gae = 0
        values = self.values + [last_value]
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                delta = self.rewards[t] - values[t]; gae = delta
            else:
                delta = self.rewards[t] + gamma*values[t+1] - values[t]
                gae = delta + gamma*gae_lambda*gae
            advantages.insert(0, gae); returns.insert(0, gae + values[t])
        return advantages, returns

    def get_tensors(self):
        return (torch.stack(self.obs), torch.stack(self.actions),
                torch.stack(self.log_probs), torch.tensor(self.values, dtype=torch.float32))


def ppo_loss_fn(model, batch):
    """The exact PPO clipped surrogate loss -- used both for the real-
    Adam baseline (via .backward()) and as SoftOpt's known objective
    (via forward-mode AD on this same function)."""
    mb_obs, mb_actions, mb_log_probs, mb_advantages, mb_returns = batch
    _, new_log_probs, entropy, new_values = model.get_action_and_value(mb_obs, mb_actions)
    log_ratio = new_log_probs - mb_log_probs
    ratio = log_ratio.exp()
    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
    v_loss = F.mse_loss(new_values, mb_returns)
    entropy_loss = -entropy.mean()
    return pg_loss + VF_COEF*v_loss + ENT_COEF*entropy_loss


def train_ppo(seed, optimizer_name="adam"):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make(ENV_NAME); env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]; act_dim = env.action_space.n
    model = ActorCritic(obs_dim, act_dim)

    use_softopt = (optimizer_name == "softopt")
    if not use_softopt:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    else:
        soft_opt = SoftOpt(model.parameters(), model, ppo_loss_fn, lr=LR, eps=1e-5, seed=seed+999)

    buffer = RolloutBuffer()
    obs, _ = env.reset(); obs = torch.tensor(obs, dtype=torch.float32)
    episode_rewards = []; current_episode_reward = 0

    for step in range(TOTAL_TIMESTEPS):
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        current_episode_reward += reward
        buffer.add(obs, action, log_prob, reward, done, value.item())
        obs = torch.tensor(next_obs, dtype=torch.float32)
        if done:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0
            obs, _ = env.reset(); obs = torch.tensor(obs, dtype=torch.float32)

        if len(buffer.obs) >= STEPS_PER_UPDATE:
            with torch.no_grad():
                _, _, _, last_value = model.get_action_and_value(obs)
            advantages, returns = buffer.compute_returns_and_advantages(last_value.item(), GAMMA, GAE_LAMBDA)
            b_obs, b_actions, b_log_probs, b_values = buffer.get_tensors()
            b_advantages = torch.tensor(advantages, dtype=torch.float32)
            b_returns = torch.tensor(returns, dtype=torch.float32)
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            indices = np.arange(len(buffer.obs))
            for epoch in range(N_EPOCHS):
                np.random.shuffle(indices)
                for start in range(0, len(indices), BATCH_SIZE):
                    end = start + BATCH_SIZE
                    bi = indices[start:end]
                    batch = (b_obs[bi], b_actions[bi], b_log_probs[bi], b_advantages[bi], b_returns[bi])

                    if not use_softopt:
                        loss = ppo_loss_fn(model, batch)
                        optimizer.zero_grad(); loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                        optimizer.step()
                    else:
                        soft_opt.step(batch)
            buffer.clear()

    env.close()
    return np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else (
        np.mean(episode_rewards) if episode_rewards else -200)


def main():
    print("=" * 70)
    print("PPO LunarLander: real torch.optim.Adam vs SoftOpt (standalone)")
    print("=" * 70)
    print(f"Timesteps: {TOTAL_TIMESTEPS:,} | Seeds: {NUM_SEEDS} | LR: {LR}")
    print("=" * 70)

    adam_results, soft_results = [], []
    for seed in range(NUM_SEEDS):
        print(f"\n[Seed {seed+1}/{NUM_SEEDS}]")
        torch_state = torch.get_rng_state(); np_state = np.random.get_state()

        print("  Running real torch.optim.Adam... ", end="", flush=True)
        adam_score = train_ppo(seed, "adam")
        print(f"Final Score: {adam_score:6.1f}")

        torch.set_rng_state(torch_state); np.random.set_state(np_state)

        print("  Running SoftOpt...                ", end="", flush=True)
        soft_score = train_ppo(seed, "softopt")
        print(f"Final Score: {soft_score:6.1f}")

        adam_results.append(adam_score); soft_results.append(soft_score)
        diff = soft_score - adam_score
        print(f"  -> Delta = {diff:+.1f} ({'SoftOpt wins' if diff>0 else 'Adam wins'})")

    adam_arr = np.array(adam_results); soft_arr = np.array(soft_results)
    diff = soft_arr - adam_arr
    win_rate = np.mean(diff > 0)
    try:
        _, p_value = wilcoxon(adam_arr, soft_arr)
    except Exception:
        p_value = 1.0
    improvement = 100 * (soft_arr.mean() - adam_arr.mean()) / (abs(adam_arr.mean()) + 1e-9)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"torch.optim.Adam: {adam_arr.mean():>8.1f} +/- {adam_arr.std():.1f}")
    print(f"SoftOpt:          {soft_arr.mean():>8.1f} +/- {soft_arr.std():.1f}")
    print(f"Improvement: {improvement:+.1f}% | Win rate: {win_rate*100:.1f}% ({int(win_rate*NUM_SEEDS)}/{NUM_SEEDS}) | p={p_value:.4f}")
    print("=" * 70)

    fname = f'ppo_lunarlander_softopt_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(fname, 'w') as f:
        json.dump({'adam_mean': float(adam_arr.mean()), 'softopt_mean': float(soft_arr.mean()),
                    'improvement_pct': float(improvement), 'win_rate': float(win_rate),
                    'p_value': float(p_value)}, f, indent=2)

if __name__ == "__main__":
    main()
