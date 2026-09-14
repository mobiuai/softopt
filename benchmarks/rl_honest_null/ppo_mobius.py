#!/usr/bin/env python3
"""
================================================================================
PPO LunarLander -- real torch.optim.Adam vs Adam+Mobius-correction
================================================================================
IMPORTANT EXPECTATION-SETTING (read before running): earlier today,
Mobius-based corrections were ALREADY tested extensively on RL directly
(Mobius last-layer, Mobius full-network, Mobius averaged-over-8-directions
-- all on SB3's A2C/LunarLander) and found NULL every time (best case a
stable tie, no edge; one small-sample "promising" result did not
replicate at n=20). That is a DIFFERENT implementation from this exact
from-scratch PPO (matching ppo_lunarlander.py precisely), so this is
still worth running for completeness and rigor -- but based on strong
prior evidence, do not expect Mobius to fix full RL here. The reason
Mobius fixed QAOA (sign-indefinite curvature on an otherwise FIXED,
known objective) is structurally different from RL's actual problem
(a continuously-MOVING policy/value target) -- Mobius addresses the
former, not the latter. A null result here would be confirmatory, not
surprising.

Uses softopt_torch.py's forward-mode-AD machinery (torch.func.jvp) to
get D1/D2 on the exact PPO surrogate loss, then applies the SAME
Mobius-B bounded step used for QAOA, instead of SoftOpt's raw-Newton
correction (script10's approach, which was NOT separately validated as
better -- this is a genuinely new arm, not a re-run of a known-null one).

USAGE:
    pip install numpy torch gymnasium scipy
    python3 script18_ppo_lunarlander_mobius.py

TIMING: same order of magnitude as script10 (PPO/100K timesteps) --
expect roughly 3+ hours for the full 30-seed config. Reduce NUM_SEEDS
for a faster check.
================================================================================
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical
from scipy.stats import wilcoxon
import os, sys
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
sys.path.insert(0, '.')

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
ETA = 0.3  # Mobius step-size, same as QAOA


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


def mobius_B(x, y):
    denom = abs(x) + abs(y)
    if denom < 1e-12: return 0.0
    sgn = 1.0 if x >= 0 else -1.0
    return y * sgn / denom


def get_flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].reshape(p.shape))
        idx += n


def mobius_correction(model, batch, rng):
    """D1/D2 via torch.func.jvp on the exact PPO loss (same forward-mode-AD
    approach as softopt_torch.py), then a bounded Mobius-B step."""
    import torch.func as tfunc
    from torch.nn.utils.stateless import _reparametrize_module

    flat0 = get_flat_params(model)
    n = flat0.numel()
    delta_np = rng.choice([-1., 1.], size=n).astype(np.float32)
    delta = torch.tensor(delta_np)

    named = dict(model.named_parameters())
    names = list(named.keys())

    def scalar_loss(flat):
        pd = {}
        idx = 0
        for name in names:
            p = named[name]
            nn_ = p.numel()
            pd[name] = flat[idx:idx+nn_].reshape(p.shape)
            idx += nn_
        with _reparametrize_module(model, pd):
            return ppo_loss_fn(model, batch)

    with torch.no_grad():
        _, D1 = tfunc.jvp(scalar_loss, (flat0,), (delta,))
        def d_along(fp):
            _, d = tfunc.jvp(scalar_loss, (fp,), (delta,))
            return d
        _, D2 = tfunc.jvp(d_along, (flat0,), (delta,))

    B = mobius_B(float(D1), float(D2))
    new_flat = flat0 + ETA * B * delta
    set_flat_params(model, new_flat)


def train_ppo(seed, use_mobius):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make(ENV_NAME); env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]; act_dim = env.action_space.n
    model = ActorCritic(obs_dim, act_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)
    mobius_rng = np.random.default_rng(seed + 999)

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

                    loss = ppo_loss_fn(model, batch)
                    optimizer.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                    if use_mobius:
                        mobius_correction(model, batch, mobius_rng)
            buffer.clear()

    env.close()
    return np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else (
        np.mean(episode_rewards) if episode_rewards else -200)


def main():
    print("=" * 70)
    print("PPO LunarLander -- torch.optim.Adam vs Adam+Mobius-correction")
    print("=" * 70)
    print("EXPECTATION: based on today's extensive prior RL testing, Mobius")
    print("is NOT expected to fix full RL (the problem is a moving target,")
    print("not sign-indefinite curvature) -- running for completeness/rigor.")
    print(f"Timesteps: {TOTAL_TIMESTEPS:,} | Seeds: {NUM_SEEDS} | LR: {LR}")
    print("=" * 70)

    adam_results, mobius_results = [], []
    for seed in range(NUM_SEEDS):
        print(f"\n[Seed {seed+1}/{NUM_SEEDS}]")
        torch_state = torch.get_rng_state(); np_state = np.random.get_state()

        print("  Running real torch.optim.Adam...       ", end="", flush=True)
        adam_score = train_ppo(seed, use_mobius=False)
        print(f"Final Score: {adam_score:6.1f}")

        torch.set_rng_state(torch_state); np.random.set_state(np_state)

        print("  Running Adam+Mobius correction...      ", end="", flush=True)
        mobius_score = train_ppo(seed, use_mobius=True)
        print(f"Final Score: {mobius_score:6.1f}")

        adam_results.append(adam_score); mobius_results.append(mobius_score)
        diff = mobius_score - adam_score
        print(f"  -> Delta = {diff:+.1f} ({'Mobius wins' if diff>0 else 'Adam wins'})")

    adam_arr = np.array(adam_results); mobius_arr = np.array(mobius_results)
    diff = mobius_arr - adam_arr
    win_rate = np.mean(diff > 0)
    try:
        _, p_value = wilcoxon(adam_arr, mobius_arr)
    except Exception:
        p_value = 1.0
    improvement = 100 * (mobius_arr.mean() - adam_arr.mean()) / (abs(adam_arr.mean()) + 1e-9)

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"torch.optim.Adam:  {adam_arr.mean():>8.1f} +/- {adam_arr.std():.1f}")
    print(f"Adam+Mobius:       {mobius_arr.mean():>8.1f} +/- {mobius_arr.std():.1f}")
    print(f"Improvement: {improvement:+.1f}% | Win rate: {win_rate*100:.1f}% ({int(win_rate*NUM_SEEDS)}/{NUM_SEEDS}) | p={p_value:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
