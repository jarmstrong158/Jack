"""
PPO implementation with LSTM trunk for Dolly warehouse RL.

Architecture:
  state(155) → Linear → ReLU → LSTM(256) → 7 policy heads + 1 value head

The LSTM maintains a hidden state across every 15-minute step of the simulation,
giving Dolly temporal memory across the full work day and work week. This lets it
learn patterns like hustle pacing across days and management backlog build-up.

Key design:
  - Hidden state persists across ALL steps within a year (never reset mid-year)
  - Reset to zeros only at the start of each new year episode
  - Updates use full sequential evaluation (TBPTT with chunk-based gradient detach)
  - No random minibatching — sequential order must be preserved for LSTM correctness
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

from volt_sim.config import PPO as PPO_CFG, TOTAL_STATE_SIZE
from volt_sim.agent.actions import ACTION_HEAD_SIZE, NUM_ACTION_HEADS


class ActorCritic(nn.Module):
    def __init__(self,
                 state_size: int = TOTAL_STATE_SIZE,
                 head_size: int = ACTION_HEAD_SIZE,
                 num_heads: int = NUM_ACTION_HEADS,
                 hidden_size: int = PPO_CFG["hidden_size"]):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.hidden_size = hidden_size

        # Input projection: raw state → LSTM input space
        self.input_proj = nn.Linear(state_size, hidden_size)

        # LSTM: single layer, carries memory across all steps in the year
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)

        # Policy heads — one per worker, each outputs task+hustle logits
        self.policy_heads = nn.ModuleList([
            nn.Linear(hidden_size, head_size)
            for _ in range(num_heads)
        ])

        # Value head
        self.value_head = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.input_proj.weight, gain=np.sqrt(2))
        nn.init.constant_(self.input_proj.bias, 0.0)
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        for head in self.policy_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.constant_(head.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def init_hidden(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zeroed hidden state (h, c) for a fresh episode."""
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

    def step(self,
             state: torch.Tensor,
             hidden: tuple[torch.Tensor, torch.Tensor],
             action_mask: torch.Tensor = None
             ) -> tuple[list[torch.Tensor], torch.Tensor, tuple]:
        """
        Single-step forward pass.
        state:       (1, state_size)
        hidden:      (h, c), each (1, 1, hidden_size)
        action_mask: (1, num_heads, head_size) bool, optional
        Returns: logits_list, value (scalar tensor), new_hidden
        """
        x = F.relu(self.input_proj(state))       # (1, hidden_size)
        x = x.unsqueeze(1)                       # (1, 1, hidden_size)
        lstm_out, new_hidden = self.lstm(x, hidden)
        features = lstm_out.squeeze(1)           # (1, hidden_size)

        logits_list = []
        for i, head in enumerate(self.policy_heads):
            logits = head(features)              # (1, head_size)
            if action_mask is not None:
                mask = action_mask[:, i, :]      # (1, head_size)
                logits = logits.masked_fill(~mask, float("-inf"))
            logits_list.append(logits)

        value = self.value_head(features).squeeze(-1)
        return logits_list, value, new_hidden

    def evaluate_sequence(self,
                          states: list,
                          actions: list,
                          action_masks: list,
                          entry_hidden: tuple[torch.Tensor, torch.Tensor],
                          tbptt_chunk: int
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized sequence replay for PPO update.

        The LSTM and all linear layers run on entire chunks at once rather than
        step-by-step, giving ~10x speedup over the naive loop. Only mask
        application and Categorical distributions remain per-step (cheap).

        TBPTT: hidden state is detached between chunks to bound gradient length.
        Returns: (log_probs, values, entropies) — each (N,) tensor
        """
        n = len(states)
        # Stack all states and actions upfront — one allocation, no per-step copies
        states_t = torch.FloatTensor(np.array(states, dtype=np.float32))   # (N, state_size)
        actions_t = torch.LongTensor(np.array(actions))                    # (N, num_heads)

        hidden = (entry_hidden[0].clone(), entry_hidden[1].clone())
        all_log_probs = []
        all_values = []
        all_entropies = []

        for chunk_start in range(0, n, tbptt_chunk):
            chunk_end = min(chunk_start + tbptt_chunk, n)
            chunk_len = chunk_end - chunk_start

            chunk_states  = states_t[chunk_start:chunk_end]    # (chunk_len, state_size)
            chunk_actions = actions_t[chunk_start:chunk_end]   # (chunk_len, num_heads)

            # ── Single vectorized forward pass through input_proj + LSTM ──
            x = F.relu(self.input_proj(chunk_states))          # (chunk_len, hidden_size)
            lstm_out, hidden = self.lstm(x.unsqueeze(0), hidden)  # (1, chunk_len, hidden_size)
            features = lstm_out.squeeze(0)                     # (chunk_len, hidden_size)

            # Detach hidden before next chunk (TBPTT boundary)
            hidden = (hidden[0].detach(), hidden[1].detach())

            # ── Vectorized value head — entire chunk at once ──
            values_chunk = self.value_head(features).squeeze(-1)  # (chunk_len,)

            # ── Fully vectorized policy heads + masking + log_probs ──
            # Pre-stack masks for the whole chunk: (chunk_len, num_heads, head_size)
            chunk_masks_np = np.stack([
                action_masks[chunk_start + t]
                if action_masks[chunk_start + t] is not None
                else np.ones((self.num_heads, self.head_size), dtype=bool)
                for t in range(chunk_len)
            ])
            chunk_masks = torch.BoolTensor(chunk_masks_np)  # (chunk_len, num_heads, head_size)

            per_head_lp  = []
            per_head_ent = []

            for i, head in enumerate(self.policy_heads):
                logits = head(features)                                 # (chunk_len, head_size)
                # Use -1e9 instead of -inf: effectively zeros probability but stays
                # finite and differentiable — -inf causes NaN in entropy backward.
                logits = logits.masked_fill(~chunk_masks[:, i, :], -1e9)
                log_p  = F.log_softmax(logits, dim=-1)                 # (chunk_len, head_size)
                probs  = torch.softmax(logits, dim=-1)
                # Gather the log_prob of the action actually taken
                acts_i = chunk_actions[:, i].unsqueeze(1)              # (chunk_len, 1)
                per_head_lp.append(log_p.gather(1, acts_i).squeeze(1))         # (chunk_len,)
                per_head_ent.append(-(probs * log_p).sum(dim=-1))

            # Stack and sum across workers — no in-place ops, gradient flows cleanly
            all_log_probs.append(torch.stack(per_head_lp).sum(dim=0))   # (chunk_len,)
            all_entropies.append(torch.stack(per_head_ent).sum(dim=0))  # (chunk_len,)
            all_values.extend(values_chunk.unbind(0))

        return (torch.cat(all_log_probs),
                torch.stack(all_values),
                torch.cat(all_entropies))

    def get_action(self,
                   state: np.ndarray,
                   hidden: tuple[torch.Tensor, torch.Tensor],
                   action_mask: list[list[bool]] = None
                   ) -> tuple[list[int], torch.Tensor, torch.Tensor, tuple]:
        """
        Sample actions for a single timestep during environment interaction.
        Returns: actions, log_prob, value, new_hidden
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        mask_t = (torch.BoolTensor(action_mask).unsqueeze(0)
                  if action_mask is not None else None)

        with torch.no_grad():
            logits_list, value, new_hidden = self.step(state_t, hidden, mask_t)

        actions = []
        log_probs = []
        for logits in logits_list:
            dist = Categorical(logits=logits.squeeze(0))
            action = dist.sample()
            actions.append(action.item())
            log_probs.append(dist.log_prob(action))

        log_prob = torch.stack(log_probs).sum()
        return actions, log_prob, value.squeeze(0), new_hidden


class RolloutBuffer:
    """Stores one rollout's worth of transitions for a sequential PPO update."""

    def __init__(self):
        self.states: list = []
        self.actions: list = []
        self.log_probs: list = []
        self.rewards: list = []
        self.values: list = []
        self.dones: list = []
        self.action_masks: list = []

        # LSTM hidden state at the START of this rollout.
        # Stored so evaluate_sequence can replay from the exact same starting point.
        self.entry_hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def set_entry_hidden(self, hidden: tuple[torch.Tensor, torch.Tensor]):
        """Snapshot hidden state before collection begins. Call before each rollout."""
        self.entry_hidden = (hidden[0].detach().clone(),
                             hidden[1].detach().clone())

    def add(self, state, action, log_prob, reward, value, done, action_mask=None):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def compute_returns(self, gamma: float,
                        gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
        """GAE advantage and return computation."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            next_value = 0.0 if t == n - 1 else self.values[t + 1]
            non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + gamma * next_value * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.action_masks.clear()
        # entry_hidden is intentionally NOT cleared — it's overwritten by
        # set_entry_hidden() before each new collection anyway.


class PPOAgent:
    def __init__(self, state_size: int = TOTAL_STATE_SIZE):
        self.model = ActorCritic(state_size=state_size)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=PPO_CFG["lr"]
        )
        self.buffer = RolloutBuffer()

        self.clip_epsilon = PPO_CFG["clip_epsilon"]
        self.entropy_coeff = PPO_CFG["entropy_coeff"]
        self.value_loss_coeff = PPO_CFG["value_loss_coeff"]
        self.max_grad_norm = PPO_CFG["max_grad_norm"]
        self.epochs = PPO_CFG["epochs_per_update"]
        self.gamma = PPO_CFG["gamma"]
        self.gae_lambda = PPO_CFG["gae_lambda"]
        self.tbptt_chunk_size = PPO_CFG["tbptt_chunk_size"]

        # Live LSTM hidden state — persists across all steps within a year.
        # Reset to zeros at the start of each new year via reset_hidden().
        self.hidden: tuple[torch.Tensor, torch.Tensor] = self.model.init_hidden()

    def reset_hidden(self):
        """Reset LSTM memory. Call at the start of each new year episode."""
        self.hidden = self.model.init_hidden()

    def select_action(self,
                      state: np.ndarray,
                      action_mask: list[list[bool]] = None
                      ) -> tuple[list[int], torch.Tensor, float]:
        """
        Sample actions and advance LSTM hidden state by one step.
        Hidden state is updated in-place so it persists into the next call.
        """
        actions, log_prob, value, new_hidden = self.model.get_action(
            state, self.hidden, action_mask
        )
        self.hidden = new_hidden
        return actions, log_prob, value.item()

    def store_transition(self, state, action, log_prob, reward, value, done,
                         action_mask=None):
        mask_array = np.array(action_mask) if action_mask is not None else None
        self.buffer.add(state, action, log_prob, reward, value, done, mask_array)

    def update(self) -> dict:
        """
        PPO update over the current rollout buffer.
        Processes steps in sequential order to maintain LSTM temporal correctness.
        """
        if self.buffer.entry_hidden is None:
            self.buffer.entry_hidden = self.model.init_hidden()

        advantages, returns = self.buffer.compute_returns(self.gamma, self.gae_lambda)

        # Normalize advantages
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - advantages.mean()) / adv_std

        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)
        old_log_probs_t = torch.FloatTensor([lp.item() for lp in self.buffer.log_probs])

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_updates": 0}

        for _ in range(self.epochs):
            # Sequential replay through LSTM — must not shuffle
            log_probs, values, entropy = self.model.evaluate_sequence(
                self.buffer.states,
                self.buffer.actions,
                self.buffer.action_masks,
                self.buffer.entry_hidden,
                self.tbptt_chunk_size,
            )

            ratio = torch.exp(log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                 1.0 + self.clip_epsilon) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Normalize returns before MSE — keeps value_loss ~O(1) regardless of
            # reward scale, preventing large gradients from destabilizing the LSTM.
            ret_mean = returns_t.mean()
            ret_std  = returns_t.std() + 1e-8
            value_loss = F.mse_loss((values - ret_mean) / ret_std,
                                    (returns_t - ret_mean) / ret_std)
            entropy_loss = -entropy.mean()

            loss = (policy_loss
                    + self.value_loss_coeff * value_loss
                    + self.entropy_coeff * entropy_loss)

            self.optimizer.zero_grad()
            loss.backward()
            # NaN guard: if exploding gradients produce NaN, skip this step cleanly
            total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                self.optimizer.zero_grad()
                continue
            self.optimizer.step()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += -entropy_loss.item()
            metrics["n_updates"] += 1

        for k in ("policy_loss", "value_loss", "entropy"):
            metrics[k] /= max(1, metrics["n_updates"])

        self.buffer.clear()
        return metrics

    # Bumped whenever the architecture changes in a backward-incompatible way.
    # load() rejects files with a different version instead of attempting a broken load.
    ARCH_VERSION = "lstm-v1"

    def save(self, path: str, state_stats=None):
        data = {
            "arch_version": self.ARCH_VERSION,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if state_stats is not None:
            data["state_stats"] = {
                "n": state_stats.n,
                "mean": state_stats.mean,
                "var": state_stats.var,
                "_m2": state_stats._m2,
            }
        torch.save(data, path)

    def load(self, path: str, state_stats=None) -> bool:
        checkpoint = torch.load(path, weights_only=False)

        # Architecture version check — catches stale pre-LSTM checkpoints cleanly
        saved_version = checkpoint.get("arch_version", "unknown")
        if saved_version != self.ARCH_VERSION:
            print(f"Checkpoint architecture mismatch: saved={saved_version!r}, current={self.ARCH_VERSION!r}")
            return False

        try:
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except (RuntimeError, ValueError) as e:
            print(f"Checkpoint incompatible (weights shape mismatch): {e}")
            return False
        if state_stats is not None and "state_stats" in checkpoint:
            ss = checkpoint["state_stats"]
            state_stats.n = ss["n"]
            state_stats.mean = ss["mean"]
            state_stats.var = ss["var"]
            state_stats._m2 = ss.get("_m2", ss["var"] * max(1, ss["n"] - 1))
        return True
