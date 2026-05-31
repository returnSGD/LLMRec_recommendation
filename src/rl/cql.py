"""
Conservative Q-Learning (CQL) Trainer for offline RL.

CQL adds a conservative penalty that pushes down Q-values for
out-of-distribution (OOD) actions, preventing overestimation on
unseen state-action pairs. Critical for recommendation because
the logging policy (P5) only covers a narrow action distribution.

Reference: Kumar et al., "Conservative Q-Learning for Offline RL", NeurIPS 2020.

CQL loss:
  L = α * (E_{s,a~D}[log Σ exp(Q(s,a'))] - E_{s,a~D}[Q(s,a)])  ← conservative penalty
    + E_{s,a,r,s'~D}[(Q(s,a) - (r + γ E_{a'~π}[Q_target(s',a')]))²]  ← TD error
"""
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .policy import POMDPPolicy
from .buffer import ReplayBuffer


class CQLTrainer:
    """
    Conservative Q-Learning for offline recommendation RL.

    Trains two Q-networks (double Q-learning) + target networks.
    """

    def __init__(self,
                 policy: POMDPPolicy,
                 target_policy: POMDPPolicy,
                 buffer: ReplayBuffer,
                 device: str = "cuda",
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 cql_alpha: float = 1.0,
                 cql_num_samples: int = 10,
                 lr: float = 3e-4,
                 grad_clip: float = 1.0,
                 writer: SummaryWriter = None):
        self.policy = policy.to(device)
        self.target_policy = target_policy.to(device)
        self.target_policy.load_state_dict(policy.state_dict())
        self.target_policy.eval()

        self.buffer = buffer
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.cql_alpha = cql_alpha
        self.cql_num_samples = cql_num_samples
        self.grad_clip = grad_clip
        self.writer = writer

        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.train_step = 0

    def train_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single CQL training step on a sampled batch.

        Returns dict of loss components for logging.
        """
        self.train_step += 1
        B = batch["actions"].shape[0]

        # --- Current Q-values ---
        q_values = self.policy(
            batch["user_emb"],
            batch["memory_context"],
            batch["retrieval_confidence"],
            batch["periodic_flags"],
            batch["action_mask"],
        )  # (B, |A|)

        q_chosen = q_values.gather(1, batch["actions"].unsqueeze(-1)).squeeze(-1)  # (B,)

        # --- Target Q-values (double Q) ---
        with torch.no_grad():
            # Next Q from target network, action selection from current policy
            next_q_current = self.policy(
                batch["next_user_emb"],
                batch["next_memory_context"],
                batch["next_retrieval_confidence"],
                batch["next_periodic_flags"],
                batch["next_action_mask"],
            )
            next_actions = next_q_current.argmax(dim=-1, keepdim=True)

            next_q_target = self.target_policy(
                batch["next_user_emb"],
                batch["next_memory_context"],
                batch["next_retrieval_confidence"],
                batch["next_periodic_flags"],
                batch["next_action_mask"],
            )
            next_q_chosen = next_q_target.gather(1, next_actions).squeeze(-1)

            target = batch["rewards"] + self.gamma * (1 - batch["dones"]) * next_q_chosen

        # --- TD Error Loss ---
        td_loss = F.mse_loss(q_chosen, target)

        # --- CQL Conservative Penalty ---
        # Penalty = E_{s~D}[log Σ_a exp(Q(s,a)) - Q(s,a*)]
        # Pushes down Q on ALL actions, pulls up Q on dataset actions

        # Log-sum-exp over all valid actions
        logsumexp = torch.logsumexp(q_values, dim=-1)  # (B,)
        cql_penalty = (logsumexp - q_chosen).mean()

        # --- Total Loss ---
        total_loss = td_loss + self.cql_alpha * cql_penalty

        # --- Optimization ---
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.optimizer.step()

        # --- Soft update target ---
        self._soft_update()

        # Logging
        metrics = {
            "loss/total": total_loss.item(),
            "loss/td": td_loss.item(),
            "loss/cql_penalty": cql_penalty.item(),
            "q/mean": q_values.mean().item(),
            "q/chosen_mean": q_chosen.mean().item(),
            "q/max": q_values.max().item(),
        }

        if self.writer and self.train_step % 100 == 0:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, self.train_step)

        return metrics

    def _soft_update(self):
        """Polyak averaging: θ_target = τ·θ + (1-τ)·θ_target"""
        with torch.no_grad():
            for target_param, param in zip(self.target_policy.parameters(),
                                           self.policy.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def train(self, num_steps: int, batch_size: int = 256,
              eval_fn=None, eval_every: int = 2000):
        """Full training loop."""
        import time
        from tqdm import tqdm

        pbar = tqdm(range(num_steps), desc="CQL Training")
        metrics_history = []

        for step in pbar:
            if len(self.buffer) < batch_size:
                continue

            batch = self.buffer.sample(batch_size, self.device)
            metrics = self.train_batch(batch)
            metrics_history.append(metrics)

            if step % 100 == 0:
                recent = {k: np.mean([m[k] for m in metrics_history[-100:]])
                          for k in metrics_history[-1]}
                pbar.set_postfix(recent)

            if eval_every > 0 and step > 0 and step % eval_every == 0 and eval_fn:
                eval_results = eval_fn(self.policy, step)
                if self.writer:
                    for k, v in eval_results.items():
                        self.writer.add_scalar(f"eval/{k}", v, step)

        return metrics_history

    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "target_policy": self.target_policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self.train_step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.target_policy.load_state_dict(ckpt["target_policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.train_step = ckpt["train_step"]
