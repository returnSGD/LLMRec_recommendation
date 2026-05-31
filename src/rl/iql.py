"""
Implicit Q-Learning (IQL) Trainer for offline RL.

IQL avoids querying OOD actions entirely by using expectile regression
to estimate the value function, then using advantage-weighted regression
to extract the policy. More stable than CQL on some benchmarks.

Reference: Kostrikov et al., "Offline Reinforcement Learning with
Implicit Q-Learning", ICLR 2022.

Key components:
  1. Value network V_ψ: expectile regression on Q(s,a) samples
  2. Q network Q_θ: standard TD learning, targets use V_ψ for next state
  3. Policy π_φ: advantage-weighted regression (AWR) — maximize
     log π(a|s) * exp((Q(s,a) - V(s)) / temperature)
"""
from typing import Dict, Optional
import copy
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


class ValueNetwork(nn.Module):
    """State-value network V(s). Independent of action."""

    def __init__(self, user_dim=256, memory_dim=256, hidden_dim=256,
                 num_layers=2, dropout=0.1, use_periodic=True):
        super().__init__()
        input_dim = user_dim + memory_dim * 2 + 1
        if use_periodic:
            input_dim += 10

        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, user_emb, memory_ctx, confidence, periodic_flags=None):
        if periodic_flags is not None:
            x = torch.cat([user_emb, memory_ctx, confidence, periodic_flags], dim=-1)
        else:
            x = torch.cat([user_emb, memory_ctx, confidence], dim=-1)
        return self.net(x).squeeze(-1)


class IQLTrainer:
    """
    Implicit Q-Learning for offline recommendation RL.

    Three networks: Q (value for state-action), V (value for state), π (policy).
    """

    def __init__(self,
                 q_network: POMDPPolicy,
                 v_network: ValueNetwork,
                 buffer,
                 device: str = "cuda",
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.7,
                 temperature: float = 3.0,
                 lr: float = 3e-4,
                 grad_clip: float = 1.0,
                 writer: SummaryWriter = None):
        self.q_net = q_network.to(device)        # Q(s,a)
        self.q_target = copy.deepcopy(q_network).to(device)
        self.q_target.eval()
        for p in self.q_target.parameters():
            p.requires_grad = False

        self.v_net = v_network.to(device)         # V(s)

        self.buffer = buffer
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.grad_clip = grad_clip
        self.writer = writer

        self.q_optim = optim.AdamW(self.q_net.parameters(), lr=lr, weight_decay=1e-4)
        self.v_optim = optim.AdamW(self.v_net.parameters(), lr=lr, weight_decay=1e-4)
        self.train_step = 0

    def train_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single IQL training step.

        Step 1: Update V(s) via expectile regression on Q(s,a)
        Step 2: Update Q(s,a) via TD with V(s') target
        Step 3: Update policy π via advantage-weighted regression (optional, done separately)
        """
        self.train_step += 1

        # --- Step 1: Value network update ---
        with torch.no_grad():
            q_values = self.q_net(
                batch["user_emb"], batch["memory_context"],
                batch["retrieval_confidence"], batch["periodic_flags"],
                batch["action_mask"],
            )
            q_selected = q_values.gather(1, batch["actions"].unsqueeze(-1)).squeeze(-1)

        v_pred = self.v_net(
            batch["user_emb"], batch["memory_context"],
            batch["retrieval_confidence"], batch["periodic_flags"],
        )

        # Asymmetric loss: expectile regression
        # L_τ(x) = |τ - I(x < 0)| · x²
        diff = q_selected - v_pred
        weight = torch.abs(self.expectile - (diff < 0).float())
        v_loss = (weight * diff ** 2).mean()

        self.v_optim.zero_grad()
        v_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), self.grad_clip)
        self.v_optim.step()

        # --- Step 2: Q network update ---
        # Q(s,a) ← r + γ·V(s')
        with torch.no_grad():
            next_v = self.v_net(
                batch["next_user_emb"], batch["next_memory_context"],
                batch["next_retrieval_confidence"], batch["next_periodic_flags"],
            )
            target = batch["rewards"] + self.gamma * (1 - batch["dones"]) * next_v

        q_pred = self.q_net(
            batch["user_emb"], batch["memory_context"],
            batch["retrieval_confidence"], batch["periodic_flags"],
            batch["action_mask"],
        )
        q_chosen = q_pred.gather(1, batch["actions"].unsqueeze(-1)).squeeze(-1)

        q_loss = F.mse_loss(q_chosen, target)

        self.q_optim.zero_grad()
        q_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.q_optim.step()

        # --- Soft update target ---
        with torch.no_grad():
            for tp, p in zip(self.q_target.parameters(), self.q_net.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        metrics = {
            "loss/v": v_loss.item(),
            "loss/q": q_loss.item(),
            "q/mean": q_chosen.mean().item(),
            "v/mean": v_pred.mean().item(),
        }

        if self.writer and self.train_step % 100 == 0:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, self.train_step)

        return metrics

    def extract_policy_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Advantage-Weighted Regression loss for the policy (Step 3).

        L_AWR = -E_{s,a~D}[exp((Q(s,a) - V(s)) / τ) · log π(a|s)]

        This is called separately from train_batch when we want to
        also update the policy. The policy network IS the q_network's
        action selection head — we reuse it.

        Returns:
            awr_loss: scalar loss for policy update
        """
        with torch.no_grad():
            q_values = self.q_net(
                batch["user_emb"], batch["memory_context"],
                batch["retrieval_confidence"], batch["periodic_flags"],
                batch["action_mask"],
            )
            v_values = self.v_net(
                batch["user_emb"], batch["memory_context"],
                batch["retrieval_confidence"], batch["periodic_flags"],
            )
            advantages = (q_values - v_values.unsqueeze(-1)) / self.temperature
            weights = torch.exp(torch.clamp(advantages, -10, 5))
            weights = weights.gather(1, batch["actions"].unsqueeze(-1)).squeeze(-1)

        # Policy logits
        q_pred = self.q_net(
            batch["user_emb"], batch["memory_context"],
            batch["retrieval_confidence"], batch["periodic_flags"],
            batch["action_mask"],
        )

        # AWR: maximize weighted log-likelihood of dataset actions
        log_probs = F.log_softmax(q_pred, dim=-1)
        log_probs_chosen = log_probs.gather(1, batch["actions"].unsqueeze(-1)).squeeze(-1)
        awr_loss = -(weights * log_probs_chosen).mean()

        return awr_loss

    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_target.state_dict(),
            "v_net": self.v_net.state_dict(),
            "q_optim": self.q_optim.state_dict(),
            "v_optim": self.v_optim.state_dict(),
            "train_step": self.train_step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.v_net.load_state_dict(ckpt["v_net"])
        self.q_optim.load_state_dict(ckpt["q_optim"])
        self.v_optim.load_state_dict(ckpt["v_optim"])
        self.train_step = ckpt["train_step"]
