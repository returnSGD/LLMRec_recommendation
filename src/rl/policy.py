"""
POMDP Policy Network for RL-based recommendation.

Architecture:
  State (user_emb + memory_context + confidence) → MLP → Q-values (|A|)
                                                         → ε-greedy action

With uncertainty-adaptive epsilon:
  ε_effective = ε_base + (1 - ε_base) * memory_uncertainty * α_unc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class POMDPPolicy(nn.Module):
    """
    Q-network that takes POMDP state and outputs Q-values for each macro-action.

    State components (concatenated):
      - user_embedding: (user_dim,) user interest representation
      - memory_context: (2 * memory_dim,) short-term + long-term aggregated
      - retrieval_confidence: scalar
      - periodic_flags: (n_periodic,) binary flags for due items
    """

    def __init__(self,
                 user_dim: int = 256,
                 memory_dim: int = 256,
                 num_actions: int = 16,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 use_periodic_context: bool = True):
        super().__init__()
        self.user_dim = user_dim

        # Input: user_emb + short_mem + long_mem + confidence + periodic
        input_dim = user_dim + memory_dim * 2 + 1
        if use_periodic_context:
            input_dim += 10  # top-10 periodic item signals

        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.feature_net = nn.Sequential(*layers)
        self.q_head = nn.Linear(hidden_dim, num_actions)

        # Value head for dueling architecture
        self.v_head = nn.Linear(hidden_dim, 1)

        self.num_actions = num_actions

    def forward(self, user_emb: torch.Tensor,
                memory_context: torch.Tensor,
                retrieval_confidence: torch.Tensor,
                periodic_flags: torch.Tensor = None,
                action_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            user_emb: (B, user_dim)
            memory_context: (B, 2 * memory_dim) [short_term; long_term]
            retrieval_confidence: (B, 1)
            periodic_flags: (B, 10) optional
            action_mask: (B, num_actions) 1=valid, 0=invalid

        Returns:
            q_values: (B, num_actions)
        """
        if periodic_flags is not None:
            state = torch.cat([user_emb, memory_context,
                               retrieval_confidence, periodic_flags], dim=-1)
        else:
            state = torch.cat([user_emb, memory_context, retrieval_confidence], dim=-1)

        features = self.feature_net(state)

        # Dueling: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        value = self.v_head(features)
        advantage = self.q_head(features)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)

        # Apply action mask (set invalid actions to -inf)
        if action_mask is not None:
            q_values = q_values.masked_fill(action_mask == 0, float("-inf"))

        return q_values

    def get_action(self, user_emb: torch.Tensor,
                   memory_context: torch.Tensor,
                   retrieval_confidence: torch.Tensor,
                   periodic_flags: torch.Tensor = None,
                   action_mask: torch.Tensor = None,
                   epsilon: float = 0.0) -> torch.Tensor:
        """
        Epsilon-greedy action selection.

        Returns:
            actions: (B,) selected action indices
        """
        q_values = self.forward(user_emb, memory_context,
                                retrieval_confidence, periodic_flags,
                                action_mask)

        B = q_values.shape[0]
        actions = torch.zeros(B, dtype=torch.long, device=q_values.device)

        # Greedy selection
        greedy = q_values.argmax(dim=-1)

        # Random exploration (respecting mask)
        if epsilon > 0:
            if action_mask is not None:
                # Sample uniformly among valid actions
                valid_counts = action_mask.sum(dim=-1)
                random_choice = torch.distributions.Categorical(
                    probs=action_mask / valid_counts.unsqueeze(-1).clamp(min=1)
                ).sample()
            else:
                random_choice = torch.randint(0, self.num_actions,
                                              (B,), device=q_values.device)

            explore_mask = torch.rand(B, device=q_values.device) < epsilon
            actions = torch.where(explore_mask, random_choice, greedy)
        else:
            actions = greedy

        return actions


class UncertaintyAdaptiveEpsilon:
    """
    Adaptive epsilon based on memory retrieval confidence.

    ε_effective = ε_base + (ε_max - ε_base) * (1 - confidence)
    = ε_base + (ε_max - ε_base) * uncertainty

    High uncertainty → high epsilon → more exploration.
    Low uncertainty → low epsilon → more exploitation.
    """

    def __init__(self, epsilon_base: float = 0.05,
                 epsilon_max: float = 0.5,
                 uncertainty_scale: float = 1.0):
        self.epsilon_base = epsilon_base
        self.epsilon_max = epsilon_max
        self.uncertainty_scale = uncertainty_scale

    def get_epsilon(self, retrieval_confidence: float,
                    global_step: int = None,
                    decay_steps: int = 10_000) -> float:
        """
        Compute effective epsilon from memory confidence.

        Args:
            retrieval_confidence: scalar [0,1] from memory retrieval
            global_step: for global epsilon annealing
            decay_steps: total steps for global epsilon to reach base
        """
        # Global annealing
        if global_step is not None and decay_steps > 0:
            progress = min(1.0, global_step / decay_steps)
            base = self.epsilon_base + (0.3 - self.epsilon_base) * (1 - progress)
        else:
            base = self.epsilon_base

        # Uncertainty adjustment
        uncertainty = 1.0 - retrieval_confidence
        adaptive = base + (self.epsilon_max - base) * uncertainty * self.uncertainty_scale

        return np.clip(adaptive, 0.01, 0.9)
