"""
Full RL + Memory recommendation model.

Integrates three components:
  1. P5/T5 backbone (shared encoder for user/item understanding)
  2. Hierarchical Memory System (short-term + long-term FAISS store)
  3. POMDP RL Policy (discrete action decision)

Architecture flow:
  User Query → P5 Encoder → user_emb
  user_emb → Memory Manager → (short_context, long_context, confidence)
  [user_emb; memory_context; confidence] → RL Policy → discrete action
  action → candidate retrieval strategy → recommended items
  (optional) → P5 Decoder → natural language explanation
"""
from typing import Dict, List, Optional, Tuple
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .memory import MemoryEncoder, MemoryManager
from .rl import POMDPEnv, POMDPPolicy, RewardFunction, UncertaintyAdaptiveEpsilon
from .rl.actions import ACTION_REGISTRY, NUM_ACTIONS, action_to_candidate_strategy


class MemoryRLModel(nn.Module):
    """
    End-to-end recommendation model with external memory and RL decision layer.

    Can be trained in two phases:
      Phase 1: Behavior Cloning (warm-start policy from P5's recommendations)
      Phase 2: Offline RL (CQL/IQL) on collected interaction data
    """

    def __init__(self, config: Config, p5_model=None):
        super().__init__()
        self.config = config

        # P5 backbone (frozen or fine-tunable)
        self.p5 = p5_model
        if p5_model is not None:
            self.user_dim = p5_model.config.d_model  # 512 for t5-small
        else:
            self.user_dim = config.memory.memory_dim

        # Memory system
        self.memory_encoder = MemoryEncoder(
            num_users=50_000,  # will be expanded for real data
            num_items=50_000,
            embed_dim=config.memory.memory_dim,
            memory_dim=config.memory.memory_dim,
        )
        self.memory_manager = MemoryManager(
            memory_dim=config.memory.memory_dim,
            short_term_capacity=config.memory.short_term_capacity,
            top_k=config.memory.top_k_retrieval,
            time_decay_lambda=config.memory.time_decay_lambda,
        )
        self.memory_manager.set_encoder(self.memory_encoder)

        # RL policy
        self.policy = POMDPPolicy(
            user_dim=self.user_dim,
            memory_dim=config.memory.memory_dim,
            num_actions=NUM_ACTIONS,
            hidden_dim=config.rl.hidden_dim,
            num_layers=config.rl.num_layers,
            dropout=config.rl.dropout,
        )

        # Exploration
        self.epsilon_scheduler = UncertaintyAdaptiveEpsilon(
            epsilon_base=config.rl.epsilon_end,
            epsilon_max=config.rl.epsilon_start,
        )

        # Reward
        self.reward_fn = RewardFunction()

        # User embedding cache (for efficiency)
        self.register_buffer("_dummy", torch.zeros(1))

    def encode_user(self, user_interaction_ids: List[int],
                    user_id: int = None) -> np.ndarray:
        """
        Encode a user's interaction history into an embedding vector.
        Uses P5 encoder if available, otherwise simple average of item embeddings.
        """
        if self.p5 is not None:
            # Use P5 encoder to get user representation
            # Simplified: average over the user's sequential prompts
            with torch.no_grad():
                # Encode via P5's encoder (requires proper tokenization)
                # For MVP, return a placeholder using item ID embeddings
                if len(user_interaction_ids) == 0:
                    return np.zeros(self.user_dim, dtype=np.float32)
                # Use item embeddings from P5's shared embedding as proxy
                ids = torch.tensor(user_interaction_ids[:50],
                                   device=self._dummy.device)  # max 50 items
                embs = self.p5.shared(ids)
                user_emb = embs.mean(dim=0).cpu().numpy()
                return user_emb.astype(np.float32)
        else:
            # Simple average of item embeddings from memory encoder
            if len(user_interaction_ids) == 0:
                return np.zeros(self.user_dim, dtype=np.float32)
            ids = torch.tensor(user_interaction_ids[:50])
            embs = self.memory_encoder.item_embed(ids)
            return embs.mean(dim=0).detach().cpu().numpy().astype(np.float32)

    def forward(self, user_emb: np.ndarray,
                user_id: int = None,
                current_time: float = None,
                deterministic: bool = False) -> Dict:
        """
        One-step recommendation decision.

        Args:
            user_emb: (user_dim,) user embedding vector
            user_id: user identifier for memory retrieval
            current_time: unix timestamp
            deterministic: if True, use greedy action (ε=0)

        Returns:
            dict with:
              - action_id: chosen action
              - action_name: human-readable name
              - q_values: (NUM_ACTIONS,) Q-values
              - strategy: candidate retrieval strategy dict
              - state: full state dict (for logging / buffer)
              - confidence: memory retrieval confidence
        """
        # Get memory-augmented state
        state = self.memory_manager.get_state_vector(
            user_id=user_id,
            user_embedding=user_emb,
            current_time=current_time,
        )

        # Convert to tensors
        user_emb_t = torch.from_numpy(user_emb).unsqueeze(0)
        memory_ctx_t = torch.from_numpy(
            np.concatenate([state["short_term_context"],
                            state["long_term_context"]])
        ).unsqueeze(0)
        confidence_t = torch.tensor([[state["retrieval_confidence"]]], dtype=torch.float)
        # Periodic signals: pad to fixed length 10
        psv = list(state["periodic_signals"].values())
        psv_float = [float(v) for v in psv[:10]]
        if len(psv_float) < 10:
            psv_float += [0.0] * (10 - len(psv_float))
        periodic_t = torch.tensor([psv_float], dtype=torch.float32)

        # Epsilon for exploration
        if deterministic:
            epsilon = 0.0
        else:
            epsilon = self.epsilon_scheduler.get_epsilon(
                state["retrieval_confidence"],
                global_step=getattr(self, "_train_step", None),
            )

        # Get action
        with torch.no_grad():
            q_values = self.policy(
                user_emb_t, memory_ctx_t, confidence_t, periodic_t,
            )
            action_id = self.policy.get_action(
                user_emb_t, memory_ctx_t, confidence_t, periodic_t,
                epsilon=epsilon,
            ).item()

        strategy = action_to_candidate_strategy(action_id)

        return {
            "action_id": action_id,
            "action_name": ACTION_REGISTRY[action_id].name,
            "q_values": q_values.squeeze(0).numpy(),
            "strategy": strategy,
            "state": state,
            "confidence": state["retrieval_confidence"],
            "epsilon": epsilon,
        }

    def recommend(self, user_emb: np.ndarray, user_id: int = None,
                  item_pool: List[int] = None, topk: int = 20,
                  deterministic: bool = True) -> Dict:
        """
        High-level recommendation: decide action → retrieve candidates → return ranked list.

        This replaces P5's beam search with a 1-pass policy decision.
        """
        result = self.forward(user_emb, user_id, deterministic=deterministic)
        strategy = result["strategy"]

        # In a full implementation, this would call different candidate
        # retrieval backends based on the strategy. For MVP, return the
        # strategy alongside the action.
        return {
            "action": result["action_name"],
            "strategy": strategy,
            "candidates": item_pool[:topk] if item_pool else [],
            "confidence": result["confidence"],
            "is_exploration": strategy.get("is_exploration", False),
        }

    def add_interaction(self, user_id: int, item_id: int,
                        action_type: int, rating: float,
                        category_id: int = None):
        """Record a new interaction into the memory system."""
        self.memory_manager.add_interaction(
            user_id=user_id,
            item_id=item_id,
            action_type=action_type,
            rating=rating,
            category_id=category_id,
        )

    def get_action_distribution(self, user_emb: np.ndarray,
                                user_id: int = None) -> np.ndarray:
        """Get the softmax distribution over actions (for analysis)."""
        state = self.memory_manager.get_state_vector(
            user_id=user_id,
            user_embedding=user_emb,
        )
        user_emb_t = torch.from_numpy(user_emb).unsqueeze(0)
        memory_ctx_t = torch.from_numpy(
            np.concatenate([state["short_term_context"],
                            state["long_term_context"]])
        ).unsqueeze(0)
        confidence_t = torch.tensor([[state["retrieval_confidence"]]], dtype=torch.float)

        with torch.no_grad():
            q_values = self.policy(user_emb_t, memory_ctx_t, confidence_t)
            probs = F.softmax(q_values, dim=-1).squeeze(0).numpy()

        return probs
