"""
POMDP Environment for RL-based recommendation.

The environment wraps the interaction loop:
  Agent observes user state → chooses action → environment returns reward + next state.
"""
from typing import Dict, Optional, Tuple
from collections import deque

import numpy as np
import torch

from .actions import ACTION_REGISTRY, NUM_ACTIONS, get_action_mask
from .reward import RewardFunction, RewardComponents


class POMDPEnv:
    """
    Recommendation POMDP environment.

    State space (partially observable):
      - User interest embedding (from interaction history)
      - Memory context (short-term + long-term retrieved)
      - Memory retrieval confidence
      - Periodic purchase signals

    Action space: 16 discrete macro-actions (see actions.py)

    Reward: multi-component (see reward.py)
    """

    def __init__(self,
                 memory_manager,
                 user_encoder,  # encodes user interaction history → embedding
                 reward_function: RewardFunction = None,
                 max_steps_per_episode: int = 50,
                 device: str = "cpu"):
        self.memory = memory_manager
        self.user_encoder = user_encoder
        self.reward_fn = reward_function or RewardFunction()
        self.max_steps = max_steps_per_episode
        self.device = device

        # Episode state
        self.current_user_id: Optional[int] = None
        self.current_user_emb: Optional[np.ndarray] = None
        self.step_count: int = 0
        self.episode_rewards: list = []
        self.category_history: set = set()

    def reset(self, user_id: int, initial_user_emb: np.ndarray = None,
              category_history: set = None) -> Dict:
        """
        Start a new episode for a user.

        Returns the initial POMDP state dict.
        """
        self.current_user_id = user_id
        self.step_count = 0
        self.episode_rewards = []
        self.category_history = category_history or set()

        if initial_user_emb is None:
            initial_user_emb = np.zeros(256, dtype=np.float32)

        self.current_user_emb = initial_user_emb

        return self._get_state()

    def step(self, action_id: int,
             user_feedback: Dict = None) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one recommendation action and observe user feedback.

        Args:
            action_id: index of the chosen macro-action
            user_feedback: dict with keys:
              - clicked: bool
              - purchased: bool
              - dwell_time: float
              - item_categories: set of recommended item categories
              - negative_signal: str or None
              - retention: float (1.0 or 0.0)
              - item_id: int
              - rating: float [0,1]

        Returns:
            next_state: dict
            reward: float
            done: bool
            info: dict with RewardComponents
        """
        if user_feedback is None:
            user_feedback = {}

        action = ACTION_REGISTRY[action_id]
        self.step_count += 1

        # Compute reward
        reward_components = self.reward_fn.compute(
            clicked=user_feedback.get("clicked", False),
            purchased=user_feedback.get("purchased", False),
            dwell_time=user_feedback.get("dwell_time", 0.0),
            num_new_categories=user_feedback.get("new_categories", 0),
            is_exploration_action=action.is_exploration,
            retention_signal=user_feedback.get("retention", 0.0),
            negative_signal=user_feedback.get("negative_signal"),
            is_periodic_recall=action.requires_periodic,
            category_history=self.category_history,
            recommended_categories=user_feedback.get("recommended_categories", set()),
        )

        reward = reward_components.total
        self.episode_rewards.append(reward)

        # Update category history
        if user_feedback.get("recommended_categories"):
            self.category_history.update(user_feedback["recommended_categories"])

        # Add interaction to memory
        if user_feedback.get("item_id") is not None:
            action_type = 1 if user_feedback.get("purchased") else (
                2 if user_feedback.get("negative_signal") == "skip" else 0
            )
            self.memory.add_interaction(
                user_id=self.current_user_id,
                item_id=user_feedback["item_id"],
                action_type=action_type,
                rating=user_feedback.get("rating", 0.5),
                timestamp=user_feedback.get("timestamp"),
                category_id=user_feedback.get("category_id"),
            )

        # Termination
        done = (self.step_count >= self.max_steps or
                user_feedback.get("retention", 1.0) <= 0.0)

        # Get next state
        next_state = self._get_state()

        info = {
            "reward_components": reward_components,
            "action_name": action.name,
            "step": self.step_count,
        }

        return next_state, reward, done, info

    def _get_state(self) -> Dict:
        """Construct the current POMDP state."""
        memory_state = self.memory.get_state_vector(
            user_id=self.current_user_id,
            user_embedding=self.current_user_emb,
        )

        # Build periodic flags (top-10 due items)
        periodic_flags = np.zeros(10, dtype=np.float32)
        periodic_items = list(memory_state["periodic_signals"].items())
        for i, (item_id, is_due) in enumerate(periodic_items[:10]):
            periodic_flags[i] = float(is_due)

        # Build action mask
        has_periodic = any(memory_state["periodic_signals"].values())

        action_mask = get_action_mask(
            user_history={"has_abandoned": False, "unique_brands": 0},
            purchased_items=set(),
            available_categories=self.category_history,
            has_periodic_patterns=has_periodic,
            is_new_user=(len(self.category_history) <= 2),
        )

        return {
            "user_emb": self.current_user_emb.astype(np.float32),
            "memory_context": np.concatenate([
                memory_state["short_term_context"],
                memory_state["long_term_context"],
            ]).astype(np.float32),
            "retrieval_confidence": np.array([memory_state["retrieval_confidence"]],
                                             dtype=np.float32),
            "periodic_flags": periodic_flags,
            "action_mask": action_mask,
            "user_id": self.current_user_id,
        }

    def get_state_tensors(self, state: Dict) -> Tuple[torch.Tensor, ...]:
        """Convert state dict to tensors for policy network."""
        return (
            torch.from_numpy(state["user_emb"]).unsqueeze(0).to(self.device),
            torch.from_numpy(state["memory_context"]).unsqueeze(0).to(self.device),
            torch.from_numpy(state["retrieval_confidence"]).unsqueeze(0).to(self.device),
            torch.from_numpy(state["periodic_flags"]).unsqueeze(0).to(self.device),
            torch.from_numpy(state["action_mask"]).unsqueeze(0).to(self.device),
        )
