"""
Multi-component reward function.

Reward = w_click * R_click
       + w_purchase * R_purchase
       + w_dwell * R_dwell
       + w_diversity * R_diversity
       + w_exploration * R_exploration
       + w_retention * R_retention
       - w_negative * R_negative

This explicitly models negative feedback (skips, returns, bad ratings)
which P5 does not differentiate from positive signals.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class RewardComponents:
    """Decomposed reward — enables analysis of which signals drive performance."""
    click: float = 0.0
    purchase: float = 0.0
    dwell: float = 0.0
    diversity: float = 0.0
    exploration: float = 0.0
    retention: float = 0.0
    negative: float = 0.0
    periodic_recall: float = 0.0

    @property
    def total(self) -> float:
        return (self.click + self.purchase + self.dwell +
                self.diversity + self.exploration + self.retention +
                self.negative + self.periodic_recall)

    def to_dict(self) -> Dict[str, float]:
        return {
            "click": self.click,
            "purchase": self.purchase,
            "dwell": self.dwell,
            "diversity": self.diversity,
            "exploration": self.exploration,
            "retention": self.retention,
            "negative": self.negative,
            "periodic_recall": self.periodic_recall,
            "total": self.total,
        }


class RewardFunction:
    """
    Computes the step-level reward given user feedback to a recommendation.

    Designed to be differential — each component is computed independently
    so we can do ablation: removing negative feedback, removing exploration
    bonus, etc.
    """

    def __init__(self, weights: Optional[dict] = None):
        # Default weights calibrated to make components comparable in scale
        self.w = {
            "click": 1.0,
            "purchase": 3.0,
            "dwell": 0.1,          # per 10s dwell time
            "diversity": 0.15,      # bonus for reaching beyond top categories
            "exploration": 0.1,     # bonus for taking explore_* actions
            "retention": 10.0,      # sparse but very high value
            "negative": -0.5,       # skip/return/bad rating
            "periodic_recall": 2.0, # recalling periodic needs
        }
        if weights:
            self.w.update(weights)

    def compute(self,
                clicked: bool = False,
                purchased: bool = False,
                dwell_time: float = 0.0,
                num_new_categories: int = 0,
                is_exploration_action: bool = False,
                retention_signal: float = 0.0,
                negative_signal: str = None,
                # negative_signal: None / "skip" / "return" / "bad_rating"
                is_periodic_recall: bool = False,
                category_history: set = None,
                recommended_categories: set = None,
                ) -> RewardComponents:
        """
        Compute the full reward from user feedback signals.

        Args:
            clicked: whether user clicked the recommendation
            purchased: whether user purchased
            dwell_time: seconds spent viewing (0 if skip)
            num_new_categories: number of new categories in recommendation
            is_exploration_action: whether the RL action was explore_*
            retention_signal: 1.0 if user stayed, 0.0 if churned (sparse)
            negative_signal: type of negative feedback
            is_periodic_recall: whether this was a periodic purchase recall
            category_history: user's historical categories
            recommended_categories: categories in current recommendation
        """
        r = RewardComponents()

        # Positive signals
        r.click = self.w["click"] * float(clicked)
        r.purchase = self.w["purchase"] * float(purchased)
        r.dwell = self.w["dwell"] * (dwell_time / 10.0)

        # Diversity bonus: user saw items from new categories
        if num_new_categories > 0 and category_history and recommended_categories:
            new_cats = recommended_categories - category_history
            actual_new = min(num_new_categories, len(new_cats))
            r.diversity = self.w["diversity"] * actual_new

        # Exploration bonus for taking exploration actions
        if is_exploration_action:
            r.exploration = self.w["exploration"] * (1.0 + 0.5 * float(clicked))
            # Bonus is larger if exploration led to a click

        # Retention signal (only non-zero on churn/user-stay events)
        r.retention = self.w["retention"] * retention_signal

        # Negative feedback
        neg_map = {"skip": -0.3, "return": -1.0, "bad_rating": -0.5}
        if negative_signal and negative_signal in neg_map:
            r.negative = self.w["negative"] * (-neg_map[negative_signal])

        # Periodic recall bonus
        if is_periodic_recall and (clicked or purchased):
            r.periodic_recall = self.w["periodic_recall"] * float(purchased or clicked)

        return r

    def compute_from_interaction(self, interaction: Dict,
                                 action_info: Dict,
                                 user_state: Dict) -> RewardComponents:
        """Convenience wrapper that extracts fields from interaction dict."""
        return self.compute(
            clicked=interaction.get("clicked", False),
            purchased=interaction.get("purchased", False),
            dwell_time=interaction.get("dwell_time", 0.0),
            num_new_categories=interaction.get("new_categories", 0),
            is_exploration_action=action_info.get("is_exploration", False),
            retention_signal=interaction.get("retention", 0.0),
            negative_signal=interaction.get("negative_signal", None),
            is_periodic_recall=action_info.get("is_periodic_recall", False),
            category_history=user_state.get("category_history", set()),
            recommended_categories=action_info.get("recommended_categories", set()),
        )
