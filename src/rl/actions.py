"""
Discrete macro-action space for the RL recommendation policy.

Actions are high-level intents (<20), each mapping to a specific
recommendation strategy. This is much more efficient than P5's
beam-search text generation.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable


@dataclass
class Action:
    """A single macro-action in the RL policy."""
    id: int
    name: str
    description: str
    is_exploration: bool = False  # exploration actions get bonus reward
    requires_periodic: bool = False  # needs periodic pattern detected


# ============================================================
# Action Definitions (16 actions)
# ============================================================

ACTION_DEFINITIONS: List[Action] = [
    # --- Exploitation (6 actions) ---
    Action(0, "exploit_high_ctr", "Recommend items with highest predicted CTR", is_exploration=False),
    Action(1, "exploit_similar", "Recommend items similar to user's last purchase", is_exploration=False),
    Action(2, "exploit_trending", "Recommend currently trending items in user's categories", is_exploration=False),
    Action(3, "exploit_collaborative", "Recommend items from similar users' purchases", is_exploration=False),
    Action(4, "exploit_price_range", "Recommend items within user's typical price range", is_exploration=False),
    Action(5, "exploit_brand_loyalty", "Recommend items from user's frequently purchased brands", is_exploration=False),

    # --- Exploration (5 actions) ---
    Action(6, "explore_new_category", "Recommend items from categories user hasn't tried", is_exploration=True),
    Action(7, "explore_new_brand", "Recommend items from brands new to the user", is_exploration=True),
    Action(8, "explore_price_up", "Recommend items slightly above typical price range", is_exploration=True),
    Action(9, "explore_price_down", "Recommend budget alternatives to familiar items", is_exploration=True),
    Action(10, "explore_trending_global", "Recommend globally trending items outside user's history", is_exploration=True),

    # --- Memory-driven (3 actions) ---
    Action(11, "remind_periodic_purchase", "Recommend items the user buys periodically", requires_periodic=True),
    Action(12, "remind_abandoned", "Recommend items user viewed but didn't buy", is_exploration=False),
    Action(13, "remind_seasonal", "Recommend items relevant to current season based on history", is_exploration=False),

    # --- Diversity (2 actions) ---
    Action(14, "increase_diversity", "Intentionally diversify the recommendation slate", is_exploration=True),
    Action(15, "serendipity", "Recommend unexpected but potentially interesting items", is_exploration=True),
]

ACTION_REGISTRY: Dict[int, Action] = {a.id: a for a in ACTION_DEFINITIONS}
ACTION_NAMES: Dict[str, int] = {a.name: a.id for a in ACTION_DEFINITIONS}
NUM_ACTIONS = len(ACTION_DEFINITIONS)

# Action groups for masking
EXPLOITATION_ACTIONS = [a.id for a in ACTION_DEFINITIONS if not a.is_exploration]
EXPLORATION_ACTIONS = [a.id for a in ACTION_DEFINITIONS if a.is_exploration]
PERIODIC_ACTIONS = [a.id for a in ACTION_DEFINITIONS if a.requires_periodic]


def get_action_mask(user_history: Dict,
                    purchased_items: set,
                    available_categories: set,
                    has_periodic_patterns: bool = False,
                    is_new_user: bool = False) -> np.ndarray:
    """
    Build an action mask for valid actions given the current user state.

    Invalidation rules:
      - remind_periodic_purchase: masked if no periodic patterns detected
      - remind_abandoned: masked if user has no abandoned items
      - exploit_brand_loyalty: masked if user has <2 purchases from same brand
      - For new users: encourage exploration by masking some exploitation actions
    """
    import numpy as np
    mask = np.ones(NUM_ACTIONS, dtype=np.float32)

    # Periodic actions require detected patterns
    if not has_periodic_patterns:
        for aid in PERIODIC_ACTIONS:
            mask[aid] = 0.0

    # Abandoned reminder needs abandoned items
    if not user_history.get("has_abandoned", False):
        mask[ACTION_NAMES["remind_abandoned"]] = 0.0

    # Brand loyalty needs brand history
    if user_history.get("unique_brands", 0) < 2:
        mask[ACTION_NAMES["exploit_brand_loyalty"]] = 0.0

    # For very new users (cold start), push exploration
    if is_new_user:
        mask[ACTION_NAMES["exploit_brand_loyalty"]] = 0.0
        mask[ACTION_NAMES["exploit_price_range"]] = 0.0
        # Give higher effective probability to exploration by
        # keeping exploitation actions at 1.0 but boosting exploration
        # (handled by epsilon in policy, not masking)

    return mask


def action_to_candidate_strategy(action_id: int) -> Dict:
    """
    Translate a discrete action into a candidate retrieval strategy.
    This replaces P5's beam search with a deterministic item-fetching plan.
    """
    action = ACTION_REGISTRY[action_id]
    strategies = {
        "exploit_high_ctr": {"method": "rank_by_ctr", "topk": 20},
        "exploit_similar": {"method": "similar_to_last", "topk": 20},
        "exploit_trending": {"method": "trending_in_category", "topk": 20},
        "exploit_collaborative": {"method": "collaborative_filter", "topk": 20},
        "exploit_price_range": {"method": "price_band", "topk": 20},
        "exploit_brand_loyalty": {"method": "same_brand", "topk": 20},
        "explore_new_category": {"method": "diverse_categories", "topk": 20, "exclude_seen": True},
        "explore_new_brand": {"method": "diverse_brands", "topk": 20, "exclude_seen": True},
        "explore_price_up": {"method": "price_band", "topk": 20, "percentile": "p70-p90"},
        "explore_price_down": {"method": "price_band", "topk": 20, "percentile": "p10-p40"},
        "explore_trending_global": {"method": "global_trending", "topk": 20},
        "remind_periodic_purchase": {"method": "periodic_recall", "topk": 20},
        "remind_abandoned": {"method": "abandoned_items", "topk": 20},
        "remind_seasonal": {"method": "seasonal", "topk": 20},
        "increase_diversity": {"method": "diverse_mix", "topk": 20, "num_categories": 5},
        "serendipity": {"method": "serendipitous", "topk": 20},
    }
    result = strategies.get(action.name, {"method": "default", "topk": 20})
    result["action_name"] = action.name
    result["is_exploration"] = action.is_exploration
    return result
