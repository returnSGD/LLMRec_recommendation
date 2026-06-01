"""
Online RL Environment Simulation for Exploration-Exploitation Tradeoff Analysis.

Simulates user-agent interaction over multiple timesteps to evaluate how
exploration parameters affect the exploitation-exploration balance.

Key insight to demonstrate (from idea.md §4.2):
  Short-term CTR may drop with exploration, but long-term retention and
  cumulative reward exceed pure-exploitation baselines.

Architecture:
  SimulatedUser (preference profile, periodic patterns, novelty-seeking)
       ↕  feedback (click/purchase/dwell/skip)
  OnlineEnv (wraps the agent's policy → strategy → item retrieval loop)
       ↕  action + state
  POMDPPolicy (the same policy network from src/rl/)
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Add project root so we can import existing RL components
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.rl.policy import POMDPPolicy, UncertaintyAdaptiveEpsilon
from src.rl.actions import (
    ACTION_REGISTRY, NUM_ACTIONS, ACTION_DEFINITIONS,
    EXPLOITATION_ACTIONS, EXPLORATION_ACTIONS,
)
from src.rl.reward import RewardFunction, RewardComponents
from src.memory.manager import MemoryManager, ShortTermMemory
from src.memory.encoder import MemoryEncoder

# ===========================================================================
# 1. Simulated World — Users & Items
# ===========================================================================

@dataclass
class SimItem:
    """A simulated item with ground-truth properties."""
    id: int
    category: int          # semantic category (0–N_cat-1)
    brand: int
    price: float           # normalized [0, 1]
    base_ctr: float        # population-level CTR
    embedding: np.ndarray  # (D,) dense feature vector
    quality: float         # [0,1] true quality — drives purchase probability


@dataclass
class UserProfile:
    """A simulated user with diverse, learnable preference structure."""
    id: int
    # Core preferences: which categories & brands the user genuinely likes
    preferred_categories: List[int]       # top-3 liked categories
    disfavored_categories: List[int]      # categories to avoid
    brand_loyalty: Dict[int, float]       # brand_id → loyalty weight [0,1]
    price_sensitivity: float              # 0=insensitive, 1=highly sensitive
    novelty_seeking: float                # 0=pure exploitation, 1=always wants new
    # Periodic purchase pattern
    periodic_item_categories: Dict[int, float]  # category_id → interval (days)
    last_periodic_purchase_time: Dict[int, float] = field(default_factory=dict)
    # Learning dynamics — user's interest evolves over time
    interest_drift_rate: float = 0.02     # how fast preferences shift
    fatigue_rate: float = 0.05            # how fast repeated items get stale
    # Memory — tracks exposure count per item (for fatigue modeling)
    exposure_counts: Dict[int, int] = field(default_factory=dict)
    newly_discovered_categories: set = field(default_factory=set)


# ===========================================================================
# 2. Simulated Feedback Model
# ===========================================================================

class SimulatedFeedback:
    """
    Models how a simulated user responds to a recommended slate of items.

    Given the user profile + recommended items, generates:
      - clicked: bool (whether user clicked any item)
      - purchased: bool
      - dwell_time: float (seconds, proxy for engagement)
      - negative_signal: str / None ("skip" if no click)
      - item_id: int (the item user interacted with, if any)
      - rating: float [0,1]
      - category_id: int
      - recommended_categories: set
      - new_categories: int (number of novel categories in the slate)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def respond(self, user: UserProfile,
                recommended_items: List[SimItem],
                action_is_exploration: bool) -> dict:
        if not recommended_items:
            return self._empty_response()

        # Choose which item the user engages with (attention model)
        scores = []
        for item in recommended_items:
            s = self._item_appeal(user, item, action_is_exploration)
            scores.append(s)
        scores = np.array(scores, dtype=np.float32)

        # Softmax selection
        probs = np.exp(np.clip(scores * 2, -10, 10))
        probs /= probs.sum() + 1e-8

        chosen_idx = int(self.rng.choice(len(recommended_items), p=probs))
        chosen_item = recommended_items[chosen_idx]
        appeal = float(np.clip(scores[chosen_idx], 0, 1))

        # Click decision: probability ≈ sigmoid(appeal - 0.3)
        click_prob = 1.0 / (1.0 + math.exp(-(appeal - 0.3) * 8))
        clicked = self.rng.rand() < click_prob

        # Purchase decision: higher quality threshold
        if clicked:
            purchase_prob = chosen_item.quality * appeal * 0.6
            purchased = self.rng.rand() < purchase_prob
        else:
            purchased = False

        # Dwell time
        if purchased:
            dwell = self.rng.uniform(30, 120)
        elif clicked:
            dwell = self.rng.uniform(5, 30)
        else:
            dwell = self.rng.uniform(0.2, 2.0)

        # Negative signal
        if not clicked:
            negative_signal = "skip"
        elif not purchased and dwell < 3:
            negative_signal = "skip"
        else:
            negative_signal = None

        # Rating
        if purchased:
            rating = self.rng.uniform(0.6, 1.0)
        elif clicked:
            rating = self.rng.uniform(0.2, 0.8)
        else:
            rating = self.rng.uniform(0.0, 0.3)

        # Categories
        recommended_cats = {it.category for it in recommended_items}
        known_cats = set(user.preferred_categories)
        new_cats_found = len(recommended_cats - known_cats -
                             set(user.disfavored_categories))

        # Update user's discovered categories
        if new_cats_found > 0 and (clicked or purchased):
            for cat in (recommended_cats - known_cats -
                        set(user.disfavored_categories)):
                user.newly_discovered_categories.add(cat)

        # Update exposure counts (fatigue modeling)
        for item in recommended_items:
            user.exposure_counts[item.id] = user.exposure_counts.get(item.id, 0) + 1

        return {
            "clicked": clicked,
            "purchased": purchased,
            "dwell_time": dwell,
            "negative_signal": negative_signal,
            "item_id": chosen_item.id,
            "rating": rating,
            "category_id": chosen_item.category,
            "recommended_categories": recommended_cats,
            "new_categories": new_cats_found,
            "retention": 1.0,  # always retained in simulation (we track via cumulative reward)
        }

    def _item_appeal(self, user: UserProfile, item: SimItem,
                     is_exploration: bool) -> float:
        """Compute how much a user likes an item [0, 1]."""
        # Category match
        if item.category in user.preferred_categories:
            cat_score = 0.8
        elif item.category in user.disfavored_categories:
            cat_score = 0.05
        else:
            cat_score = 0.3  # neutral

        # Brand loyalty
        brand_weight = user.brand_loyalty.get(item.brand, 0.1)
        brand_score = 0.3 + 0.7 * brand_weight

        # Price sensitivity
        price_gap = abs(item.price - 0.5)  # assume user prefers mid-range
        price_score = 1.0 - price_gap * user.price_sensitivity

        # Fatigue: repeated exposure reduces appeal
        exposure = user.exposure_counts.get(item.id, 0)
        fatigue_penalty = math.exp(-user.fatigue_rate * exposure)

        # Novelty bonus: exploration actions reach new items
        if is_exploration and item.category not in user.preferred_categories:
            novelty_bonus = 1.0 + 0.5 * user.novelty_seeking
        else:
            novelty_bonus = 1.0

        # Periodicity: items in periodic categories get boost when "due"
        periodic_boost = 1.0
        for cat, interval in user.periodic_item_categories.items():
            if item.category == cat:
                last_purchase = user.last_periodic_purchase_time.get(cat, -100)
                time_since = time.time() / 86400.0 - last_purchase  # days
                if time_since > interval * 0.8:
                    periodic_boost = 2.0  # strong boost when due

        appeal = (cat_score * 0.4 + brand_score * 0.2 + price_score * 0.15 +
                  item.quality * 0.25)
        appeal *= fatigue_penalty * novelty_bonus * periodic_boost
        return float(np.clip(appeal, 0.0, 1.0))

    @staticmethod
    def _empty_response() -> dict:
        return {
            "clicked": False, "purchased": False, "dwell_time": 0.0,
            "negative_signal": "skip", "item_id": -1, "rating": 0.0,
            "category_id": -1, "recommended_categories": set(),
            "new_categories": 0, "retention": 1.0,
        }


# ===========================================================================
# 3. Simulated Item Pool & Strategy Retriever
# ===========================================================================

class SimItemPool:
    """Generates and manages a simulated item catalog."""

    def __init__(self, n_items: int = 2000, n_categories: int = 20,
                 n_brands: int = 50, embed_dim: int = 128, seed: int = 42):
        self.n_items = n_items
        self.n_categories = n_categories
        self.n_brands = n_brands
        self.rng = np.random.RandomState(seed)
        self.items: Dict[int, SimItem] = {}

        # Category-level embeddings (prototypes)
        self.cat_embeddings = self.rng.randn(n_categories, embed_dim).astype(np.float32)
        self.cat_embeddings = self.cat_embeddings / (
            np.linalg.norm(self.cat_embeddings, axis=1, keepdims=True) + 1e-8)

        # Brand embeddings
        self.brand_embeddings = self.rng.randn(n_brands, embed_dim).astype(np.float32)

        self._generate_items(embed_dim)

        # Build inverse indices for fast retrieval
        self._by_category: Dict[int, List[int]] = defaultdict(list)
        self._by_brand: Dict[int, List[int]] = defaultdict(list)
        for it in self.items.values():
            self._by_category[it.category].append(it.id)
            self._by_brand[it.brand].append(it.id)

        # Global trending (fake: use most recent 10% as trending)
        all_ids = sorted(self.items.keys())
        self.trending_ids = all_ids[-int(n_items * 0.1):]

        # CTR-based ranking
        self.ctr_ranking = sorted(all_ids,
                                  key=lambda iid: self.items[iid].base_ctr,
                                  reverse=True)

    def _generate_items(self, embed_dim: int):
        for i in range(self.n_items):
            cat = int(self.rng.randint(0, self.n_categories))
            brand = int(self.rng.randint(0, self.n_brands))
            quality = float(np.clip(self.rng.beta(2, 3), 0.05, 1.0))
            # Base CTR correlated with quality
            base_ctr = float(np.clip(quality * 0.7 + self.rng.uniform(0, 0.3), 0.01, 0.5))

            # Embedding = category prototype + noise
            emb = self.cat_embeddings[cat] + 0.2 * self.rng.randn(embed_dim).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)

            self.items[i] = SimItem(
                id=i, category=cat, brand=brand,
                price=float(self.rng.uniform(0.1, 1.0)),
                base_ctr=base_ctr, embedding=emb, quality=quality,
            )

    def retrieve(self, strategy: dict, user: UserProfile,
                 topk: int = 20) -> List[SimItem]:
        """
        Implements the same retrieval logic as `action_to_candidate_strategy`.

        Different strategies → different candidate sets. This is the key
        mechanism for exploration vs exploitation in item space.
        """
        method = strategy.get("method", "rank_by_ctr")
        exclude_seen = strategy.get("exclude_seen", False)
        is_exploration = strategy.get("is_exploration", False)

        pool_ids = set(self.items.keys())

        if exclude_seen:
            seen_ids = {iid for iid, c in user.exposure_counts.items() if c > 0}
            pool_ids -= seen_ids

        if not pool_ids:
            pool_ids = set(self.items.keys())  # fallback

        if method == "rank_by_ctr":
            candidates = [iid for iid in self.ctr_ranking if iid in pool_ids]
            return [self.items[iid] for iid in candidates[:topk]]

        elif method == "similar_to_last":
            # Use user's preferred category as proxy for "last interaction"
            if user.preferred_categories:
                cat = user.preferred_categories[0]
                cand = self._by_category.get(cat, list(pool_ids))
            else:
                cand = list(pool_ids)
            scored = sorted(cand, key=lambda iid: self.items[iid].quality, reverse=True)
            return [self.items[iid] for iid in scored[:topk]]

        elif method == "trending_in_category":
            if user.preferred_categories:
                cat = user.preferred_categories[0]
                cand = [iid for iid in self.trending_ids
                        if iid in pool_ids and self.items[iid].category == cat]
            else:
                cand = self.trending_ids
            return [self.items[iid] for iid in cand[:topk]]

        elif method == "collaborative_filter":
            # Simulate: items from same categories as preferred
            cand = []
            for cat in user.preferred_categories:
                cand.extend(self._by_category.get(cat, []))
            cand = [iid for iid in cand if iid in pool_ids]
            scored = sorted(cand, key=lambda iid: self.items[iid].quality, reverse=True)
            return [self.items[iid] for iid in scored[:topk]]

        elif method == "price_band":
            percentile = strategy.get("percentile", "p30-p70")
            all_prices = sorted(self.items[iid].price for iid in pool_ids)
            if percentile == "p70-p90":
                lo, hi = np.percentile(all_prices, [70, 90])
            elif percentile == "p10-p40":
                lo, hi = np.percentile(all_prices, [10, 40])
            else:
                lo, hi = np.percentile(all_prices, [30, 70])
            cand = [iid for iid in pool_ids if lo <= self.items[iid].price <= hi]
            return [self.items[iid] for iid in cand[:topk]]

        elif method == "same_brand":
            # Pick most frequent brand
            if user.brand_loyalty:
                top_brand = max(user.brand_loyalty, key=user.brand_loyalty.get)
                cand = self._by_brand.get(top_brand, list(pool_ids))
            else:
                cand = list(pool_ids)
            return [self.items[iid] for iid in cand[:topk]]

        elif method in ("diverse_categories", "diverse_brands", "diverse_mix"):
            # Pick items from diverse categories
            cand = list(pool_ids)
            self.rng.shuffle(cand)
            seen_cats = set()
            diverse = []
            for iid in cand:
                if self.items[iid].category not in seen_cats:
                    diverse.append(iid)
                    seen_cats.add(self.items[iid].category)
                if len(diverse) >= topk:
                    break
            return [self.items[iid] for iid in diverse[:topk]]

        elif method == "global_trending":
            cand = [iid for iid in self.trending_ids if iid in pool_ids]
            return [self.items[iid] for iid in cand[:topk]]

        elif method == "serendipitous":
            # Random but high quality
            cand = [iid for iid in pool_ids if self.items[iid].quality > 0.7]
            self.rng.shuffle(cand)
            return [self.items[iid] for iid in cand[:topk]]

        else:  # default: rank by quality + CTR
            cand = list(pool_ids)
            scored = sorted(cand, key=lambda iid: (self.items[iid].quality + self.items[iid].base_ctr),
                          reverse=True)
            return [self.items[iid] for iid in scored[:topk]]


# ===========================================================================
# 4. Online Simulation Loop
# ===========================================================================

@dataclass
class StepResult:
    step: int
    action_id: int
    action_name: str
    is_exploration: bool
    epsilon: float
    reward: float
    reward_components: Dict[str, float]
    clicked: bool
    purchased: bool
    num_new_categories: int
    num_unique_categories_so_far: int


class OnlineSimulator:
    """
    Runs the online interaction loop: Agent ↔ Simulated User.

    Each step:
      1. Agent observes state (user emb + memory context + confidence)
      2. Agent selects action (epsilon-greedy)
      3. Strategy → retrieve item candidates
      4. Simulated user responds (click/purchase/skip/dwell)
      5. Reward computed → memory updated → next state
    """

    def __init__(self,
                 policy: POMDPPolicy,
                 item_pool: SimItemPool,
                 reward_fn: RewardFunction,
                 epsilon_scheduler: UncertaintyAdaptiveEpsilon,
                 memory_manager: MemoryManager,
                 memory_encoder: MemoryEncoder,
                 device: str = "cpu",
                 seed: int = 42):
        self.policy = policy.to(device)
        self.policy.eval()
        self.item_pool = item_pool
        self.reward_fn = reward_fn
        self.epsilon_scheduler = epsilon_scheduler
        self.memory = memory_manager
        self.memory_encoder = memory_encoder
        self.device = device
        self.rng = np.random.RandomState(seed)
        self.feedback_model = SimulatedFeedback(seed=seed)

        # For managing user embeddings
        self._user_embedding: Dict[int, np.ndarray] = {}  # user_id → embedding

    def run_episode(self, user: UserProfile,
                    max_steps: int = 50,
                    deterministic: bool = False,
                    global_step: int = 0,
                    decay_steps: int = 10000) -> List[StepResult]:
        """Run a full episode for one user."""

        # Initialize or get user embedding
        if user.id not in self._user_embedding:
            self._user_embedding[user.id] = self._make_user_embedding(user)

        user_emb = self._user_embedding[user.id]
        history = []

        for step in range(max_steps):
            # --- Get memory-augmented state ---
            memory_state = self.memory.get_state_vector(
                user_id=user.id,
                user_embedding=user_emb,
            )

            # Action mask
            from src.rl.actions import get_action_mask
            has_periodic = len(memory_state["periodic_signals"]) > 0
            action_mask = get_action_mask(
                user_history={"has_abandoned": False, "unique_brands": len(user.brand_loyalty)},
                purchased_items=set(),
                available_categories=set(user.preferred_categories) | user.newly_discovered_categories,
                has_periodic_patterns=has_periodic,
                is_new_user=(len(user.exposure_counts) <= 3),
            )

            # --- Epsilon ---
            if deterministic:
                epsilon = 0.0
            else:
                epsilon = self.epsilon_scheduler.get_epsilon(
                    retrieval_confidence=memory_state["retrieval_confidence"],
                    global_step=global_step,
                    decay_steps=decay_steps,
                )

            # --- Policy forward ---
            with torch.no_grad():
                user_emb_t = torch.from_numpy(user_emb).unsqueeze(0).to(self.device)
                mem_ctx = np.concatenate([
                    memory_state["short_term_context"],
                    memory_state["long_term_context"],
                ]).astype(np.float32)
                mem_ctx_t = torch.from_numpy(mem_ctx).unsqueeze(0).to(self.device)
                conf_t = torch.tensor([[memory_state["retrieval_confidence"]]],
                                      dtype=torch.float32).to(self.device)
                mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
                # Periodic flags: pad to fixed length 10
                psv = list(memory_state["periodic_signals"].values())
                psv_float = [float(v) for v in psv[:10]]
                if len(psv_float) < 10:
                    psv_float += [0.0] * (10 - len(psv_float))
                periodic_t = torch.tensor([psv_float], dtype=torch.float32).to(self.device)

                action_id = self.policy.get_action(
                    user_emb_t, mem_ctx_t, conf_t,
                    periodic_flags=periodic_t,
                    action_mask=mask_t,
                    epsilon=epsilon,
                ).item()

            action = ACTION_REGISTRY[action_id]

            # --- Retrieve candidates ---
            from src.rl.actions import action_to_candidate_strategy
            strategy = action_to_candidate_strategy(action_id)
            recommended = self.item_pool.retrieve(strategy, user, topk=20)

            # --- Simulated feedback ---
            feedback = self.feedback_model.respond(
                user, recommended, action.is_exploration,
            )

            # --- Reward ---
            reward_components = self.reward_fn.compute(
                clicked=feedback["clicked"],
                purchased=feedback["purchased"],
                dwell_time=feedback["dwell_time"],
                num_new_categories=feedback["new_categories"],
                is_exploration_action=action.is_exploration,
                retention_signal=feedback["retention"],
                negative_signal=feedback["negative_signal"],
                is_periodic_recall=action.requires_periodic,
                category_history=set(user.preferred_categories),
                recommended_categories=feedback["recommended_categories"],
            )

            # --- Update memory ---
            self.memory.add_interaction(
                user_id=user.id,
                item_id=feedback["item_id"],
                action_type=1 if feedback["purchased"] else (
                    2 if feedback["negative_signal"] == "skip" else 0),
                rating=feedback["rating"],
                category_id=feedback.get("category_id"),
            )

            # --- Update user embedding (simple moving average shift) ---
            if feedback["clicked"] or feedback["purchased"]:
                item = self.item_pool.items.get(feedback["item_id"])
                if item is not None:
                    item_emb = item.embedding[:user_emb.shape[0]]  # ensure shape match
                    lr = 0.1
                    user_emb = (1 - lr) * user_emb + lr * item_emb
                    user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
            self._user_embedding[user.id] = user_emb

            # --- Log ---
            history.append(StepResult(
                step=step,
                action_id=action_id,
                action_name=action.name,
                is_exploration=action.is_exploration,
                epsilon=epsilon,
                reward=reward_components.total,
                reward_components=reward_components.to_dict(),
                clicked=feedback["clicked"],
                purchased=feedback["purchased"],
                num_new_categories=feedback["new_categories"],
                num_unique_categories_so_far=len(user.newly_discovered_categories) + len(user.preferred_categories),
            ))

            # Drift user preferences (very slowly)
            if self.rng.rand() < user.interest_drift_rate and user.newly_discovered_categories:
                new_cat = self.rng.choice(list(user.newly_discovered_categories))
                if new_cat not in user.preferred_categories:
                    if self.rng.rand() < 0.3:
                        replace_idx = self.rng.randint(0, len(user.preferred_categories))
                        user.preferred_categories[replace_idx] = new_cat

        return history


    def _make_user_embedding(self, user: UserProfile) -> np.ndarray:
        """Create initial user embedding from preferred categories."""
        cats = user.preferred_categories
        if cats:
            embs = [self.item_pool.cat_embeddings[c] for c in cats]
            return np.mean(embs, axis=0).astype(np.float32)
        return np.zeros(128, dtype=np.float32)


# ===========================================================================
# 5. User Generator
# ===========================================================================

def generate_users(n_users: int, item_pool: SimItemPool,
                   seed: int = 42) -> List[UserProfile]:
    """Create diverse simulated user profiles."""
    rng = np.random.RandomState(seed)
    users = []
    all_cats = list(range(item_pool.n_categories))

    for uid in range(n_users):
        # Pick 2-4 preferred categories
        n_pref = rng.randint(2, 5)
        pref_cats = sorted(rng.choice(all_cats, n_pref, replace=False).tolist())

        # 0-2 disfavored categories
        remaining = [c for c in all_cats if c not in pref_cats]
        n_disf = rng.randint(0, min(3, len(remaining)))
        disf_cats = sorted(rng.choice(remaining, n_disf, replace=False).tolist()) if n_disf > 0 else []

        # Brand loyalty (10-30% of brands the user likes)
        n_loyal = rng.randint(2, 8)
        loyal_brands = rng.choice(item_pool.n_brands, n_loyal, replace=False)
        brand_loyalty = {int(b): float(np.clip(rng.beta(2, 1), 0.1, 1.0))
                         for b in loyal_brands}

        # Price sensitivity
        price_sensitivity = float(np.clip(rng.normal(0.5, 0.2), 0.1, 1.0))

        # Novelty seeking (key parameter for exploration experiments)
        novelty_seeking = float(np.clip(rng.beta(2, 3), 0.05, 0.95))

        # Periodic purchase patterns (30% of users)
        periodic = {}
        if rng.rand() < 0.3:
            n_periodic = rng.randint(1, 3)
            for _ in range(n_periodic):
                cat = rng.choice(pref_cats)
                interval = rng.choice([7, 14, 21, 30, 60, 90])  # days
                periodic[int(cat)] = float(interval)

        users.append(UserProfile(
            id=uid,
            preferred_categories=pref_cats,
            disfavored_categories=disf_cats,
            brand_loyalty=brand_loyalty,
            price_sensitivity=price_sensitivity,
            novelty_seeking=novelty_seeking,
            periodic_item_categories=periodic,
            interest_drift_rate=float(np.clip(rng.normal(0.02, 0.01), 0.005, 0.05)),
            fatigue_rate=float(np.clip(rng.normal(0.05, 0.02), 0.01, 0.1)),
        ))

    return users


# ===========================================================================
# 6. Experiment Runner — Sweeps & Analysis
# ===========================================================================

@dataclass
class SweepConfig:
    """Single parameter configuration for a sweep run."""
    label: str
    epsilon_base: float = 0.05
    epsilon_max: float = 0.5
    use_uncertainty_adaptive: bool = True
    exploration_bonus_weight: float = 0.1
    entropy_coef: float = 0.01
    deterministic: bool = False


@dataclass
class SweepResult:
    config: SweepConfig
    # Aggregated metrics over all users & steps
    total_reward: float
    mean_step_reward: float
    click_rate: float
    purchase_rate: float
    exploration_action_rate: float        # fraction of exploration actions taken
    action_entropy: float                 # entropy of action distribution
    unique_actions_used: int
    unique_categories_discovered: float   # avg per user
    final_user_embedding_diversity: float # pairwise cosine distance among users
    per_step_rewards: List[float]
    per_step_clicks: List[float]
    action_counts: Dict[str, int]


class EETradeoffExperiment:
    """
    Runs sweeps over exploration parameters and collects tradeoff metrics.
    """

    def __init__(self,
                 n_users: int = 100,
                 n_items: int = 2000,
                 max_steps_per_episode: int = 50,
                 device: str = "cpu",
                 seed: int = 42):
        self.n_users = n_users
        self.n_items = n_items
        self.max_steps = max_steps_per_episode
        self.device = device
        self.seed = seed

        # Create simulated world
        self.item_pool = SimItemPool(n_items=n_items, seed=seed)
        self.all_users = generate_users(n_users, self.item_pool, seed=seed)

    def _build_agent(self, sweep_cfg: SweepConfig):
        """Construct a fresh agent for a given sweep configuration."""
        from src.rl.reward import RewardFunction

        # Memory system
        memory_encoder = MemoryEncoder(
            num_users=self.n_users, num_items=self.n_items,
            embed_dim=128, memory_dim=256,
        )
        memory = MemoryManager(
            memory_dim=256, short_term_capacity=50, top_k=20,
            time_decay_lambda=0.01,
        )
        memory.set_encoder(memory_encoder)

        # Policy
        policy = POMDPPolicy(
            user_dim=128,
            memory_dim=256,
            num_actions=NUM_ACTIONS,
            hidden_dim=128,
            num_layers=2,
            dropout=0.0,
        )

        # Epsilon scheduler
        epsilon_scheduler = UncertaintyAdaptiveEpsilon(
            epsilon_base=sweep_cfg.epsilon_base,
            epsilon_max=sweep_cfg.epsilon_max,
            uncertainty_scale=1.0 if sweep_cfg.use_uncertainty_adaptive else 0.0,
        )

        # Reward function (tune exploration bonus)
        reward_fn = RewardFunction(weights={
            "click": 1.0,
            "purchase": 3.0,
            "dwell": 0.1,
            "diversity": 0.15,
            "exploration": sweep_cfg.exploration_bonus_weight,
            "retention": 10.0,
            "negative": -0.5,
            "periodic_recall": 2.0,
        })

        return OnlineSimulator(
            policy=policy, item_pool=self.item_pool,
            reward_fn=reward_fn, epsilon_scheduler=epsilon_scheduler,
            memory_manager=memory, memory_encoder=memory_encoder,
            device=self.device, seed=self.seed,
        )

    def sweep(self, configs: List[SweepConfig]) -> List[SweepResult]:
        """Run the simulation for each configuration."""
        results = []
        for i, cfg in enumerate(configs):
            print(f"[{i+1}/{len(configs)}] {cfg.label} "
                  f"(ε_base={cfg.epsilon_base}, adaptive={cfg.use_uncertainty_adaptive}, "
                  f"expl_bonus={cfg.exploration_bonus_weight})")
            sim = self._build_agent(cfg)

            all_histories = []
            for user in self.all_users:
                # Reset user state for each config
                user_copy = copy.deepcopy(user)
                history = sim.run_episode(
                    user_copy,
                    max_steps=self.max_steps,
                    deterministic=cfg.deterministic,
                    decay_steps=10000,
                )
                all_histories.append(history)

            results.append(self._aggregate(cfg, all_histories))
        return results

    def _aggregate(self, cfg: SweepConfig,
                   all_histories: List[List[StepResult]]) -> SweepResult:
        """Compute aggregate metrics from all episode histories."""
        all_steps = [s for h in all_histories for s in h]
        n_steps = len(all_steps)
        n_users = len(all_histories)

        total_reward = sum(s.reward for s in all_steps)
        click_count = sum(1 for s in all_steps if s.clicked)
        purchase_count = sum(1 for s in all_steps if s.purchased)
        explore_count = sum(1 for s in all_steps if s.is_exploration)

        # Action distribution entropy
        action_counts = defaultdict(int)
        for s in all_steps:
            action_counts[s.action_name] += 1
        total_actions = sum(action_counts.values())
        probs = np.array([c / total_actions for c in action_counts.values()])
        action_entropy = float(-np.sum(probs * np.log(probs + 1e-8)))

        # Unique categories discovered per user
        unique_cats = []
        for hist in all_histories:
            if hist:
                unique_cats.append(hist[-1].num_unique_categories_so_far)

        # Per-step curves
        step_rewards = defaultdict(list)
        step_clicks = defaultdict(list)
        for hist in all_histories:
            for s in hist:
                step_rewards[s.step].append(s.reward)
                step_clicks[s.step].append(float(s.clicked))

        per_step_rewards = [np.mean(step_rewards[i]) for i in range(self.max_steps)
                           if i in step_rewards]
        per_step_clicks = [np.mean(step_clicks[i]) for i in range(self.max_steps)
                          if i in step_clicks]

        return SweepResult(
            config=cfg,
            total_reward=total_reward,
            mean_step_reward=total_reward / n_steps if n_steps else 0,
            click_rate=click_count / n_steps if n_steps else 0,
            purchase_rate=purchase_count / n_steps if n_steps else 0,
            exploration_action_rate=explore_count / n_steps if n_steps else 0,
            action_entropy=action_entropy,
            unique_actions_used=len(action_counts),
            unique_categories_discovered=np.mean(unique_cats) if unique_cats else 0,
            final_user_embedding_diversity=0.0,  # computed separately if needed
            per_step_rewards=per_step_rewards,
            per_step_clicks=per_step_clicks,
            action_counts=dict(action_counts),
        )


# ===========================================================================
# 7. Pre-defined Sweep Configurations
# ===========================================================================

def build_sweep_configs() -> List[SweepConfig]:
    """Generate a comprehensive set of exploration configurations."""

    configs = []

    # --- Pure exploitation baseline ---
    configs.append(SweepConfig(
        label="Pure Exploit (ε=0)",
        epsilon_base=0.0, epsilon_max=0.0,
        use_uncertainty_adaptive=False,
        exploration_bonus_weight=0.0,
        deterministic=True,
    ))

    # --- Epsilon sweep (fixed, no adaptive) ---
    for eps in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        configs.append(SweepConfig(
            label=f"Fixed ε={eps}",
            epsilon_base=eps, epsilon_max=eps,
            use_uncertainty_adaptive=False,
            exploration_bonus_weight=0.1,
        ))

    # --- Uncertainty-adaptive epsilon sweep ---
    for eps_base in [0.02, 0.05, 0.10]:
        for eps_max in [0.3, 0.5, 0.7]:
            configs.append(SweepConfig(
                label=f"Adaptive ε (base={eps_base}, max={eps_max})",
                epsilon_base=eps_base, epsilon_max=eps_max,
                use_uncertainty_adaptive=True,
                exploration_bonus_weight=0.1,
            ))

    # --- Exploration bonus sweep ---
    for bonus in [0.0, 0.05, 0.15, 0.3, 0.5]:
        configs.append(SweepConfig(
            label=f"Expl Bonus={bonus}",
            epsilon_base=0.05, epsilon_max=0.5,
            use_uncertainty_adaptive=True,
            exploration_bonus_weight=bonus,
        ))

    # --- Fixed ε with exploration bonus ---
    for eps in [0.1, 0.3]:
        for bonus in [0.05, 0.2, 0.5]:
            configs.append(SweepConfig(
                label=f"Fixed ε={eps} + Bonus={bonus}",
                epsilon_base=eps, epsilon_max=eps,
                use_uncertainty_adaptive=False,
                exploration_bonus_weight=bonus,
            ))

    return configs


# ===========================================================================
# 8. Visualization & Reporting
# ===========================================================================

def print_tradeoff_summary(results: List[SweepResult]):
    """Print a formatted summary table."""
    print("\n" + "=" * 120)
    print("Exploration-Exploitation Tradeoff Summary")
    print("=" * 120)
    header = (f"{'Config':<40} {'TotalR':>8} {'MeanR':>8} {'Click%':>7} "
              f"{'Purch%':>7} {'Expl%':>6} {'ActEnt':>6} {'#Acts':>5} "
              f"{'NewCat':>7}")
    print(header)
    print("-" * 120)

    # Sort by total reward
    sorted_results = sorted(results, key=lambda r: r.total_reward, reverse=True)
    for r in sorted_results:
        print(f"{r.config.label:<40} "
              f"{r.total_reward:>8.1f} {r.mean_step_reward:>8.4f} "
              f"{r.click_rate:>6.3f} {r.purchase_rate:>6.3f} "
              f"{r.exploration_action_rate:>5.3f} {r.action_entropy:>5.3f} "
              f"{r.unique_actions_used:>5d} {r.unique_categories_discovered:>6.2f}")

    print("-" * 120)

    # --- Key comparisons ---
    pure_exploit = [r for r in results if r.config.label == "Pure Exploit (ε=0)"]
    adaptive_best = [r for r in results if "Adaptive" in r.config.label]
    if pure_exploit and adaptive_best:
        pe = pure_exploit[0]
        # Find best adaptive by total reward
        best_adaptive = max(adaptive_best, key=lambda r: r.total_reward)

        print("\n=== KEY INSIGHT (idea.md §4.2) ===")
        print(f"Pure Exploit (ε=0):   TotalR={pe.total_reward:.1f}, "
              f"Click={pe.click_rate:.3f}, Purch={pe.purchase_rate:.3f}, "
              f"NewCat={pe.unique_categories_discovered:.2f}")
        print(f"Best Adaptive:         TotalR={best_adaptive.total_reward:.1f}, "
              f"Click={best_adaptive.click_rate:.3f}, Purch={best_adaptive.purchase_rate:.3f}, "
              f"NewCat={best_adaptive.unique_categories_discovered:.2f}")

        if best_adaptive.total_reward > pe.total_reward:
            delta_reward = best_adaptive.total_reward - pe.total_reward
            delta_cats = (best_adaptive.unique_categories_discovered -
                         pe.unique_categories_discovered)
            print(f"\n  → Exploration gives +{delta_reward:.1f} total reward "
                  f"({delta_reward/pe.total_reward*100:.1f}% improvement)")
            print(f"  → +{delta_cats:.1f} new categories discovered per user")
            if best_adaptive.click_rate < pe.click_rate:
                print(f"  → Short-term click rate {best_adaptive.click_rate:.3f} < "
                      f"{pe.click_rate:.3f} (pure exploit)")
                print(f"     But long-term cumulative reward is HIGHER — "
                      f"the core insight!")
            else:
                print(f"  → Both short-term and long-term metrics improved")


def plot_tradeoff_curves(results: List[SweepResult], output_dir: str):
    """Generate matplotlib visualizations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- Plot 1: Epsilon vs Cumulative Reward ---
    fixed_eps = [r for r in results if "Fixed ε=" in r.config.label
                 and "Bonus" not in r.config.label]
    if fixed_eps:
        epsilons = [r.config.epsilon_base for r in fixed_eps]
        rewards = [r.total_reward for r in fixed_eps]
        clicks = [r.click_rate for r in fixed_eps]
        purchases = [r.purchase_rate for r in fixed_eps]
        new_cats = [r.unique_categories_discovered for r in fixed_eps]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Reward
        axes[0].plot(epsilons, rewards, "o-", color="tab:blue", markersize=8)
        axes[0].set_xlabel("Epsilon")
        axes[0].set_ylabel("Total Cumulative Reward")
        axes[0].set_title("Exploration Rate vs Total Reward")
        axes[0].grid(True, alpha=0.3)

        # Click rate
        axes[1].plot(epsilons, clicks, "s-", color="tab:green", markersize=8)
        axes[1].set_xlabel("Epsilon")
        axes[1].set_ylabel("Click Rate")
        axes[1].set_title("Exploration Rate vs Short-term Click Rate")
        axes[1].grid(True, alpha=0.3)

        # New categories
        ax2 = axes[2]
        ax2.plot(epsilons, new_cats, "D-", color="tab:orange", markersize=8)
        ax2.set_xlabel("Epsilon")
        ax2.set_ylabel("New Categories / User", color="tab:orange")
        ax2.set_title("Exploration Rate vs Category Discovery")
        ax2.grid(True, alpha=0.3)

        # Also plot purchase rate on twin axis for the last plot
        ax2b = ax2.twinx()
        ax2b.plot(epsilons, purchases, "^-", color="tab:red", markersize=8)
        ax2b.set_ylabel("Purchase Rate", color="tab:red")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "epsilon_vs_reward.svg"))
        plt.close()
        print(f"  Saved epsilon_vs_reward.svg")

    # --- Plot 2: Adaptive vs Fixed epsilon comparison ---
    adaptive = [r for r in results if "Adaptive" in r.config.label]
    if adaptive and fixed_eps:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Fixed epsilon points
        ax.scatter([r.click_rate for r in fixed_eps],
                   [r.total_reward for r in fixed_eps],
                   c=[r.config.epsilon_base for r in fixed_eps],
                   cmap="Blues", s=100, marker="o", label="Fixed ε",
                   edgecolors="black", linewidth=0.5)

        # Adaptive epsilon points
        ax.scatter([r.click_rate for r in adaptive],
                   [r.total_reward for r in adaptive],
                   c=[r.config.epsilon_max for r in adaptive],
                   cmap="Reds", s=100, marker="^", label="Adaptive ε",
                   edgecolors="black", linewidth=0.5)

        ax.set_xlabel("Click Rate (short-term proxy)")
        ax.set_ylabel("Total Cumulative Reward (long-term proxy)")
        ax.set_title("Exploration-Exploitation Tradeoff Frontier\n"
                     "Lower-left = explore; Upper-right = exploit")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ee_tradeoff_frontier.svg"))
        plt.close()
        print(f"  Saved ee_tradeoff_frontier.svg")

    # --- Plot 3: Action distribution heatmap ---
    top_results = sorted(results, key=lambda r: r.total_reward, reverse=True)[:10]
    if top_results:
        all_action_names = sorted(set(
            name for r in top_results for name in r.action_counts.keys()
        ))
        n_configs = len(top_results)
        n_actions = len(all_action_names)

        matrix = np.zeros((n_configs, n_actions))
        for i, r in enumerate(top_results):
            total = sum(r.action_counts.values())
            for j, name in enumerate(all_action_names):
                matrix[i, j] = r.action_counts.get(name, 0) / total

        fig, ax = plt.subplots(figsize=(14, max(6, n_configs * 0.6)))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.5)

        ax.set_xticks(range(n_actions))
        ax.set_xticklabels(all_action_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_configs))
        ax.set_yticklabels([r.config.label[:45] for r in top_results], fontsize=8)
        ax.set_title("Action Distribution (Top 10 Configs by Total Reward)")

        plt.colorbar(im, ax=ax, label="Action Probability")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "action_distribution.svg"))
        plt.close()
        print(f"  Saved action_distribution.svg")

    # --- Plot 4: Per-step reward curves ---
    fig, ax = plt.subplots(figsize=(12, 6))
    key_configs = (
        [r for r in results if r.config.label == "Pure Exploit (ε=0)"] +
        [r for r in results if "Adaptive ε (base=0.05, max=0.5)" in r.config.label] +
        [r for r in results if "Fixed ε=0.2" in r.config.label]
    )
    for r in key_configs:
        if r.per_step_rewards:
            ax.plot(r.per_step_rewards, label=r.config.label, linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Per-Step Reward: Exploration vs Exploitation Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_step_reward.svg"))
    plt.close()
    print(f"  Saved per_step_reward.svg")


# ===========================================================================
# 9. Main Entry Point
# ===========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Online RL simulation for exploration-exploitation tradeoff")
    parser.add_argument("--n_users", type=int, default=50,
                       help="Number of simulated users")
    parser.add_argument("--n_items", type=int, default=2000,
                       help="Number of simulated items")
    parser.add_argument("--max_steps", type=int, default=50,
                       help="Steps per episode")
    parser.add_argument("--output_dir", type=str, default="outputs/ee_tradeoff",
                       help="Output directory for results and plots")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: fewer configs, fewer users")
    args = parser.parse_args()

    if args.quick:
        args.n_users = 20
        configs = build_sweep_configs()[:6]  # first 6 configs only
    else:
        configs = build_sweep_configs()

    print(f"Running exploration-exploitation tradeoff simulation...")
    print(f"  Users: {args.n_users}, Items: {args.n_items}, Steps: {args.max_steps}")
    print(f"  Configurations: {len(configs)}")
    print(f"  Device: {args.device}")

    t0 = time.time()

    experiment = EETradeoffExperiment(
        n_users=args.n_users,
        n_items=args.n_items,
        max_steps_per_episode=args.max_steps,
        device=args.device,
        seed=args.seed,
    )

    results = experiment.sweep(configs)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/len(configs):.1f}s per config)")

    # --- Output ---
    os.makedirs(args.output_dir, exist_ok=True)

    # Text summary
    print_tradeoff_summary(results)

    # Save JSON
    json_results = []
    for r in results:
        json_results.append({
            "config": {
                "label": r.config.label,
                "epsilon_base": r.config.epsilon_base,
                "epsilon_max": r.config.epsilon_max,
                "use_uncertainty_adaptive": r.config.use_uncertainty_adaptive,
                "exploration_bonus_weight": r.config.exploration_bonus_weight,
            },
            "total_reward": r.total_reward,
            "mean_step_reward": r.mean_step_reward,
            "click_rate": r.click_rate,
            "purchase_rate": r.purchase_rate,
            "exploration_action_rate": r.exploration_action_rate,
            "action_entropy": r.action_entropy,
            "unique_actions_used": r.unique_actions_used,
            "unique_categories_discovered": r.unique_categories_discovered,
            "action_counts": r.action_counts,
        })

    json_path = os.path.join(args.output_dir, "ee_tradeoff_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Plots
    try:
        plot_tradeoff_curves(results, args.output_dir)
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
