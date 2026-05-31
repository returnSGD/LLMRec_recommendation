"""
Global configuration for the RL + Memory recommendation system.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MemoryConfig:
    """Two-level memory system configuration."""

    # Short-term memory
    short_term_capacity: int = 50  # max recent interactions to keep

    # Long-term memory (FAISS)
    memory_dim: int = 256  # compressed vector dimension
    faiss_index_type: str = "IVFFlat"  # or "Flat" for exact search
    faiss_nlist: int = 100  # number of clusters for IVF
    top_k_retrieval: int = 20  # number of memory fragments to retrieve

    # Time decay
    time_decay_lambda: float = 0.01  # decay rate (per day), exp(-λ·Δt)

    # Memory update
    aggregation: str = "weighted_avg"  # how to merge old + new memory

    # Uncertainty estimation
    uncertainty_method: str = "entropy"  # entropy of similarity distribution


@dataclass
class RLConfig:
    """RL decision layer configuration."""

    # POMDP state
    state_dim: int = 512  # user_emb(256) + memory_emb(256)
    hidden_dim: int = 256

    # Action space
    num_actions: int = 16  # discrete macro-actions

    # Policy network
    policy_arch: str = "mlp"  # or "transformer"
    num_layers: int = 3
    dropout: float = 0.1

    # Exploration
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    use_uncertainty_adaptive_epsilon: bool = True  # auxiliary contribution 3
    entropy_coef: float = 0.01  # entropy regularization

    # Training
    algorithm: str = "cql"  # "cql" or "iql"
    gamma: float = 0.99  # discount factor
    tau: float = 0.005  # target network soft-update rate
    lr: float = 3e-4
    batch_size: int = 256
    grad_clip: float = 1.0

    # CQL specific
    cql_alpha: float = 1.0  # conservative penalty weight
    cql_num_samples: int = 10  # actions sampled for conservative penalty

    # IQL specific
    iql_expectile: float = 0.7  # expectile for value (0.5=mean, 1.0=max)
    iql_temperature: float = 3.0  # temperature for advantage-weighted regression

    # Negative feedback
    use_negative_feedback: bool = True

    # Action masking
    use_action_masking: bool = True


@dataclass
class RewardWeights:
    """Multi-component reward function weights."""

    click: float = 1.0
    purchase: float = 2.0
    dwell_time: float = 0.1  # per 10 seconds
    diversity_bonus: float = 0.1  # when exploring new categories
    exploration_bonus: float = 0.05  # for explore_* actions
    retention_signal: float = 5.0  # long-term retention (sparse)
    negative_penalty: float = -0.5  # skip/return/bad rating
    periodic_recall_bonus: float = 1.5  # successfully recalling periodic needs


@dataclass
class TrainingConfig:
    """Overall training configuration."""

    # Backbone
    backbone: str = "t5-small"  # shared with P5 for fair comparison

    # Data
    dataset: str = "beauty"
    data_dir: str = "data"

    # Training
    total_steps: int = 50_000
    eval_every: int = 2_000
    save_every: int = 5_000
    seed: int = 42

    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True

    # Output
    output_dir: str = "outputs/memory_rl"
    log_dir: str = "outputs/logs"


@dataclass
class Config:
    """Master configuration."""
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            memory=MemoryConfig(**d.get("memory", {})),
            rl=RLConfig(**d.get("rl", {})),
            reward=RewardWeights(**d.get("reward", {})),
            training=TrainingConfig(**d.get("training", {})),
        )
