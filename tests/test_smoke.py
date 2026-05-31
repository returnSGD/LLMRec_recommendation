"""
Smoke test for the RL + Memory recommendation system.

Run: python -m pytest tests/test_smoke.py -v
or just: python tests/test_smoke.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch


def test_config():
    """Test config construction."""
    from src.config import Config, MemoryConfig, RLConfig, TrainingConfig, RewardWeights

    config = Config()
    assert config.memory.memory_dim == 256
    assert config.rl.num_actions == 16
    assert config.training.backbone == "t5-small"
    assert config.reward.click == 1.0
    print("[PASS] Config construction")


def test_memory_encoder():
    """Test that memory encoder compresses interactions into vectors."""
    from src.memory import MemoryEncoder

    encoder = MemoryEncoder(
        num_users=1000, num_items=5000,
        embed_dim=64, memory_dim=256,
    )

    B = 8
    user_ids = torch.randint(1, 1000, (B,))
    item_ids = torch.randint(1, 5000, (B,))
    action_types = torch.randint(0, 4, (B,))
    ratings = torch.rand(B)
    time_deltas = torch.rand(B)

    vecs = encoder(user_ids, item_ids, action_types, ratings, time_deltas)
    assert vecs.shape == (B, 256)
    # L2 normalized
    norms = vecs.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5)
    print("[PASS] Memory encoder: (8, 256) with L2 normalization")


def test_faiss_store():
    """Test FAISS vector store add + search + time decay."""
    from src.memory import FAISSVectorStore

    dim = 64
    store = FAISSVectorStore(dim=dim, index_type="Flat", time_decay_lambda=0.01)

    # Add 100 entries
    N = 100
    vecs = np.random.randn(N, dim).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    user_ids = np.random.randint(1, 10, N).tolist()
    item_ids = np.random.randint(1, 100, N).tolist()
    action_types = np.random.randint(0, 4, N).tolist()
    now = time.time()
    timestamps = [now - np.random.randint(0, 365) * 86400 for _ in range(N)]

    ids = store.add(vecs, user_ids, item_ids, action_types, timestamps)
    assert len(ids) == N
    assert len(store) == N

    # Search
    query = np.random.randn(dim).astype(np.float32)
    query = query / np.linalg.norm(query)
    distances, indices, confidences = store.search(query, k=10, current_time=now)

    assert len(distances) <= 10
    assert len(confidences) <= 10
    assert 0 <= float(np.mean(confidences)) <= 1.0
    print(f"[PASS] FAISS store: {N} entries, top-10 retrieval, "
          f"mean confidence={np.mean(confidences):.3f}")

    # User-filtered search
    u_d, u_i, u_c = store.search_by_user(
        user_ids[0], query, k=5, current_time=now,
    )
    assert len(u_d) <= 5
    print(f"[PASS] User-filtered search: {len(u_d)} results")


def test_memory_manager():
    """Test two-level memory manager."""
    from src.memory import MemoryEncoder, MemoryManager

    encoder = MemoryEncoder(num_users=1000, num_items=5000,
                            embed_dim=64, memory_dim=128)
    manager = MemoryManager(memory_dim=128, short_term_capacity=20, top_k=10)
    manager.set_encoder(encoder)

    # Add interactions
    user_id = 42
    now = time.time()
    for i in range(30):
        manager.add_interaction(
            user_id=user_id,
            item_id=np.random.randint(1, 5000),
            action_type=np.random.randint(0, 4),
            rating=np.random.uniform(0, 1),
            timestamp=now - (30 - i) * 86400,
            category_id=np.random.randint(0, 50),
        )

    assert len(manager.short_term) == 20  # capped
    assert len(manager.long_term) == 30   # all stored in long-term

    # Retrieve
    user_emb = np.random.randn(128).astype(np.float32)
    user_emb = user_emb / np.linalg.norm(user_emb)

    state = manager.get_state_vector(user_id=user_id, user_embedding=user_emb,
                                     current_time=now)

    assert "short_term_context" in state
    assert "long_term_context" in state
    assert "retrieval_confidence" in state
    assert 0 <= state["retrieval_confidence"] <= 1.0
    assert state["short_term_context"].shape == (128,)
    assert state["long_term_context"].shape == (128,)

    print(f"[PASS] Memory manager: ST={len(manager.short_term)}, "
          f"LT={len(manager.long_term)}, confidence={state['retrieval_confidence']:.3f}")


def test_action_space():
    """Test action definitions and masking."""
    from src.rl.actions import (ACTION_REGISTRY, NUM_ACTIONS, get_action_mask,
                                 action_to_candidate_strategy)

    assert NUM_ACTIONS == 16
    assert len(ACTION_REGISTRY) == 16

    # Check all actions have names and IDs
    for aid, action in ACTION_REGISTRY.items():
        assert action.id == aid
        assert isinstance(action.name, str)
        assert isinstance(action.description, str)

    # Action mask
    mask = get_action_mask(
        user_history={"has_abandoned": False, "unique_brands": 1},
        purchased_items=set(),
        available_categories={1, 2, 3},
        has_periodic_patterns=False,
        is_new_user=False,
    )
    assert mask.shape == (16,)
    # Periodic actions should be masked when no patterns
    from src.rl.actions import PERIODIC_ACTIONS
    for aid in PERIODIC_ACTIONS:
        assert mask[aid] == 0.0, f"Periodic action {aid} should be masked"

    # Strategy mapping
    for aid in range(NUM_ACTIONS):
        strategy = action_to_candidate_strategy(aid)
        assert "method" in strategy

    print(f"[PASS] Action space: {NUM_ACTIONS} actions defined")


def test_reward_function():
    """Test multi-component reward computation."""
    from src.rl.reward import RewardFunction, RewardComponents

    rf = RewardFunction()

    # Full positive feedback
    r = rf.compute(clicked=True, purchased=True, dwell_time=60.0)
    assert r.click == 1.0
    assert r.purchase == 3.0
    assert abs(r.dwell - 0.6) < 1e-6
    assert r.total > 3.0

    # Skip (negative feedback): w["negative"] * (-neg_map["skip"])
    # = -0.5 * (-(-0.3)) = -0.5 * 0.3 = -0.15
    r = rf.compute(clicked=False, negative_signal="skip")
    assert r.click == 0.0
    assert abs(r.negative - (-0.15)) < 1e-6
    assert r.total < 0

    print(f"[PASS] Reward function: click+pur={rf.compute(clicked=True, purchased=True).total:.1f}, "
          f"skip={rf.compute(negative_signal='skip').total:.2f}")


def test_policy_network():
    """Test policy network forward and action selection."""
    from src.rl.policy import POMDPPolicy, UncertaintyAdaptiveEpsilon

    policy = POMDPPolicy(user_dim=128, memory_dim=128, num_actions=16, hidden_dim=128)

    B = 4
    user_emb = torch.randn(B, 128)
    memory_ctx = torch.randn(B, 256)  # 2*128
    confidence = torch.rand(B, 1)
    periodic = torch.rand(B, 10)
    action_mask = torch.ones(B, 16)

    # Q-values
    q = policy(user_emb, memory_ctx, confidence, periodic, action_mask)
    assert q.shape == (B, 16)

    # Action selection
    actions = policy.get_action(user_emb, memory_ctx, confidence, periodic,
                                action_mask, epsilon=0.0)
    assert actions.shape == (B,)
    assert (0 <= actions).all() and (actions < 16).all()

    # With exploration
    actions_r = policy.get_action(user_emb, memory_ctx, confidence, periodic,
                                  action_mask, epsilon=0.5)
    assert actions_r.shape == (B,)

    print(f"[PASS] Policy network: Q shape {q.shape}, greedy action={actions[0].item()}")


def test_uncertainty_adaptive_epsilon():
    """Test adaptive epsilon computation."""
    from src.rl.policy import UncertaintyAdaptiveEpsilon

    scheduler = UncertaintyAdaptiveEpsilon(
        epsilon_base=0.05, epsilon_max=0.5, uncertainty_scale=1.0,
    )

    eps_high_uncertainty = scheduler.get_epsilon(0.1)   # low confidence → high ε
    eps_low_uncertainty = scheduler.get_epsilon(0.9)    # high confidence → low ε

    assert eps_high_uncertainty > eps_low_uncertainty
    assert 0.01 <= eps_high_uncertainty <= 0.9
    assert 0.01 <= eps_low_uncertainty <= 0.9

    print(f"[PASS] Adaptive epsilon: high_unc→ε={eps_high_uncertainty:.3f}, "
          f"low_unc→ε={eps_low_uncertainty:.3f}")


def test_replay_buffer():
    """Test replay buffer add + sample."""
    from src.rl.buffer import ReplayBuffer

    buffer = ReplayBuffer(capacity=1000)

    for i in range(500):
        state = {
            "user_emb": np.random.randn(128).astype(np.float32),
            "memory_context": np.random.randn(256).astype(np.float32),
            "retrieval_confidence": np.array([0.7], dtype=np.float32),
            "periodic_flags": np.zeros(10, dtype=np.float32),
            "action_mask": np.ones(16, dtype=np.float32),
        }
        action = np.random.randint(0, 16)
        reward = np.random.randn()
        next_state = {
            "user_emb": np.random.randn(128).astype(np.float32),
            "memory_context": np.random.randn(256).astype(np.float32),
            "retrieval_confidence": np.array([0.8], dtype=np.float32),
            "periodic_flags": np.zeros(10, dtype=np.float32),
            "action_mask": np.ones(16, dtype=np.float32),
        }
        buffer.add(state, action, reward, next_state, done=False)

    assert len(buffer) == 500

    batch = buffer.sample(32)
    for key in ["user_emb", "actions", "rewards", "dones",
                "next_user_emb", "next_memory_context"]:
        assert key in batch, f"Missing key: {key}"
    assert batch["actions"].shape == (32,)
    assert batch["rewards"].shape == (32,)
    print(f"[PASS] Replay buffer: {len(buffer)} transitions, batch size 32")


def test_cql_trainer():
    """Test CQL trainer single step."""
    from src.rl import ReplayBuffer, CQLTrainer
    from src.rl.policy import POMDPPolicy
    from src.rl.iql import ValueNetwork  # needed to approximate memory_dim
    import copy

    user_dim, mem_dim, n_act = 64, 64, 16
    policy = POMDPPolicy(user_dim=user_dim, memory_dim=mem_dim,
                         num_actions=n_act, hidden_dim=64)
    target = POMDPPolicy(user_dim=user_dim, memory_dim=mem_dim,
                         num_actions=n_act, hidden_dim=64)
    target.load_state_dict(policy.state_dict())

    buffer = ReplayBuffer(capacity=1000)
    state_keys = ["user_emb", "memory_context", "retrieval_confidence",
                  "periodic_flags", "action_mask"]
    for i in range(500):
        s = {
            "user_emb": np.random.randn(user_dim).astype(np.float32),
            "memory_context": np.random.randn(2*mem_dim).astype(np.float32),
            "retrieval_confidence": np.array([0.7], dtype=np.float32),
            "periodic_flags": np.zeros(10, dtype=np.float32),
            "action_mask": np.ones(n_act, dtype=np.float32),
        }
        ns = {k: v.copy() for k, v in s.items()}
        buffer.add(s, np.random.randint(0, n_act), np.random.randn(), ns, False)

    trainer = CQLTrainer(policy, target, buffer, device="cpu",
                         gamma=0.99, cql_alpha=0.1)

    batch = buffer.sample(32, "cpu")
    metrics = trainer.train_batch(batch)
    assert "loss/total" in metrics
    print(f"[PASS] CQL trainer: loss={metrics['loss/total']:.4f}")


def test_iql_trainer():
    """Test IQL trainer single step."""
    from src.rl import ReplayBuffer, IQLTrainer, ValueNetwork
    from src.rl.policy import POMDPPolicy

    user_dim, mem_dim, n_act = 64, 64, 16
    q_net = POMDPPolicy(user_dim=user_dim, memory_dim=mem_dim,
                         num_actions=n_act, hidden_dim=64)
    v_net = ValueNetwork(user_dim=user_dim, memory_dim=mem_dim, hidden_dim=64)

    buffer = ReplayBuffer(capacity=1000)
    for i in range(500):
        s = {
            "user_emb": np.random.randn(user_dim).astype(np.float32),
            "memory_context": np.random.randn(2*mem_dim).astype(np.float32),
            "retrieval_confidence": np.array([0.7], dtype=np.float32),
            "periodic_flags": np.zeros(10, dtype=np.float32),
            "action_mask": np.ones(n_act, dtype=np.float32),
        }
        ns = {k: v.copy() for k, v in s.items()}
        buffer.add(s, np.random.randint(0, n_act), np.random.randn(), ns, False)

    trainer = IQLTrainer(q_net, v_net, buffer, device="cpu")

    batch = buffer.sample(32, "cpu")
    metrics = trainer.train_batch(batch)
    assert "loss/q" in metrics
    print(f"[PASS] IQL trainer: q_loss={metrics['loss/q']:.4f}, "
          f"v_loss={metrics['loss/v']:.4f}")


def test_end_to_end():
    """End-to-end test: model.build → memory → policy → action → feedback."""
    from src.config import Config
    from src.model import MemoryRLModel

    config = Config()
    config.memory.memory_dim = 64  # small for testing
    config.rl.hidden_dim = 64

    model = MemoryRLModel(config, p5_model=None)

    # Encode a user (no P5 model → user_dim=memory_dim=64)
    user_emb = model.encode_user(user_interaction_ids=list(range(1, 11)),
                                 user_id=1)
    assert user_emb.shape == (64,)

    # Add some interactions to memory
    now = time.time()
    for i in range(15):
        model.add_interaction(
            user_id=1, item_id=np.random.randint(1, 5000),
            action_type=np.random.randint(0, 4),
            rating=np.random.uniform(0, 1),
            category_id=np.random.randint(0, 50),
        )

    # Make a recommendation decision
    result = model.forward(user_emb, user_id=1, deterministic=True)
    assert "action_id" in result
    assert "strategy" in result
    assert 0 <= result["action_id"] < 16

    print(f"[PASS] End-to-end: action={result['action_name']}, "
          f"confidence={result['confidence']:.3f}, "
          f"strategy={result['strategy']['method']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Smoke Test: RL + Memory Recommendation System")
    print("=" * 60)

    tests = [
        test_config,
        test_memory_encoder,
        test_faiss_store,
        test_memory_manager,
        test_action_space,
        test_reward_function,
        test_policy_network,
        test_uncertainty_adaptive_epsilon,
        test_replay_buffer,
        test_cql_trainer,
        test_iql_trainer,
        test_end_to_end,
    ]

    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    print(f"{'=' * 60}")
