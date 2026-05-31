"""
Quick smoke test for RL + Memory experiment pipeline.

Verifies:
  1. P5 tokenizer and model loading
  2. Data bridge: P5 data → RL format
  3. RL+Memory model initialization
  4. Memory system (encode → store → retrieve)
  5. Forward pass: state → policy → action → candidates
  6. Comparison evaluation (P5 beam search vs RL strategy ranking)

Does NOT do full training — just validates the pipeline is correct.
"""
import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(Path(_PROJECT_ROOT) / "reproduce"))


def test_tokenizer():
    print("=" * 50)
    print("Test 1: Tokenizer")
    print("=" * 50)
    from tokenization import P5Tokenizer
    tok = P5Tokenizer.from_pretrained('t5-small', max_length=512, do_lower_case=True)
    print(f"  vocab_size: {tok.vocab_size}")
    enc = tok.encode('hello world user_123 item_456')
    print(f"  encode('hello world'): {enc[:10]}...")
    dec = tok.decode(enc[:5])
    print(f"  decode: {dec}")
    print("  PASSED\n")
    return tok


def test_model_loading():
    print("=" * 50)
    print("Test 2: P5 Model Loading")
    print("=" * 50)
    from modeling_p5 import P5
    from transformers import T5Config

    config = T5Config.from_pretrained('t5-small')
    config.losses = 'rating,sequential,explanation,review,traditional'

    model = P5.from_pretrained('t5-small', config=config)
    if hasattr(model.encoder, 'whole_word_embeddings'):
        model.encoder.whole_word_embeddings.weight.data.normal_(mean=0.0, std=1.0)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model params: {params:.2f}M")
    print(f"  Encoder type: {type(model.encoder).__name__}")
    print("  PASSED\n")
    return model


def test_data_bridge():
    print("=" * 50)
    print("Test 3: Data Bridge")
    print("=" * 50)
    from src.rl_experiment import SequentialDataBridge

    bridge = SequentialDataBridge('data', 'beauty')
    print(f"  Users: {len(bridge.user_sequences)}")
    print(f"  Items: {len(bridge.all_items)}")

    # Get a tiny train/test split for testing
    train_data = bridge.get_train_data(sample_ratio=0.01)  # 1% = ~600 samples
    test_data = bridge.get_test_data(max_users=100)
    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")
    print("  PASSED\n")
    return bridge, train_data, test_data


def test_memory_system():
    print("=" * 50)
    print("Test 4: Memory System")
    print("=" * 50)
    from src.rl_experiment import get_3060_config
    from src.memory import MemoryManager, MemoryEncoder

    config = get_3060_config(0.05)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = MemoryEncoder(
        num_users=50000, num_items=50000,
        num_actions=4, embed_dim=128,
        memory_dim=config.memory.memory_dim,
    ).to(device)

    manager = MemoryManager(
        memory_dim=config.memory.memory_dim,
        short_term_capacity=config.memory.short_term_capacity,
        top_k=config.memory.top_k_retrieval,
        time_decay_lambda=config.memory.time_decay_lambda,
    )
    manager.set_encoder(encoder)

    # Add some test interactions
    import time
    for i in range(10):
        manager.add_interaction(
            user_id=1, item_id=i + 100,
            action_type=i % 4, rating=0.5 + (i % 5) / 10.0,
            timestamp=time.time() - (10 - i) * 86400,
        )

    print(f"  Short-term memory: {len(manager.short_term)} entries")
    print(f"  Long-term memory: {len(manager.long_term)} entries")

    # Retrieve
    query = np.random.randn(config.memory.memory_dim).astype(np.float32)
    query = query / np.linalg.norm(query)
    vecs, entries, conf = manager.retrieve(query, user_id=1)
    print(f"  Retrieved: {len(vecs)} vectors, confidence={conf:.3f}")

    # State vector
    state = manager.get_state_vector(user_id=1, user_embedding=query)
    print(f"  State keys: {list(state.keys())}")
    print(f"  Short-term ctx shape: {state['short_term_context'].shape}")
    print(f"  Long-term ctx shape: {state['long_term_context'].shape}")
    print(f"  Confidence: {state['retrieval_confidence']:.3f}")
    print("  PASSED\n")
    return manager, encoder


def test_rl_policy():
    print("=" * 50)
    print("Test 5: RL Policy Network")
    print("=" * 50)
    from src.rl.policy import POMDPPolicy, UncertaintyAdaptiveEpsilon
    from src.rl.actions import NUM_ACTIONS, ACTION_REGISTRY

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy = POMDPPolicy(
        user_dim=512,
        memory_dim=128,
        num_actions=NUM_ACTIONS,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    # Dummy inputs
    B = 4
    ue = torch.randn(B, 512).to(device)
    mc = torch.randn(B, 256).to(device)  # 2 * memory_dim
    cf = torch.rand(B, 1).to(device)
    pf = torch.zeros(B, 10).to(device)

    # Forward
    q_values = policy(ue, mc, cf, pf)
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Num actions: {NUM_ACTIONS}")

    # Action selection
    actions = policy.get_action(ue, mc, cf, pf, epsilon=0.1)
    print(f"  Selected actions: {actions.tolist()}")

    # Action names
    for aid in actions[:4]:
        print(f"    {aid}: {ACTION_REGISTRY[aid.item()].name}")

    # Epsilon scheduler
    eps = UncertaintyAdaptiveEpsilon(epsilon_base=0.05, epsilon_max=0.5)
    e_low = eps.get_epsilon(0.9)  # high confidence → low epsilon
    e_high = eps.get_epsilon(0.1)  # low confidence → high epsilon
    print(f"  Epsilon (conf=0.9): {e_low:.3f}")
    print(f"  Epsilon (conf=0.1): {e_high:.3f}")
    print("  PASSED\n")
    return policy


def test_candidate_retrieval(bridge):
    print("=" * 50)
    print("Test 6: Candidate Retrieval")
    print("=" * 50)
    from src.rl_experiment import CandidateRetriever
    from src.rl.actions import action_to_candidate_strategy

    # Build retriever with a small item set
    item_embs = {}
    for i, item_id in enumerate(bridge.all_items[:1000]):
        item_embs[int(item_id)] = np.random.randn(512).astype(np.float32)

    retriever = CandidateRetriever(
        item_embs=item_embs,
        user_sequences=bridge.user_sequences,
        topk=20,
    )
    print(f"  Item matrix: {retriever.item_matrix.shape}")

    # Test different strategies
    user_emb = np.random.randn(512).astype(np.float32)
    for strategy_name in ['similar_to_last', 'diverse_categories']:
        strategy = {"method": strategy_name, "topk": 10}
        candidates = retriever.retrieve("user_1", user_emb, strategy)
        print(f"  {strategy_name}: {len(candidates)} candidates, "
              f"first 3: {candidates[:3]}")

    # Test action → strategy → candidates
    for aid in [0, 6, 11]:  # exploit, explore, periodic
        strategy = action_to_candidate_strategy(aid)
        candidates = retriever.retrieve("user_1", user_emb, strategy)
        print(f"  Action {aid} ({strategy['method']}): "
              f"{len(candidates)} candidates")

    print("  PASSED\n")
    return retriever


def test_rl_memory_model(bridge, p5_model):
    print("=" * 50)
    print("Test 7: RL+Memory Model (Full Pipeline)")
    print("=" * 50)
    from src.rl_experiment import get_3060_config
    from src.rl_experiment import RLMemoryRecommender, CandidateRetriever

    config = get_3060_config(0.05)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p5_model = p5_model.to(device)
    p5_model.eval()

    # Build item embeddings
    item_embs = {}
    for i, item_id in enumerate(bridge.all_items[:1000]):
        with torch.no_grad():
            iid = int(item_id) % p5_model.shared.num_embeddings
            emb = p5_model.shared(torch.tensor([iid], device=device))
            item_embs[int(item_id)] = emb[0].cpu().numpy().astype(np.float32)

    retriever = CandidateRetriever(
        item_embs=item_embs,
        user_sequences=bridge.user_sequences,
        topk=20,
    )

    # Build RL+Memory model
    model = RLMemoryRecommender(
        p5_model=p5_model,
        config=config,
        user_sequences=bridge.user_sequences,
        device=device,
    )
    model.set_retriever(retriever)

    extra_params = sum(p.numel() for m in [model.memory_encoder, model.policy]
                        for p in m.parameters())
    print(f"  P5 backbone: 60.75M params")
    print(f"  RL+Memory extra: {extra_params/1e6:.2f}M params")
    print(f"  Total: {(60.75 + extra_params/1e6):.2f}M params")

    # Test forward pass with a real user
    users = list(bridge.user_sequences.keys())
    if users:
        test_user = users[0]
        hist = bridge.user_sequences[test_user]
        print(f"  Test user: {test_user}, history len: {len(hist)}")

        # Add some memory
        for iid in hist[:10]:
            model.add_to_memory(test_user, iid, action_type=1, rating=0.7)

        # Forward pass
        result = model.forward(test_user, deterministic=True, topk=20)
        print(f"  Action: {result['action_name']} (#{result['action_id']})")
        print(f"  Candidates: {len(result['candidates'])} items")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Epsilon: {result['epsilon']:.3f}")

        # Action distribution
        dist = model.get_action_distribution(test_user)
        top_actions = np.argsort(dist)[-5:][::-1]
        print(f"  Top-5 actions:")
        for aid in top_actions:
            from src.rl.actions import ACTION_REGISTRY
            print(f"    {ACTION_REGISTRY[aid].name}: {dist[aid]:.3f}")

    print("  PASSED\n")
    return model


def test_p5_eval(bridge, p5_model, tokenizer):
    print("=" * 50)
    print("Test 8: P5 Baseline Evaluation (5 samples)")
    print("=" * 50)
    from src.rl_experiment import evaluate_p5_baseline

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p5_model = p5_model.to(device)
    p5_model.eval()

    test_samples = bridge.get_test_data(max_users=5)
    results = evaluate_p5_baseline(
        p5_model, tokenizer, test_samples,
        bridge.all_items, device,
        beam_size=5, max_eval=5,
    )
    print(f"  Results: {results}")
    print("  PASSED\n")
    return results


def test_rl_eval(bridge, rl_model):
    print("=" * 50)
    print("Test 9: RL+Memory Evaluation (5 samples)")
    print("=" * 50)
    from src.rl_experiment import evaluate_rl_memory

    test_samples = bridge.get_test_data(max_users=5)
    results = evaluate_rl_memory(rl_model, test_samples, max_eval=5)
    print(f"  Results: {results}")
    print("  PASSED\n")
    return results


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("\n" + "#" * 60)
    print("#  RL + Memory Experiment — SMOKE TEST")
    print("#" * 60 + "\n")

    # Run all tests
    tokenizer = test_tokenizer()
    p5_model = test_model_loading()
    bridge, train_data, test_data = test_data_bridge()
    memory, encoder = test_memory_system()
    policy = test_rl_policy()
    retriever = test_candidate_retrieval(bridge)
    rl_model = test_rl_memory_model(bridge, p5_model)
    p5_results = test_p5_eval(bridge, p5_model, tokenizer)
    rl_results = test_rl_eval(bridge, rl_model)

    # Final comparison
    from src.rl_experiment import print_comparison
    print_comparison(p5_results, rl_results)

    print("\n" + "#" * 60)
    print("#  ALL TESTS PASSED — Pipeline is working!")
    print("#" * 60)


if __name__ == '__main__':
    main()
