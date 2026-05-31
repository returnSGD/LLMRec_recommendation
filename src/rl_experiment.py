"""
RL + Memory Hybrid Architecture — Comparison Experiment vs P5 Baseline.

Architecture:
  P5 Encoder → user/item embeddings
       ├── Memory System (short-term buffer + long-term FAISS store + time decay)
       └── RL Policy (POMDP → 16 discrete macro-actions → candidate retrieval)

Comparison:
  P5 Baseline:  Beam search (B=20), all-item ranking
  RL + Memory:  1-pass policy decision → strategy-driven candidate ranking

Usage (local RTX 3060 quick test):
  python -m src.rl_experiment --dataset beauty --backbone t5-small \
      --p5_checkpoint reproduce/output/xxx/Epoch01.pth \
      --sample_ratio 0.05 --epochs 1 --batch_size 4

Usage (server full run):
  python -m src.rl_experiment --dataset beauty --backbone t5-small \
      --p5_checkpoint reproduce/output/xxx/BEST_EVAL_LOSS.pth \
      --sample_ratio 1.0 --epochs 10
"""
import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add project root and reproduce/ to path for imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(Path(_PROJECT_ROOT) / "reproduce"))

from src.config import Config, MemoryConfig, RLConfig
from src.memory import MemoryManager, MemoryEncoder
from src.rl.actions import ACTION_REGISTRY, NUM_ACTIONS, action_to_candidate_strategy
from src.rl.policy import POMDPPolicy, UncertaintyAdaptiveEpsilon


# ──────────────────────────────────────────────────────────────────
#  Config presets
# ──────────────────────────────────────────────────────────────────

def get_3060_config(sample_ratio: float = 0.05) -> Config:
    """RTX 3060 friendly config: smaller dims, batch, memory."""
    config = Config()
    config.memory = MemoryConfig(
        short_term_capacity=30,
        memory_dim=128,
        faiss_index_type="Flat",
        top_k_retrieval=10,
        time_decay_lambda=0.01,
    )
    config.rl = RLConfig(
        state_dim=512,
        hidden_dim=128,
        num_actions=NUM_ACTIONS,
        policy_arch="mlp",
        num_layers=2,
        dropout=0.1,
        epsilon_start=0.3,
        epsilon_end=0.05,
        epsilon_decay_steps=5000,
        algorithm="cql",
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        batch_size=64,
        grad_clip=1.0,
        cql_alpha=0.5,
    )
    config.training.total_steps = 2000
    return config


def get_server_config(sample_ratio: float = 0.05) -> Config:
    """RTX PRO 6000 (96GB) config: larger dims, full memory, big batches."""
    config = Config()
    config.memory = MemoryConfig(
        short_term_capacity=50,
        memory_dim=256,
        faiss_index_type="IVFFlat",
        top_k_retrieval=20,
        time_decay_lambda=0.01,
    )
    config.rl = RLConfig(
        state_dim=512,
        hidden_dim=256,
        num_actions=NUM_ACTIONS,
        policy_arch="mlp",
        num_layers=3,
        dropout=0.1,
        epsilon_start=0.3,
        epsilon_end=0.05,
        epsilon_decay_steps=10000,
        algorithm="cql",
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        batch_size=256,
        grad_clip=1.0,
        cql_alpha=1.0,
    )
    config.training.total_steps = 10000
    return config


# ──────────────────────────────────────────────────────────────────
#  Data Bridge: P5 format → RL state format
# ──────────────────────────────────────────────────────────────────

class SequentialDataBridge:
    """
    Converts P5 sequential recommendation data into RL-compatible
    (user_history, target_item) format with precomputed embeddings.
    """

    def __init__(self, data_dir: str, dataset: str = "beauty",
                 p5_model=None, tokenizer=None, device: str = "cuda"):
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.p5 = p5_model
        self.tokenizer = tokenizer
        self.device = device

        # Load P5 data files
        base = self.data_dir / dataset
        self.sequential = self._read_lines(base / "sequential_data.txt")
        self.datamaps = json.load(open(base / "datamaps.json"))
        self.user_id2name = pickle_load(base / "user_id2name.pkl")

        # Item metadata
        self.item2id = self.datamaps.get("item2id", {})
        self.id2item = self.datamaps.get("id2item", {})
        self.user2id = self.datamaps.get("user2id", {})

        # Parse sequential data into user → item sequences
        self.user_sequences = {}  # user_id → [item_id, ...]
        for line in self.sequential:
            parts = line.strip().split()
            if len(parts) >= 3:
                self.user_sequences[parts[0]] = [int(i) for i in parts[1:]]

        # All items for evaluation
        self.all_items = list(self.item2id.values())
        print(f"  Users: {len(self.user_sequences)}, Items: {len(self.all_items)}")

    def _read_lines(self, path):
        with open(path, 'r') as f:
            return [l.rstrip('\n') for l in f]

    def get_train_data(self, sample_ratio: float = 1.0, min_history: int = 2
                       ) -> List[Dict]:
        """
        Extract (history, target) pairs for offline RL training.
        Leave-last-out format: history = all but last, target = last.
        """
        samples = []
        for user_id, item_seq in self.user_sequences.items():
            for i in range(min_history, len(item_seq)):
                history = item_seq[:i]
                target = item_seq[i]
                samples.append({
                    "user_id": user_id,
                    "history": history,
                    "target_item": target,
                })

        if sample_ratio < 1.0:
            n = max(1, int(len(samples) * sample_ratio))
            random.shuffle(samples)
            samples = samples[:n]

        print(f"  Training samples: {len(samples)} (sample_ratio={sample_ratio})")
        return samples

    def get_test_data(self, max_users: int = 5000) -> List[Dict]:
        """Leave-last-out test samples."""
        samples = []
        for user_id, item_seq in self.user_sequences.items():
            if len(item_seq) >= 3:
                samples.append({
                    "user_id": user_id,
                    "history": item_seq[:-1],
                    "target_item": item_seq[-1],
                })

        if len(samples) > max_users:
            samples = random.Random(42).sample(samples, max_users)

        print(f"  Test samples: {len(samples)}")
        return samples

    def encode_user_history(self, item_seq: List[int]) -> np.ndarray:
        """Encode user's item sequence into a fixed-dim embedding."""
        if self.p5 is not None and len(item_seq) > 0:
            with torch.no_grad():
                # Use P5's shared embedding to get item representations
                ids = torch.tensor([int(i) % self.p5.shared.num_embeddings
                                    for i in item_seq[-50:]],
                                   device=self.device)
                embs = self.p5.shared(ids)
                return embs.mean(dim=0).cpu().numpy().astype(np.float32)
        return np.zeros(512, dtype=np.float32)

    def get_all_item_embs(self, max_items: int = None) -> Dict[int, np.ndarray]:
        """Precompute embeddings for all items (or top N by frequency)."""
        items = list(self.item2id.values())
        if max_items and len(items) > max_items:
            items = items[:max_items]

        item_embs = {}
        if self.p5 is not None:
            with torch.no_grad():
                for item_id in tqdm(items, desc="Encoding items"):
                    iid = int(item_id) % self.p5.shared.num_embeddings
                    emb = self.p5.shared(torch.tensor([iid], device=self.device))
                    item_embs[item_id] = emb[0].cpu().numpy().astype(np.float32)
        return item_embs


def pickle_load(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────
#  Candidate Retrieval Engine
# ──────────────────────────────────────────────────────────────────

class CandidateRetriever:
    """
    Maps RL strategy → ranked candidate list.
    Replaces P5's beam search with efficient embedding-based retrieval.
    """

    def __init__(self, item_embs: Dict[int, np.ndarray],
                 user_sequences: Dict[str, List[int]], topk: int = 20):
        self.item_embs = item_embs
        self.user_sequences = user_sequences
        self.topk = topk

        self.item_ids = list(item_embs.keys())
        self.item_matrix = np.stack(list(item_embs.values()))  # (N, D)
        self.item_norm = self.item_matrix / (np.linalg.norm(
            self.item_matrix, axis=1, keepdims=True) + 1e-8)
        self._item2idx = {iid: idx for idx, iid in enumerate(self.item_ids)}

    def retrieve(self, user_id: str, user_emb: np.ndarray,
                 strategy: Dict, history: List[int] = None,
                 exclude_items: set = None) -> List[int]:
        """
        Retrieve top-k candidates based on strategy.
        """
        method = strategy.get("method", "similar_to_last")
        exclude = exclude_items or set()
        if history:
            exclude.update(history)

        if method == "similar_to_last":
            return self._similar_to_last(user_emb, exclude)
        elif method == "diverse_categories":
            return self._diverse(user_emb, exclude)
        elif method == "periodic_recall":
            return self._periodic(user_id, exclude)
        elif method == "diverse_mix":
            return self._diverse_mix(user_emb, exclude)
        elif method == "rank_by_similarity":
            return self._similarity_rank(user_emb, exclude)
        else:
            return self._similar_to_last(user_emb, exclude)

    def _similarity_rank(self, query_emb: np.ndarray,
                         exclude: set) -> List[int]:
        """Rank all items by embedding similarity, excluding seen."""
        query = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        scores = self.item_norm @ query  # (N,)
        ranked = np.argsort(scores)[::-1]

        results = []
        for idx in ranked:
            iid = self.item_ids[idx]
            if iid not in exclude:
                results.append(int(iid))
            if len(results) >= self.topk:
                break
        return results

    def _similar_to_last(self, user_emb: np.ndarray, exclude: set
                         ) -> List[int]:
        """Recommend items similar to user's embedding (last interactions)."""
        return self._similarity_rank(user_emb, exclude)

    def _diverse(self, user_emb: np.ndarray, exclude: set) -> List[int]:
        """Diverse retrieval: pick top-1 from each of top-k distant clusters."""
        query = user_emb / (np.linalg.norm(user_emb) + 1e-8)
        scores = self.item_norm @ query
        ranked = np.argsort(scores)

        # Simplified MMR: pick far-apart items from the ranked list
        results = []
        candidate_indices = []
        for idx in ranked[::-1]:  # start from the least similar
            iid = self.item_ids[idx]
            if iid not in exclude:
                candidate_indices.append(idx)
                results.append(int(iid))
            if len(results) >= self.topk * 3:
                break

        # Re-rank for diversity: maximize pairwise distance
        if len(results) > self.topk:
            selected = self._mmr_select(candidate_indices[:len(results)],
                                        self.topk, query)
            results = [int(self.item_ids[candidate_indices[i]])
                       for i in selected]

        return results[:self.topk]

    def _mmr_select(self, indices: List[int], k: int,
                    query: np.ndarray) -> List[int]:
        """Maximal Marginal Relevance selection."""
        if len(indices) <= k:
            return list(range(len(indices)))

        embs = self.item_norm[indices]
        selected = [0]  # start with most diverse from query
        remaining = list(range(1, len(indices)))

        for _ in range(k - 1):
            best_score = -float('inf')
            best_idx = None
            for ridx in remaining:
                sim_to_query = embs[ridx] @ query
                sim_to_selected = max(embs[ridx] @ embs[s] for s in selected)
                mmr = 0.3 * sim_to_query - 0.7 * sim_to_selected
                if mmr > best_score:
                    best_score = mmr
                    best_idx = ridx
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected

    def _periodic(self, user_id: str, exclude: set) -> List[int]:
        """Recall items from periodic purchase patterns (memory-driven)."""
        # Fallback: return similar items (periodic patterns require
        # memory system with timestamp data, not available in initial setup)
        hist = self.user_sequences.get(user_id, [])
        if hist:
            hist_embs = [self.item_embs.get(i) for i in hist[-5:]
                         if i in self.item_embs]
            if hist_embs:
                user_emb = np.mean(hist_embs, axis=0)
                return self._similarity_rank(user_emb, exclude)
        return []

    def _diverse_mix(self, user_emb: np.ndarray, exclude: set) -> List[int]:
        """Mix: 50% similar + 50% diverse."""
        similar = self._similarity_rank(user_emb, exclude)[:self.topk // 2]
        diverse = self._diverse(user_emb, exclude | set(similar))[:self.topk // 2]
        return similar + diverse


# ──────────────────────────────────────────────────────────────────
#  RL + Memory Model (integrated with P5 backbone)
# ──────────────────────────────────────────────────────────────────

class RLMemoryRecommender(nn.Module):
    """
    Complete RL + Memory recommendation model wrapped around P5 encoder.

    State:  P5 encodes user history → user_emb
           Memory system retrieves context + confidence
           → POMDP state

    Action: RL policy outputs strategy index
            → CandidateRetriever produces ranked item list

    This replaces P5's beam search (20× decoder) with 1× policy forward.
    """

    def __init__(self, p5_model, config: Config,
                 user_sequences: Dict[str, List[int]],
                 device: str = "cuda"):
        super().__init__()
        self.p5 = p5_model
        self.config = config
        self.device = device
        self.user_sequences = user_sequences

        user_dim = p5_model.config.d_model  # 512 for t5-small

        # Memory encoder (compresses interactions)
        self.memory_encoder = MemoryEncoder(
            num_users=50000, num_items=50000,
            num_actions=4, embed_dim=128,
            memory_dim=config.memory.memory_dim,
        ).to(device)

        # Memory manager
        self.memory_manager = MemoryManager(
            memory_dim=config.memory.memory_dim,
            short_term_capacity=config.memory.short_term_capacity,
            top_k=config.memory.top_k_retrieval,
            time_decay_lambda=config.memory.time_decay_lambda,
            device=device,
        )
        self.memory_manager.set_encoder(self.memory_encoder)

        # RL Policy network (Q-values for each macro-action)
        self.policy = POMDPPolicy(
            user_dim=user_dim,
            memory_dim=config.memory.memory_dim,
            num_actions=NUM_ACTIONS,
            hidden_dim=config.rl.hidden_dim,
            num_layers=config.rl.num_layers,
            dropout=config.rl.dropout,
        ).to(device)

        # Exploration scheduler
        self.epsilon_scheduler = UncertaintyAdaptiveEpsilon(
            epsilon_base=config.rl.epsilon_end,
            epsilon_max=config.rl.epsilon_start,
        )

        # Candidate retriever (initialized after item_embs computed)
        self.retriever: Optional[CandidateRetriever] = None
        self._train_step = 0

    def set_retriever(self, retriever: CandidateRetriever):
        self.retriever = retriever

    def encode_user(self, user_id: str) -> np.ndarray:
        """Encode user's full interaction history via P5 shared embedding."""
        item_seq = self.user_sequences.get(user_id, [])
        if not item_seq or self.p5 is None:
            return np.zeros(self.p5.config.d_model if self.p5 else 512,
                            dtype=np.float32)
        with torch.no_grad():
            ids = torch.tensor([int(i) % self.p5.shared.num_embeddings
                                for i in item_seq[-50:]],
                               device=self.device)
            return self.p5.shared(ids).mean(dim=0).cpu().numpy().astype(np.float32)

    def get_state(self, user_id: str, user_emb: np.ndarray = None
                  ) -> Dict[str, np.ndarray]:
        """Build the full POMDP state."""
        if user_emb is None:
            user_emb = self.encode_user(user_id)

        mem_state = self.memory_manager.get_state_vector(
            user_id=hash(user_id) % 50000,
            user_embedding=user_emb.astype(np.float32),
        )

        st_context = mem_state["short_term_context"]
        lt_context = mem_state["long_term_context"]
        if st_context.shape != lt_context.shape:
            dim = self.config.memory.memory_dim
            st_context = np.zeros(dim, dtype=np.float32)
            lt_context = np.zeros(dim, dtype=np.float32)

        memory_ctx = np.concatenate([st_context, lt_context]).astype(np.float32)
        confidence = np.array([mem_state["retrieval_confidence"]], dtype=np.float32)

        psv = list(mem_state["periodic_signals"].values())[:10]
        periodic = np.zeros(10, dtype=np.float32)
        for i, v in enumerate(psv):
            periodic[i] = float(v)

        return {
            "user_emb": user_emb.astype(np.float32),
            "memory_context": memory_ctx,
            "retrieval_confidence": confidence,
            "periodic_flags": periodic,
        }

    def forward(self, user_id: str, user_emb: np.ndarray = None,
                deterministic: bool = False, topk: int = 20) -> Dict:
        """
        One recommendation step: state → policy → strategy → candidates.
        Returns ranked item list + metadata.
        """
        state = self.get_state(user_id, user_emb)

        ue_t = torch.from_numpy(state["user_emb"]).unsqueeze(0).to(self.device)
        mc_t = torch.from_numpy(state["memory_context"]).unsqueeze(0).to(self.device)
        cf_t = torch.from_numpy(state["retrieval_confidence"]
                                ).unsqueeze(0).to(self.device)
        pf_t = torch.from_numpy(state["periodic_flags"]
                                ).unsqueeze(0).to(self.device)

        epsilon = 0.0 if deterministic else self.epsilon_scheduler.get_epsilon(
            float(state["retrieval_confidence"][0]),
            global_step=self._train_step,
        )

        with torch.no_grad():
            q_values = self.policy(ue_t, mc_t, cf_t, pf_t)
            action_id = self.policy.get_action(
                ue_t, mc_t, cf_t, pf_t, epsilon=epsilon).item()

        strategy = action_to_candidate_strategy(action_id)
        action_name = ACTION_REGISTRY[action_id].name

        candidates = []
        if self.retriever is not None:
            history = self.user_sequences.get(user_id, [])
            exclude = set(history)
            candidates = self.retriever.retrieve(
                user_id, state["user_emb"], strategy, history, exclude)
            if len(candidates) > topk:
                candidates = candidates[:topk]

        return {
            "action_id": action_id,
            "action_name": action_name,
            "strategy": strategy,
            "candidates": candidates,
            "confidence": float(state["retrieval_confidence"][0]),
            "epsilon": epsilon,
            "q_values": q_values.squeeze(0).cpu().numpy(),
            "state": state,
        }

    def add_to_memory(self, user_id: str, item_id: int,
                      action_type: int = 0, rating: float = 0.5):
        """Record an interaction into the memory system."""
        self.memory_manager.add_interaction(
            user_id=hash(user_id) % 50000,
            item_id=int(item_id) % 50000,
            action_type=action_type,
            rating=rating,
            timestamp=time.time(),
        )

    def get_action_distribution(self, user_id: str) -> np.ndarray:
        """Get softmax distribution over actions (for analysis)."""
        state = self.get_state(user_id)
        ue_t = torch.from_numpy(state["user_emb"]).unsqueeze(0).to(self.device)
        mc_t = torch.from_numpy(state["memory_context"]).unsqueeze(0).to(self.device)
        cf_t = torch.from_numpy(state["retrieval_confidence"]
                                ).unsqueeze(0).to(self.device)
        pf_t = torch.from_numpy(state["periodic_flags"]
                                ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy(ue_t, mc_t, cf_t, pf_t)
            return F.softmax(q, dim=-1).squeeze(0).cpu().numpy()


# ──────────────────────────────────────────────────────────────────
#  Training
# ──────────────────────────────────────────────────────────────────

def train_rl_policy(model: RLMemoryRecommender,
                    data_bridge: SequentialDataBridge,
                    config: Config,
                    train_samples: List[Dict],
                    p5_model=None):
    """
    Offline RL training on pre-collected recommendation data.

    Approach:
      1. BC warm-start: train policy to predict which strategy would have
         placed the target item high in the ranking
      2. CQL fine-tuning: conservative Q-learning on offline data
    """
    print("\n" + "=" * 60)
    print("Training RL Policy")
    print("=" * 60)

    # ── Phase 1: Prepare training data ──
    print("\nPhase 1: Preparing offline training data...")
    model._train_step = 0

    # Simplified: For each training sample, compute which strategy
    # would have retrieved the target item. Use that as the supervised label.
    print(f"  Building training buffer from {len(train_samples)} samples...")

    policy_buffer = []
    for sample in tqdm(train_samples[:config.training.total_steps * 4],
                       desc="Building buffer"):
        user_id = sample["user_id"]
        history = sample["history"]
        target = sample["target_item"]

        if model.retriever is None:
            continue

        user_emb = data_bridge.encode_user_history(history)

        # For BC: determine which strategy produces best rank for target
        best_action, best_rank = 0, float('inf')
        for aid in range(NUM_ACTIONS):
            strategy = action_to_candidate_strategy(aid)
            candidates = model.retriever.retrieve(
                user_id, user_emb, strategy, history)
            if target in candidates:
                rank = candidates.index(target) + 1
                if rank < best_rank:
                    best_rank = rank
                    best_action = aid

        # If no strategy finds the target, use "exploit_similar" as default
        if best_rank == float('inf'):
            best_action = 1  # exploit_similar

        # Store (state → action) for BC
        state = model.get_state(user_id, user_emb)
        policy_buffer.append({
            "user_emb": state["user_emb"],
            "memory_context": state["memory_context"],
            "retrieval_confidence": state["retrieval_confidence"],
            "periodic_flags": state["periodic_flags"],
            "action": best_action,
        })

        if len(policy_buffer) >= config.training.total_steps * 4:
            break

    print(f"  Buffer size: {len(policy_buffer)}")

    # ── Phase 2: Behavior Cloning warm-start ──
    print("\nPhase 2: Behavior Cloning warm-start...")
    model.policy.train()
    optimizer = torch.optim.AdamW(model.policy.parameters(), lr=1e-3)
    bc_steps = config.training.total_steps

    for step in range(bc_steps):
        batch = random.sample(policy_buffer, min(config.rl.batch_size,
                                                   len(policy_buffer)))

        ue = torch.from_numpy(np.stack([b["user_emb"]
                                        for b in batch])).to(model.device)
        mc = torch.from_numpy(np.stack([b["memory_context"]
                                        for b in batch])).to(model.device)
        cf = torch.from_numpy(np.stack([b["retrieval_confidence"]
                                        for b in batch])).to(model.device)
        pf = torch.from_numpy(np.stack([b["periodic_flags"]
                                        for b in batch])).to(model.device)
        actions = torch.tensor([b["action"] for b in batch],
                               device=model.device)

        q_values = model.policy(ue, mc, cf, pf)
        loss = F.cross_entropy(q_values, actions)

        optimizer.zero_grad()
        loss.backward()
        if config.rl.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(),
                                          config.rl.grad_clip)
        optimizer.step()
        model._train_step += 1

        if step % 200 == 0:
            acc = (q_values.argmax(-1) == actions).float().mean()
            print(f"  BC Step {step}/{bc_steps}: loss={loss.item():.4f}, "
                  f"acc={acc.item():.3f}")

    print("  BC warm-start complete.")

    # ── Phase 3: CQL offline fine-tuning ──
    print("\nPhase 3: CQL offline fine-tuning...")

    # Simple CQL implementation
    # Q-values for state-action pairs get reward=1 if strategy finds target
    cql_steps = config.training.total_steps // 2
    cql_alpha = config.rl.cql_alpha

    # Build target network
    target_policy = POMDPPolicy(
        user_dim=model.p5.config.d_model,
        memory_dim=config.memory.memory_dim,
        num_actions=NUM_ACTIONS,
        hidden_dim=config.rl.hidden_dim,
        num_layers=config.rl.num_layers,
        dropout=config.rl.dropout,
    ).to(model.device)
    target_policy.load_state_dict(model.policy.state_dict())

    cql_optimizer = torch.optim.AdamW(model.policy.parameters(),
                                       lr=config.rl.lr)

    for step in range(cql_steps):
        batch = random.sample(policy_buffer, min(config.rl.batch_size,
                                                   len(policy_buffer)))

        ue = torch.from_numpy(np.stack([b["user_emb"]
                                        for b in batch])).to(model.device)
        mc = torch.from_numpy(np.stack([b["memory_context"]
                                        for b in batch])).to(model.device)
        cf = torch.from_numpy(np.stack([b["retrieval_confidence"]
                                        for b in batch])).to(model.device)
        pf = torch.from_numpy(np.stack([b["periodic_flags"]
                                        for b in batch])).to(model.device)
        actions = torch.tensor([b["action"] for b in batch],
                               device=model.device)

        # Current Q-values
        q_all = model.policy(ue, mc, cf, pf)
        q_chosen = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (reward = 1.0 for the "correct" strategy)
        with torch.no_grad():
            target_q = target_policy(ue, mc, cf, pf)
            target_max = target_q.max(dim=1).values
            target_value = 1.0 + config.rl.gamma * target_max

        # TD error
        td_loss = F.mse_loss(q_chosen, target_value)

        # CQL conservative penalty: push down Q for all actions
        cql_loss = cql_alpha * (q_all.mean() - q_chosen.mean())

        total_loss = td_loss + cql_loss

        cql_optimizer.zero_grad()
        total_loss.backward()
        if config.rl.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(),
                                          config.rl.grad_clip)
        cql_optimizer.step()

        # Soft update target network
        with torch.no_grad():
            for p, tp in zip(model.policy.parameters(),
                            target_policy.parameters()):
                tp.data.copy_(config.rl.tau * p.data +
                             (1 - config.rl.tau) * tp.data)

        model._train_step += 1

        if step % 200 == 0:
            print(f"  CQL Step {step}/{cql_steps}: td_loss={td_loss.item():.4f}, "
                  f"cql_loss={cql_loss.item():.4f}, q_mean={q_all.mean().item():.3f}")

    print("  CQL training complete.")


# ──────────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────────

def evaluate_p5_baseline(p5_model, tokenizer, test_samples: List[Dict],
                         all_items: List[str], device: str,
                         beam_size: int = 20, max_eval: int = 500
                         ) -> Dict[str, float]:
    """
    Evaluate P5 baseline on sequential recommendation.
    Uses beam search (B=20) → all-item ranking → HR@k, NDCG@k.
    """
    print("\n" + "-" * 40)
    print("Evaluating P5 Baseline (Beam Search B=20)...")
    print("-" * 40)

    hr = {1: 0, 5: 0, 10: 0}
    ndcg = {5: 0.0, 10: 0.0}
    total = 0

    for sample in tqdm(test_samples[:max_eval], desc="P5 Eval"):
        try:
            user_id = sample["user_id"]
            history = sample["history"]
            target = sample["target_item"]

            if len(history) < 1:
                continue

            # Build prompt (2-3 format from P5 paper)
            history_str = ", ".join(str(h) for h in history[-30:])
            source = (f"Here is user_{user_id}'s purchase history: "
                      f"{history_str}. Try to recommend the next item.")

            input_ids = tokenizer.encode(source, truncation=True, max_length=512)
            input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = p5_model.generate(
                    input_ids=input_tensor,
                    max_length=20,
                    num_beams=beam_size,
                    num_return_sequences=min(beam_size, 20),
                    early_stopping=True,
                )

            predicted = []
            for out in outputs:
                text = tokenizer.decode(out, skip_special_tokens=True).strip()
                import re
                nums = re.findall(r'\d+', text)
                for n in nums:
                    if n not in predicted:
                        predicted.append(n)
                        break

            target_str = str(target)
            if target_str in predicted:
                rank = predicted.index(target_str) + 1
            else:
                rank = float('inf')

            for k in [1, 5, 10]:
                if rank <= k:
                    hr[k] += 1
            if rank <= 5 and rank != float('inf'):
                ndcg[5] += 1.0 / np.log2(rank + 1)
            if rank <= 10 and rank != float('inf'):
                ndcg[10] += 1.0 / np.log2(rank + 1)

            total += 1
        except Exception:
            continue

    if total == 0:
        return {f'HR@{k}': 0.0 for k in [1, 5, 10]} | \
               {f'NDCG@{k}': 0.0 for k in [5, 10]} | {'count': 0}

    return {
        'HR@1': round(hr[1] / total, 4),
        'HR@5': round(hr[5] / total, 4),
        'HR@10': round(hr[10] / total, 4),
        'NDCG@5': round(ndcg[5] / total, 4),
        'NDCG@10': round(ndcg[10] / total, 4),
        'count': total,
    }


def evaluate_rl_memory(model: RLMemoryRecommender,
                       test_samples: List[Dict],
                       max_eval: int = 500) -> Dict[str, float]:
    """
    Evaluate RL + Memory model on sequential recommendation.
    Uses 1-pass policy → strategy → candidate ranking → HR@k, NDCG@k.
    """
    print("\n" + "-" * 40)
    print("Evaluating RL + Memory Model...")
    print("-" * 40)

    hr = {1: 0, 5: 0, 10: 0}
    ndcg = {5: 0.0, 10: 0.0}
    total = 0

    action_counts = defaultdict(int)

    for sample in tqdm(test_samples[:max_eval], desc="RL+Mem Eval"):
        try:
            user_id = sample["user_id"]
            history = sample["history"]
            target = sample["target_item"]

            if len(history) < 1 or model.retriever is None:
                continue

            user_emb = None
            if model.p5 is not None:
                with torch.no_grad():
                    ids = torch.tensor([int(i) % model.p5.shared.num_embeddings
                                        for i in history[-30:]],
                                       device=model.device)
                    user_emb = model.p5.shared(ids).mean(dim=0).cpu().numpy().astype(np.float32)

            result = model.forward(user_id, user_emb, deterministic=True, topk=20)
            candidates = result["candidates"]
            action_counts[result["action_name"]] += 1

            target_int = int(target)
            if target_int in candidates:
                rank = candidates.index(target_int) + 1
            else:
                rank = float('inf')

            for k in [1, 5, 10]:
                if rank <= k:
                    hr[k] += 1
            if rank <= 5 and rank != float('inf'):
                ndcg[5] += 1.0 / np.log2(rank + 1)
            if rank <= 10 and rank != float('inf'):
                ndcg[10] += 1.0 / np.log2(rank + 1)

            total += 1
        except Exception:
            continue

    if total == 0:
        return {f'HR@{k}': 0.0 for k in [1, 5, 10]} | \
               {f'NDCG@{k}': 0.0 for k in [5, 10]} | {'count': 0}

    results = {
        'HR@1': round(hr[1] / total, 4),
        'HR@5': round(hr[5] / total, 4),
        'HR@10': round(hr[10] / total, 4),
        'NDCG@5': round(ndcg[5] / total, 4),
        'NDCG@10': round(ndcg[10] / total, 4),
        'count': total,
    }

    # Action distribution
    total_actions = sum(action_counts.values())
    if total_actions > 0:
        results['action_dist'] = {
            k: round(v / total_actions, 3)
            for k, v in sorted(action_counts.items(),
                               key=lambda x: -x[1])[:5]
        }

    return results


def print_comparison(p5_results: Dict, rl_results: Dict):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("  COMPARISON: P5 Baseline vs RL + Memory")
    print("=" * 70)
    print(f"  {'Metric':<15s} {'P5 (Beam=20)':>16s} {'RL+Memory':>16s} {'Diff':>12s}")
    print("  " + "-" * 60)

    for metric in ['HR@1', 'HR@5', 'HR@10', 'NDCG@5', 'NDCG@10']:
        p5v = p5_results.get(metric, 0)
        rlv = rl_results.get(metric, 0)
        diff = rlv - p5v
        sign = "+" if diff >= 0 else ""
        print(f"  {metric:<15s} {p5v:>16.4f} {rlv:>16.4f} {sign}{diff:>11.4f}")

    print("  " + "-" * 60)
    print(f"  {'Eval samples':<15s} {p5_results.get('count', 0):>16d} "
          f"{rl_results.get('count', 0):>16d}")

    if 'action_dist' in rl_results:
        print(f"\n  RL Action Distribution (top-5):")
        for action, prob in rl_results['action_dist'].items():
            print(f"    {action}: {prob:.1%}")

    print("=" * 70)


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="RL + Memory vs P5 Baseline Comparison Experiment")
    p.add_argument('--dataset', type=str, default='beauty')
    p.add_argument('--backbone', type=str, default='t5-small')
    p.add_argument('--p5_checkpoint', type=str, required=True,
                   help='Path to trained P5 checkpoint')
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--sample_ratio', type=float, default=1.0,
                   help='Fraction of training data (0.05 = 5%%)')
    p.add_argument('--epochs', type=int, default=1,
                   help='Number of training passes (for RL: BC+CQL steps)')
    p.add_argument('--batch_size', type=int, default=4,
                   help='Batch size (4 for 3060)')
    p.add_argument('--beam_size', type=int, default=20,
                   help='Beam size for P5 eval')
    p.add_argument('--max_eval', type=int, default=500,
                   help='Max test samples for evaluation')
    p.add_argument('--output_dir', type=str, default='outputs/rl_memory')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--config_preset', type=str, default='3060',
                   choices=['3060', 'server'],
                   help='Config preset: 3060 (6GB) or server (96GB)')
    p.add_argument('--skip_training', action='store_true',
                   help='Skip training, only evaluate')
    return p.parse_args()


def main():
    args = parse_args()

    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Output dir
    timestamp = datetime.now().strftime('%b%d_%H-%M')
    run_name = f"rlmem-{args.dataset}-sample{args.sample_ratio}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Config — select preset based on GPU
    if args.config_preset == 'server':
        config = get_server_config(args.sample_ratio)
    else:
        config = get_3060_config(args.sample_ratio)
    config.rl.batch_size = args.batch_size
    config.training.total_steps = max(2000, args.epochs * 500)

    # ── Load P5 model ──
    print("\nLoading P5 backbone...")
    from modeling_p5 import P5
    from tokenization import P5Tokenizer
    from transformers import T5Config

    t5config = T5Config.from_pretrained(args.backbone)
    t5config.losses = 'rating,sequential,explanation,review,traditional'

    tokenizer = P5Tokenizer.from_pretrained(args.backbone, max_length=512,
                                            do_lower_case=True)
    p5_model = P5.from_pretrained(args.backbone, config=t5config)
    p5_model.resize_token_embeddings(tokenizer.vocab_size)

    # Fix whole_word_embeddings (same fix as reproduce/train.py)
    if hasattr(p5_model.encoder, 'whole_word_embeddings'):
        p5_model.encoder.whole_word_embeddings.weight.data.normal_(mean=0.0, std=1.0)

    # Load trained checkpoint
    if os.path.exists(args.p5_checkpoint):
        print(f"Loading P5 weights from {args.p5_checkpoint}...")
        ckpt = torch.load(args.p5_checkpoint, map_location=device)
        state_dict = ckpt.get('model', ckpt)
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state[k] = v
        p5_model.load_state_dict(new_state, strict=False)
        print("  P5 weights loaded.")
    else:
        print(f"  WARNING: Checkpoint not found at {args.p5_checkpoint}")
        print("  Using untrained P5 backbone (expect poor results).")

    p5_model = p5_model.to(device)
    p5_model.eval()

    # ── Prepare data ──
    print("\nLoading data...")
    data_bridge = SequentialDataBridge(
        args.data_dir, args.dataset,
        p5_model=p5_model, tokenizer=tokenizer, device=device,
    )

    # Precompute item embeddings for candidate retrieval
    print("\nPrecomputing item embeddings...")
    item_embs = data_bridge.get_all_item_embs(max_items=5000)

    # Build retriever
    retriever = CandidateRetriever(
        item_embs=item_embs,
        user_sequences=data_bridge.user_sequences,
        topk=20,
    )
    print(f"  Item embedding matrix: {retriever.item_matrix.shape}")

    # ── Build RL + Memory model ──
    print("\nInitializing RL + Memory model...")
    rl_model = RLMemoryRecommender(
        p5_model=p5_model,
        config=config,
        user_sequences=data_bridge.user_sequences,
        device=device,
    )
    rl_model.set_retriever(retriever)

    total_params = sum(p.numel() for p in [
        rl_model.memory_encoder, rl_model.policy])
    print(f"  RL+Memory extra params: {total_params/1e6:.2f}M")
    print(f"  Actions: {NUM_ACTIONS}")
    print(f"  Memory dim: {config.memory.memory_dim}")
    print(f"  RL algorithm: {config.rl.algorithm}")

    # ── Training ──
    if not args.skip_training:
        train_samples = data_bridge.get_train_data(sample_ratio=args.sample_ratio)
        test_samples = data_bridge.get_test_data(max_users=args.max_eval)

        # Also warm up the memory with some user history
        print("\nWarming up memory system...")
        for sample in tqdm(train_samples[:min(2000, len(train_samples))],
                           desc="Memory warmup"):
            rl_model.add_to_memory(
                sample["user_id"], sample["target_item"],
                action_type=1, rating=0.7,
            )

        train_rl_policy(rl_model, data_bridge, config,
                        train_samples, p5_model=p5_model)
    else:
        test_samples = data_bridge.get_test_data(max_users=args.max_eval)

    # ── Evaluation: P5 vs RL+Memory ──
    print("\n" + "=" * 60)
    print("Comparison Evaluation")
    print("=" * 60)

    p5_results = evaluate_p5_baseline(
        p5_model, tokenizer, test_samples,
        data_bridge.all_items, device,
        beam_size=args.beam_size, max_eval=args.max_eval,
    )

    rl_results = evaluate_rl_memory(
        rl_model, test_samples, max_eval=args.max_eval,
    )

    print_comparison(p5_results, rl_results)

    # ── Save results ──
    results = {
        "config": {
            "dataset": args.dataset,
            "sample_ratio": args.sample_ratio,
            "epochs": args.epochs,
            "rl_algorithm": config.rl.algorithm,
            "memory_dim": config.memory.memory_dim,
            "beam_size": args.beam_size,
        },
        "p5_baseline": p5_results,
        "rl_memory": rl_results,
    }

    results_path = output_dir / "comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Save model checkpoint
    ckpt_path = output_dir / "rl_model.pth"
    torch.save({
        'memory_encoder': rl_model.memory_encoder.state_dict(),
        'policy': rl_model.policy.state_dict(),
        'config': config,
    }, ckpt_path)
    print(f"Model saved to {ckpt_path}")

    return str(output_dir)


if __name__ == '__main__':
    main()
