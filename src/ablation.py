"""
Ablation experiments for RL + Memory hybrid architecture.

Variants (6 total):
  1. p5_baseline       — P5 beam search B=20 (reference)
  2. full_system        — RL + Short-term + FAISS + Time Decay + Adaptive Epsilon
  3. no_memory          — RL policy only, zero memory context
  4. short_term_only    — Disable FAISS long-term, keep rolling buffer
  5. no_time_decay      — Full system with time_decay_lambda=0
  6. fixed_epsilon      — Full system with constant ε=0.1 (no uncertainty adaptation)

Usage (local RTX 3060):
  python -m src.ablation --dataset beauty --backbone t5-small \\
      --sample_ratio 0.05 --epochs 1 --batch_size 4 --max_eval 200
"""
import os, sys, json, time, random, argparse, re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, str(Path(_PROJECT_ROOT) / "reproduce"))

from src.config import Config, MemoryConfig, RLConfig
from src.memory import MemoryManager, MemoryEncoder
from src.rl.actions import ACTION_REGISTRY, NUM_ACTIONS, action_to_candidate_strategy
from src.rl.policy import POMDPPolicy, UncertaintyAdaptiveEpsilon

# ──────────────────────────────────────────
#  Ablation config registry
# ──────────────────────────────────────────

ABLATION_VARIANTS = {
    "full_system": {
        "name": "RL + Full Memory",
        "desc": "All components enabled",
        "use_memory": True,
        "use_long_term": True,
        "time_decay": True,
        "adaptive_epsilon": True,
    },
    "no_memory": {
        "name": "RL (No Memory)",
        "desc": "Zero memory context, RL policy only",
        "use_memory": False,
        "use_long_term": False,
        "time_decay": False,
        "adaptive_epsilon": True,
    },
    "short_term_only": {
        "name": "RL + Short-term Only",
        "desc": "Rolling buffer only, no FAISS long-term",
        "use_memory": True,
        "use_long_term": False,
        "time_decay": False,
        "adaptive_epsilon": True,
    },
    "no_time_decay": {
        "name": "RL + Full Memory (no decay)",
        "desc": "FAISS retrieval without time decay weighting",
        "use_memory": True,
        "use_long_term": True,
        "time_decay": False,
        "adaptive_epsilon": True,
    },
    "fixed_epsilon": {
        "name": "RL + Fixed Epsilon",
        "desc": "Constant epsilon, no uncertainty adaptation",
        "use_memory": True,
        "use_long_term": True,
        "time_decay": True,
        "adaptive_epsilon": False,
    },
}


# ──────────────────────────────────────────
#  3060 config
# ──────────────────────────────────────────

def get_3060_config(sample_ratio: float = 0.05) -> Config:
    config = Config()
    config.memory = MemoryConfig(
        short_term_capacity=30, memory_dim=128,
        faiss_index_type="Flat", top_k_retrieval=10,
        time_decay_lambda=0.01,
    )
    config.rl = RLConfig(
        state_dim=512, hidden_dim=128, num_actions=NUM_ACTIONS,
        policy_arch="mlp", num_layers=2, dropout=0.1,
        epsilon_start=0.3, epsilon_end=0.05, epsilon_decay_steps=5000,
        algorithm="cql", gamma=0.99, tau=0.005, lr=3e-4,
        batch_size=64, grad_clip=1.0, cql_alpha=0.5,
    )
    config.training.total_steps = 1000  # 1 epoch, quicker
    return config


# ──────────────────────────────────────────
#  Data bridge (reuse from rl_experiment)
# ──────────────────────────────────────────

class SequentialDataBridge:
    def __init__(self, data_dir: str, dataset: str = "beauty",
                 p5_model=None, device: str = "cuda"):
        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.p5 = p5_model
        self.device = device
        base = self.data_dir / dataset
        self.sequential = self._read(base / "sequential_data.txt")
        self.datamaps = json.load(open(base / "datamaps.json"))
        self.item2id = self.datamaps.get("item2id", {})
        self.id2item = self.datamaps.get("id2item", {})
        self.user2id = self.datamaps.get("user2id", {})
        self.user_sequences: Dict[str, List[int]] = {}
        for line in self.sequential:
            parts = line.strip().split()
            if len(parts) >= 3:
                self.user_sequences[parts[0]] = [int(i) for i in parts[1:]]
        self.all_items = list(self.item2id.values())
        print(f"  Users: {len(self.user_sequences)}, Items: {len(self.all_items)}")

    def _read(self, path):
        with open(path, 'r') as f:
            return [l.rstrip('\n') for l in f]

    def get_train_data(self, sample_ratio=1.0, min_history=2):
        user_samples = {}
        for uid, seq in self.user_sequences.items():
            if len(seq) <= min_history:
                continue
            seq_samples = []
            for i in range(min_history, len(seq)):
                seq_samples.append({"user_id": uid, "history": seq[:i], "target_item": seq[i]})
            user_samples[uid] = seq_samples

        if sample_ratio < 1.0:
            n_users = max(1, int(len(user_samples) * sample_ratio))
            selected = set(random.sample(list(user_samples.keys()), n_users))
            samples = []
            for uid in selected:
                samples.extend(user_samples[uid])
            random.shuffle(samples)
        else:
            samples = []
            for seq_samples in user_samples.values():
                samples.extend(seq_samples)

        n_sampled = max(1, int(len(user_samples) * sample_ratio)) if sample_ratio < 1.0 else len(user_samples)
        print(f"  Train samples: {len(samples)} (users={len(user_samples)}, sampled_users={n_sampled}, ratio={sample_ratio})")
        return samples

    def get_test_data(self, max_users=5000):
        samples = []
        for uid, seq in self.user_sequences.items():
            if len(seq) >= 3:
                samples.append({"user_id": uid, "history": seq[:-1], "target_item": seq[-1]})
        if len(samples) > max_users:
            samples = random.Random(42).sample(samples, max_users)
        print(f"  Test samples: {len(samples)}")
        return samples

    def encode_user_history(self, item_seq):
        if self.p5 is not None and len(item_seq) > 0:
            with torch.no_grad():
                ids = torch.tensor([int(i) % self.p5.shared.num_embeddings
                                    for i in item_seq[-50:]], device=self.device)
                return self.p5.shared(ids).mean(dim=0).cpu().numpy().astype(np.float32)
        return np.zeros(512, dtype=np.float32)

    def get_all_item_embs(self, max_items=None):
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


# ──────────────────────────────────────────
#  Candidate retriever (simplified from rl_experiment)
# ──────────────────────────────────────────

class CandidateRetriever:
    def __init__(self, item_embs, user_sequences, topk=20):
        self.item_embs = item_embs
        self.user_sequences = user_sequences
        self.topk = topk
        self.item_ids = list(item_embs.keys())
        self.item_matrix = np.stack(list(item_embs.values()))
        self.item_norm = self.item_matrix / (np.linalg.norm(self.item_matrix, axis=1, keepdims=True) + 1e-8)

    def retrieve(self, user_id, user_emb, strategy, history=None, exclude_items=None):
        method = strategy.get("method", "similar_to_last")
        exclude = exclude_items or set()
        if history:
            exclude.update(history)
        return self._similarity_rank(user_emb, exclude)

    def _similarity_rank(self, query_emb, exclude):
        query = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        scores = self.item_norm @ query
        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked:
            iid = self.item_ids[idx]
            if iid not in exclude:
                results.append(int(iid))
            if len(results) >= self.topk:
                break
        return results

    def _diverse(self, user_emb, exclude):
        query = user_emb / (np.linalg.norm(user_emb) + 1e-8)
        scores = self.item_norm @ query
        ranked = np.argsort(scores)
        results = []
        for idx in ranked[::-1]:
            iid = self.item_ids[idx]
            if iid not in exclude:
                results.append(int(iid))
            if len(results) >= self.topk * 3:
                break
        if len(results) > self.topk:
            selected = self._mmr_select(list(range(len(results))), self.topk, query)
            results = [results[s] for s in selected]
        return results[:self.topk]

    def _mmr_select(self, indices, k, query):
        if len(indices) <= k:
            return list(range(len(indices)))
        embs = self.item_norm[[self.item_ids.index(self.item_ids[i]) if i < len(self.item_ids) else 0 for i in indices]]
        selected = [0]
        remaining = list(range(1, len(indices)))
        for _ in range(k - 1):
            best_score, best_idx = -float('inf'), None
            for ridx in remaining:
                sim_query = float(embs[ridx] @ query)
                sim_sel = max(float(embs[ridx] @ embs[s]) for s in selected)
                mmr = 0.3 * sim_query - 0.7 * sim_sel
                if mmr > best_score:
                    best_score, best_idx = mmr, ridx
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        return selected


# ──────────────────────────────────────────
#  Ablation-aware RL+Memory Model
# ──────────────────────────────────────────

class AblationRLModel:
    """RL + Memory model that supports ablation variants."""

    def __init__(self, p5_model, config, user_sequences, variant, device="cuda",
                 cooc_embeddings=None):
        self.p5 = p5_model
        self.config = config
        self.device = device
        self.user_sequences = user_sequences
        self.variant = variant
        self.variant_cfg = ABLATION_VARIANTS[variant]
        self.use_memory = self.variant_cfg["use_memory"]
        self.use_long_term = self.variant_cfg["use_long_term"]
        self.use_time_decay = self.variant_cfg["time_decay"]
        self.use_adaptive_epsilon = self.variant_cfg["adaptive_epsilon"]
        self.cooc_embeddings = cooc_embeddings

        if cooc_embeddings is not None:
            user_dim = next(iter(cooc_embeddings.values())).shape[0]
        else:
            user_dim = p5_model.config.d_model
        mem_dim = config.memory.memory_dim

        self.memory_encoder = MemoryEncoder(
            num_users=50000, num_items=50000, num_actions=4,
            embed_dim=128, memory_dim=mem_dim,
        ).to(device)

        time_decay = config.memory.time_decay_lambda if self.use_time_decay else 0.0
        self.memory_manager = MemoryManager(
            memory_dim=mem_dim,
            short_term_capacity=config.memory.short_term_capacity,
            top_k=config.memory.top_k_retrieval,
            time_decay_lambda=time_decay,
            device=device,
        )
        self.memory_manager.set_encoder(self.memory_encoder)

        self.policy = POMDPPolicy(
            user_dim=user_dim, memory_dim=mem_dim,
            num_actions=NUM_ACTIONS,
            hidden_dim=config.rl.hidden_dim,
            num_layers=config.rl.num_layers,
            dropout=config.rl.dropout,
        ).to(device)

        if self.use_adaptive_epsilon:
            self.epsilon_scheduler = UncertaintyAdaptiveEpsilon(
                epsilon_base=config.rl.epsilon_end,
                epsilon_max=config.rl.epsilon_start,
            )
        else:
            self.fixed_epsilon = 0.1

        self.retriever: Optional[CandidateRetriever] = None
        self._train_step = 0

    def set_retriever(self, retriever):
        self.retriever = retriever

    def encode_user(self, user_id):
        item_seq = self.user_sequences.get(user_id, [])
        if not item_seq:
            if self.cooc_embeddings:
                return np.zeros(next(iter(self.cooc_embeddings.values())).shape[0], dtype=np.float32)
            return np.zeros(self.p5.config_d_model if self.p5 else 512, dtype=np.float32)

        if self.cooc_embeddings is not None:
            embs = []
            for iid in item_seq[-50:]:
                e = self.cooc_embeddings.get(int(iid))
                if e is not None:
                    embs.append(e)
            if embs:
                return np.mean(embs, axis=0).astype(np.float32)
            return np.zeros(next(iter(self.cooc_embeddings.values())).shape[0], dtype=np.float32)

        if self.p5 is None:
            return np.zeros(512, dtype=np.float32)
        with torch.no_grad():
            ids = torch.tensor([int(i) % self.p5.shared.num_embeddings
                                for i in item_seq[-50:]], device=self.device)
            return self.p5.shared(ids).mean(dim=0).cpu().numpy().astype(np.float32)

    def get_state(self, user_id, user_emb=None):
        if user_emb is None:
            user_emb = self.encode_user(user_id)

        mem_dim = self.config.memory.memory_dim

        if not self.use_memory:
            # Ablation: zero memory context
            return {
                "user_emb": user_emb.astype(np.float32),
                "memory_context": np.zeros(2 * mem_dim, dtype=np.float32),
                "retrieval_confidence": np.array([0.5], dtype=np.float32),
                "periodic_flags": np.zeros(10, dtype=np.float32),
            }

        # Build memory state
        uid_hash = hash(user_id) % 50000

        if self.use_long_term:
            mem_state = self.memory_manager.get_state_vector(
                uid_hash, user_embedding=user_emb.astype(np.float32))
            st_context = mem_state["short_term_context"]
            lt_context = mem_state["long_term_context"]
            if st_context.shape != lt_context.shape:
                st_context = np.zeros(mem_dim, dtype=np.float32)
                lt_context = np.zeros(mem_dim, dtype=np.float32)
            confidence = np.array([mem_state["retrieval_confidence"]], dtype=np.float32)
            psv = list(mem_state["periodic_signals"].values())[:10]
            periodic = np.zeros(10, dtype=np.float32)
            for i, v in enumerate(psv):
                periodic[i] = float(v)
        else:
            # Short-term only: encode recent interactions manually
            recent = self.memory_manager.short_term.get_recent()
            if self.memory_encoder is not None and recent:
                with torch.no_grad():
                    self.memory_encoder.eval()
                    d = next(self.memory_encoder.parameters()).device
                    uids = torch.tensor([r["user_id"] for r in recent], device=d)
                    iids = torch.tensor([r["item_id"] for r in recent], device=d)
                    acts = torch.tensor([r["action_type"] for r in recent], device=d)
                    rats = torch.tensor([r["rating"] for r in recent], device=d, dtype=torch.float)
                    tds = torch.tensor([r["timestamp"] / 86400.0 for r in recent], device=d, dtype=torch.float)
                    vecs = self.memory_encoder(uids, iids, acts, rats, tds)
                    st_context = vecs.mean(dim=0).cpu().numpy().astype(np.float32) if len(vecs) > 0 else np.zeros(mem_dim, dtype=np.float32)
            else:
                st_context = np.zeros(mem_dim, dtype=np.float32)
            lt_context = np.zeros(mem_dim, dtype=np.float32)
            confidence = np.array([0.5], dtype=np.float32)
            periodic = np.zeros(10, dtype=np.float32)

        memory_ctx = np.concatenate([st_context, lt_context]).astype(np.float32)

        return {
            "user_emb": user_emb.astype(np.float32),
            "memory_context": memory_ctx,
            "retrieval_confidence": confidence,
            "periodic_flags": periodic,
        }

    def forward(self, user_id, user_emb=None, deterministic=False, topk=20,
                eval_history=None):
        state = self.get_state(user_id, user_emb)

        ue_t = torch.from_numpy(state["user_emb"]).unsqueeze(0).to(self.device)
        mc_t = torch.from_numpy(state["memory_context"]).unsqueeze(0).to(self.device)
        cf_t = torch.from_numpy(state["retrieval_confidence"]).unsqueeze(0).to(self.device)
        pf_t = torch.from_numpy(state["periodic_flags"]).unsqueeze(0).to(self.device)

        if deterministic:
            epsilon = 0.0
        elif self.use_adaptive_epsilon:
            epsilon = self.epsilon_scheduler.get_epsilon(
                float(state["retrieval_confidence"][0]),
                global_step=self._train_step)
        else:
            epsilon = self.fixed_epsilon

        with torch.no_grad():
            q_values = self.policy(ue_t, mc_t, cf_t, pf_t)
            action_id = self.policy.get_action(ue_t, mc_t, cf_t, pf_t, epsilon=epsilon).item()

        strategy = action_to_candidate_strategy(action_id)
        action_name = ACTION_REGISTRY[action_id].name

        candidates = []
        if self.retriever is not None:
            if eval_history is not None:
                # During eval: use test history for exclusion, NOT full sequence
                exclude = set(eval_history)
                history_for_ret = list(eval_history)
            else:
                exclude = set(self.user_sequences.get(user_id, []))
                history_for_ret = self.user_sequences.get(user_id, [])
            candidates = self.retriever.retrieve(
                user_id, state["user_emb"], strategy,
                history_for_ret, exclude)
            if len(candidates) > topk:
                candidates = candidates[:topk]

        return {
            "action_id": action_id, "action_name": action_name,
            "candidates": candidates,
            "confidence": float(state["retrieval_confidence"][0]),
            "epsilon": epsilon,
        }

    def add_to_memory(self, user_id, item_id, action_type=0, rating=0.5):
        self.memory_manager.add_interaction(
            user_id=hash(user_id) % 50000,
            item_id=int(item_id) % 50000,
            action_type=action_type, rating=rating,
            timestamp=time.time(),
        )


# ──────────────────────────────────────────
#  Training (simplified from rl_experiment)
# ──────────────────────────────────────────

def train_ablation_model(model, data_bridge, config, train_samples):
    """BC warm-start + CQL, optimized buffer building."""
    print(f"  Training {model.variant_cfg['name']}...")

    # Build buffer (optimized: single similarity lookup, rank-based action assignment)
    import numpy as np
    policy_buffer = []
    all_item_ids = model.retriever.item_ids
    item_norm = model.retriever.item_norm
    
    _max_buf = min(config.training.total_steps * 2, 3000)
    for idx, sample in enumerate(tqdm(train_samples[:len(train_samples)], desc="  Building buffer")):
        user_id, history, target = sample["user_id"], sample["history"], sample["target_item"]
        if model.retriever is None:
            continue
        user_emb = data_bridge.encode_user_history(history)
        
        # Single similarity lookup
        query = user_emb / (np.linalg.norm(user_emb) + 1e-8)
        scores = item_norm @ query
        ranked = np.argsort(scores)[::-1]
        
        # Find target rank in similarity-ranked list
        target_rank = float('inf')
        exclude = set(history)
        pos = 0
        for r_idx in ranked:
            iid = all_item_ids[r_idx]
            if iid not in exclude:
                pos += 1
                if int(iid) == int(target):
                    target_rank = pos
                    break
        
        # Rank-based action assignment for diversity
        if target_rank == float('inf'):
            best_action = 1
        elif target_rank <= 5:
            best_action = 1  # exploit_similar
        elif target_rank <= 20:
            best_action = 14  # increase_diversity
        elif target_rank <= 50:
            best_action = 6  # explore_new_category
        else:
            best_action = [0, 7, 10, 12, 15][idx % 5]
        
        state = model.get_state(user_id, user_emb)
        policy_buffer.append({
            "user_emb": state["user_emb"], "memory_context": state["memory_context"],
            "retrieval_confidence": state["retrieval_confidence"],
            "periodic_flags": state["periodic_flags"], "action": best_action,
        })
        if len(policy_buffer) >= _max_buf:
            break

    if len(policy_buffer) < 8:
        print("  Not enough buffer, skipping training")
        return

    # BC warm-start
    model.policy.train()
    optimizer = torch.optim.AdamW(model.policy.parameters(), lr=1e-3)
    bc_steps = config.training.total_steps
    for step in range(bc_steps):
        batch = random.sample(policy_buffer, min(config.rl.batch_size, len(policy_buffer)))
        ue = torch.from_numpy(np.stack([b["user_emb"] for b in batch])).to(model.device)
        mc = torch.from_numpy(np.stack([b["memory_context"] for b in batch])).to(model.device)
        cf = torch.from_numpy(np.stack([b["retrieval_confidence"] for b in batch])).to(model.device)
        pf = torch.from_numpy(np.stack([b["periodic_flags"] for b in batch])).to(model.device)
        actions = torch.tensor([b["action"] for b in batch], device=model.device)
        q_values = model.policy(ue, mc, cf, pf)
        loss = F.cross_entropy(q_values, actions)
        optimizer.zero_grad()
        loss.backward()
        if config.rl.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), config.rl.grad_clip)
        optimizer.step()
        model._train_step += 1
    print(f"  BC complete ({bc_steps} steps)")

    # CQL fine-tuning
    target_policy = POMDPPolicy(
        user_dim=model.policy.user_dim,
        memory_dim=config.memory.memory_dim,
        num_actions=NUM_ACTIONS,
        hidden_dim=config.rl.hidden_dim,
        num_layers=config.rl.num_layers,
        dropout=config.rl.dropout,
    ).to(model.device)
    target_policy.load_state_dict(model.policy.state_dict())

    cql_optimizer = torch.optim.AdamW(model.policy.parameters(), lr=config.rl.lr)
    cql_steps = max(200, config.training.total_steps // 2)
    for step in range(cql_steps):
        batch = random.sample(policy_buffer, min(config.rl.batch_size, len(policy_buffer)))
        ue = torch.from_numpy(np.stack([b["user_emb"] for b in batch])).to(model.device)
        mc = torch.from_numpy(np.stack([b["memory_context"] for b in batch])).to(model.device)
        cf = torch.from_numpy(np.stack([b["retrieval_confidence"] for b in batch])).to(model.device)
        pf = torch.from_numpy(np.stack([b["periodic_flags"] for b in batch])).to(model.device)
        actions = torch.tensor([b["action"] for b in batch], device=model.device)

        q_all = model.policy(ue, mc, cf, pf)
        q_chosen = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q = target_policy(ue, mc, cf, pf)
            target_max = target_q.max(dim=1).values
            target_value = 1.0 + config.rl.gamma * target_max

        td_loss = F.mse_loss(q_chosen, target_value)
        cql_loss = config.rl.cql_alpha * (q_all.mean() - q_chosen.mean())
        total_loss = td_loss + cql_loss

        cql_optimizer.zero_grad()
        total_loss.backward()
        if config.rl.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), config.rl.grad_clip)
        cql_optimizer.step()

        with torch.no_grad():
            for p, tp in zip(model.policy.parameters(), target_policy.parameters()):
                tp.data.copy_(config.rl.tau * p.data + (1 - config.rl.tau) * tp.data)
        model._train_step += 1
    print(f"  CQL complete ({cql_steps} steps)")


# ──────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────

def evaluate_p5_baseline(p5_model, tokenizer, test_samples, all_items, device, beam_size=20, max_eval=200):
    print("  Evaluating P5 baseline (beam search B=20)...")
    hr = {1: 0, 5: 0, 10: 0}
    ndcg = {5: 0.0, 10: 0.0}
    total = 0

    for sample in tqdm(test_samples[:max_eval], desc="  P5 Eval"):
        try:
            user_id, history, target = sample["user_id"], sample["history"], sample["target_item"]
            if len(history) < 1:
                continue
            history_str = ", ".join(str(h) for h in history[-30:])
            source = f"Here is user_{user_id}'s purchase history: {history_str}. Try to recommend the next item."
            input_ids = tokenizer.encode(source, truncation=True, max_length=512)
            input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = p5_model.generate(
                    input_ids=input_tensor, max_length=20,
                    num_beams=beam_size, num_return_sequences=min(beam_size, 20),
                    early_stopping=True)
            predicted = []
            for out in outputs:
                text = tokenizer.decode(out, skip_special_tokens=True).strip()
                nums = re.findall(r'\d+', text)
                for n in nums:
                    if n not in predicted:
                        predicted.append(n)
                        break
            target_str = str(target)
            rank = predicted.index(target_str) + 1 if target_str in predicted else float('inf')
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
        return {f'HR@{k}': 0.0 for k in [1, 5, 10]} | {f'NDCG@{k}': 0.0 for k in [5, 10]} | {'count': 0}
    return {
        'HR@1': round(hr[1] / total, 4), 'HR@5': round(hr[5] / total, 4),
        'HR@10': round(hr[10] / total, 4), 'NDCG@5': round(ndcg[5] / total, 4),
        'NDCG@10': round(ndcg[10] / total, 4), 'count': total,
    }


def evaluate_rl_model(model, test_samples, max_eval=200):
    variant_name = model.variant_cfg['name']
    print(f"  Evaluating {variant_name}...")
    hr = {1: 0, 5: 0, 10: 0}
    ndcg = {5: 0.0, 10: 0.0}
    total = 0
    action_counts = defaultdict(int)

    for sample in tqdm(test_samples[:max_eval], desc=f"  {variant_name} Eval"):
        try:
            user_id, history, target = sample["user_id"], sample["history"], sample["target_item"]
            if len(history) < 1 or model.retriever is None:
                continue
            # Use co-occurrence encoding if available, else P5 encoding
            if model.cooc_embeddings is not None:
                # Encode from TEST history only (avoid data leakage)
                embs = []
                for iid in history[-30:]:
                    e = model.cooc_embeddings.get(int(iid))
                    if e is not None:
                        embs.append(e)
                user_emb = __import__('numpy').mean(embs, axis=0).astype(__import__('numpy').float32) if embs else __import__('numpy').zeros(next(iter(model.cooc_embeddings.values())).shape[0], dtype=__import__('numpy').float32)
            elif model.p5 is not None:
                with torch.no_grad():
                    ids = torch.tensor([int(i) % model.p5.shared.num_embeddings
                                        for i in history[-30:]], device=model.device)
                    user_emb = model.p5.shared(ids).mean(dim=0).cpu().numpy().astype(np.float32)
            else:
                user_emb = None
            result = model.forward(user_id, user_emb, deterministic=True, topk=20,
                                   eval_history=history)
            candidates = result["candidates"]
            action_counts[result["action_name"]] += 1
            target_int = int(target)
            rank = candidates.index(target_int) + 1 if target_int in candidates else float('inf')
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
        return {f'HR@{k}': 0.0 for k in [1, 5, 10]} | {f'NDCG@{k}': 0.0 for k in [5, 10]} | {'count': 0}
    results = {
        'HR@1': round(hr[1] / total, 4), 'HR@5': round(hr[5] / total, 4),
        'HR@10': round(hr[10] / total, 4), 'NDCG@5': round(ndcg[5] / total, 4),
        'NDCG@10': round(ndcg[10] / total, 4), 'count': total,
    }
    total_actions = sum(action_counts.values())
    if total_actions > 0:
        results['action_dist'] = {k: round(v / total_actions, 3)
                                  for k, v in sorted(action_counts.items(), key=lambda x: -x[1])[:5]}
    return results


# ──────────────────────────────────────────
#  Main
# ──────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RL+Memory Ablation Experiments")
    p.add_argument('--dataset', type=str, default='beauty')
    p.add_argument('--backbone', type=str, default='t5-small')
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--sample_ratio', type=float, default=1.0,
                   help='Fraction of training data (data is pre-filtered, use 1.0)')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--beam_size', type=int, default=20)
    p.add_argument('--max_eval', type=int, default=200)
    p.add_argument('--output_dir', type=str, default='outputs/ablation')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--variants', type=str, default='all',
                   help='Comma-separated: full_system,no_memory,short_term_only,no_time_decay,fixed_epsilon or "all"')
    p.add_argument('--p5_checkpoint', type=str, default=None,
                   help='Path to trained P5 checkpoint (required for meaningful results)')
    p.add_argument('--skip_p5', action='store_true', help='Skip P5 eval (only RL variants)')
    p.add_argument('--cooc_embeddings', type=str, default=None,
                   help='Path to co-occurrence embeddings pickle file (overrides P5 embeddings)')
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    timestamp = datetime.now().strftime('%b%d_%H-%M')
    run_name = f"ablation-{args.dataset}-sample{args.sample_ratio}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # ── Load P5 backbone ──
    print("\n" + "=" * 60)
    print("Loading P5 backbone...")
    from modeling_p5 import P5
    from tokenization import P5Tokenizer
    from transformers import T5Config

    # Set HF mirror for China access
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    t5config = T5Config.from_pretrained(args.backbone)
    t5config.losses = 'rating,sequential'
    tokenizer = P5Tokenizer.from_pretrained(args.backbone, max_length=512, do_lower_case=True)
    p5_model = P5.from_pretrained(args.backbone, config=t5config)
    p5_model.resize_token_embeddings(tokenizer.vocab_size)
    if hasattr(p5_model.encoder, 'whole_word_embeddings'):
        p5_model.encoder.whole_word_embeddings.weight.data.normal_(mean=0.0, std=1.0)

    # Load trained checkpoint if provided
    if args.p5_checkpoint and os.path.exists(args.p5_checkpoint):
        print(f"Loading P5 weights from {args.p5_checkpoint}...")
        ckpt = torch.load(args.p5_checkpoint, map_location=device)
        state_dict = ckpt.get('model', ckpt)
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state[k] = v
        missing, unexpected = p5_model.load_state_dict(new_state, strict=False)
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if len(missing) > 10:
            print(f"  First 10 missing keys: {missing[:10]}")
    elif args.p5_checkpoint:
        print(f"  WARNING: Checkpoint not found at {args.p5_checkpoint}")
        print("  Using untrained backbone — results will be meaningless!")
    else:
        print("  WARNING: No checkpoint provided, using untrained backbone!")

    p5_model = p5_model.to(device)
    p5_model.eval()

    # ── Data ──
    print("\nLoading data...")
    data_bridge = SequentialDataBridge(args.data_dir, args.dataset, p5_model=p5_model, device=device)

    if args.cooc_embeddings and os.path.exists(args.cooc_embeddings):
        print(f"Loading co-occurrence embeddings from {args.cooc_embeddings}...")
        with open(args.cooc_embeddings, 'rb') as f:
            item_embs = __import__('pickle').load(f)
        print(f"  Loaded {len(item_embs)} item embeddings")
    else:
        cooc_embeddings = None
    if args.cooc_embeddings and os.path.exists(args.cooc_embeddings):
        print(f"Loading co-occurrence embeddings from {args.cooc_embeddings}...")
        with open(args.cooc_embeddings, 'rb') as f:
            cooc_embeddings = __import__('pickle').load(f)
        print(f"  Loaded {len(cooc_embeddings)} item embeddings")
        item_embs = cooc_embeddings
        # Patch encode_user_history
        _orig = data_bridge.encode_user_history
        def _cooc_encode(item_seq):
            if len(item_seq) == 0:
                return np.zeros(next(iter(cooc_embeddings.values())).shape[0], dtype=np.float32)
            embs = []
            for iid in item_seq[-50:]:
                e = cooc_embeddings.get(int(iid))
                if e is not None:
                    embs.append(e)
            if embs:
                return np.mean(embs, axis=0).astype(np.float32)
            return np.zeros(next(iter(cooc_embeddings.values())).shape[0], dtype=np.float32)
        data_bridge.encode_user_history = _cooc_encode
    else:
        item_embs = data_bridge.get_all_item_embs(max_items=5000)
    retriever = CandidateRetriever(item_embs=item_embs, user_sequences=data_bridge.user_sequences, topk=20)

    train_samples = data_bridge.get_train_data(sample_ratio=args.sample_ratio)
    test_samples = data_bridge.get_test_data(max_users=args.max_eval)
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    config = get_3060_config(args.sample_ratio)
    config.rl.batch_size = args.batch_size
    config.training.total_steps = max(500, args.epochs * 1000)

    # ── Determine variants ──
    if args.variants == 'all':
        variant_names = ['full_system', 'no_memory', 'short_term_only', 'no_time_decay', 'fixed_epsilon']
    else:
        variant_names = [v.strip() for v in args.variants.split(',')]

    print("\n" + "=" * 60)
    print(f"Running {len(variant_names)} ablation variants:")
    for v in variant_names:
        print(f"  - {v}: {ABLATION_VARIANTS[v]['desc']}")
    print("=" * 60)

    all_results = {}

    # ── P5 Baseline ──
    if not args.skip_p5:
        print("\n" + "=" * 60)
        print("Evaluating P5 Baseline")
        print("=" * 60)
        p5_results = evaluate_p5_baseline(p5_model, tokenizer, test_samples,
                                          data_bridge.all_items, device,
                                          beam_size=args.beam_size, max_eval=args.max_eval)
        all_results['p5_baseline'] = p5_results
        print(f"  P5 Baseline: HR@5={p5_results['HR@5']:.4f}, HR@10={p5_results['HR@10']:.4f}, "
              f"NDCG@10={p5_results['NDCG@10']:.4f}")

    # ── Run each ablation variant ──
    for variant_name in variant_names:
        print("\n" + "=" * 60)
        print(f"Ablation Variant: {ABLATION_VARIANTS[variant_name]['name']}")
        print(f"Config: {ABLATION_VARIANTS[variant_name]['desc']}")
        print("=" * 60)

        # Build model
        model = AblationRLModel(p5_model, config, data_bridge.user_sequences,
                                variant_name, device=device, cooc_embeddings=cooc_embeddings)
        model.set_retriever(retriever)

        # Warm up memory
        for sample in train_samples[:min(1000, len(train_samples))]:
            model.add_to_memory(sample["user_id"], sample["target_item"],
                                action_type=1, rating=0.7)

        # Train
        train_ablation_model(model, data_bridge, config, train_samples)

        # Evaluate
        results = evaluate_rl_model(model, test_samples, max_eval=args.max_eval)
        all_results[variant_name] = results
        print(f"  {ABLATION_VARIANTS[variant_name]['name']}: "
              f"HR@5={results['HR@5']:.4f}, HR@10={results['HR@10']:.4f}, "
              f"NDCG@10={results['NDCG@10']:.4f}")

        # Free memory
        del model
        torch.cuda.empty_cache()

    # ── Print comparison table ──
    print("\n" + "=" * 80)
    print("  ABLATION STUDY RESULTS")
    print("=" * 80)
    variants_display = ['p5_baseline'] + variant_names if not args.skip_p5 else variant_names

    metrics = ['HR@1', 'HR@5', 'HR@10', 'NDCG@5', 'NDCG@10']

    header = f"  {'Variant':<30s}"
    for m in metrics:
        header += f" {m:>8s}"
    header += f" {'Count':>6s}"
    print(header)
    print("  " + "-" * (30 + 8 * len(metrics) + 8))

    p5_base = all_results.get('p5_baseline', {})
    for v in variants_display:
        r = all_results.get(v, {})
        label = ABLATION_VARIANTS.get(v, {}).get('name', v.replace('_', ' ').title())
        row = f"  {label:<30s}"
        for m in metrics:
            val = r.get(m, 0)
            if v != 'p5_baseline' and p5_base and p5_base.get(m, 0) > 0:
                diff = val - p5_base[m]
                sign = "+" if diff >= 0 else ""
                row += f" {val:>7.4f}{sign}"
            else:
                row += f" {val:>8.4f}"
        row += f" {r.get('count', 0):>6d}"
        print(row)

    print("=" * 80)

    # ── Action distribution summary ──
    print("\n  Action Distribution (top variant):")
    for v in variant_names:
        r = all_results.get(v, {})
        if 'action_dist' in r:
            print(f"  {ABLATION_VARIANTS[v]['name']}:")
            for act, prob in r['action_dist'].items():
                print(f"    {act}: {prob:.1%}")
            break

    # ── Save ──
    results_path = output_dir / "ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "config": {
                "dataset": args.dataset, "sample_ratio": args.sample_ratio,
                "epochs": args.epochs, "max_eval": args.max_eval,
            },
            "variants": {v: ABLATION_VARIANTS[v] for v in variant_names},
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    return str(output_dir)


if __name__ == '__main__':
    main()
