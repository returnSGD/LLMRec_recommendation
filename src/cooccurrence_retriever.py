"""
Co-occurrence based item embeddings and retriever.

Builds meaningful item embeddings from user interaction sequences without
requiring a trained P5 model. Uses item-item co-occurrence counts + SVD
to produce embeddings that capture genuine purchase patterns.

This replaces P5's random-ish embeddings (from under-trained checkpoints)
with statistically grounded representations, enabling BC to learn
meaningful policy labels instead of collapsing to a single action.
"""
import numpy as np
from collections import defaultdict
from typing import Dict, List
from tqdm import tqdm


def build_cooccurrence_embeddings(
    user_sequences: Dict[str, List[int]],
    embedding_dim: int = 128,
    window_size: int = 5,
) -> Dict[int, np.ndarray]:
    """
    Build item embeddings from co-occurrence statistics.

    1. Count item-item co-occurrences within sliding windows of user sequences
    2. Normalize to PPMI (Positive Pointwise Mutual Information)
    3. Reduce via truncated SVD to embedding_dim

    Returns dict: item_id -> np.ndarray of shape (embedding_dim,)
    """
    # Collect all items
    all_items = set()
    for seq in user_sequences.values():
        all_items.update(seq)
    item_list = sorted(all_items)
    item2idx = {item: i for i, item in enumerate(item_list)}
    n_items = len(item_list)
    print(f"  Items: {n_items}, building co-occurrence matrix...")

    # Build co-occurrence counts
    cooc = defaultdict(lambda: defaultdict(int))
    item_freq = defaultdict(int)

    for seq in tqdm(user_sequences.values(), desc="  Counting co-occurrences"):
        for i, item in enumerate(seq):
            item_freq[item] += 1
            window_start = max(0, i - window_size)
            window_end = min(len(seq), i + window_size + 1)
            for j in range(window_start, window_end):
                if i != j:
                    cooc[item][seq[j]] += 1

    # Convert to dense matrix (memory-efficient: only top items)
    # Use top 5000 items by frequency for manageable matrix size
    top_items = sorted(item_freq, key=item_freq.get, reverse=True)[:5000]
    top_idx = {item: i for i, item in enumerate(top_items)}
    n = len(top_items)
    print(f"  Using top {n} items by frequency")

    # Build sparse co-occurrence matrix
    from scipy.sparse import lil_matrix
    mat = lil_matrix((n, n), dtype=np.float32)

    total_cooc = sum(sum(d.values()) for d in cooc.values())
    for item_i in tqdm(top_items, desc="  Building matrix"):
        i = top_idx[item_i]
        for item_j, count in cooc[item_i].items():
            if item_j in top_idx:
                j = top_idx[item_j]
                # PPMI: max(0, log(P(i,j) / (P(i)*P(j))))
                p_ij = count / max(total_cooc, 1)
                p_i = item_freq[item_i] / sum(item_freq.values())
                p_j = item_freq[item_j] / sum(item_freq.values())
                ppmi = max(0.0, np.log(p_ij / max(p_i * p_j, 1e-10)))
                mat[i, j] = ppmi

    # SVD reduction
    from scipy.sparse.linalg import svds
    print(f"  Running SVD (dim={embedding_dim})...")
    mat_csr = mat.tocsr()
    U, S, Vt = svds(mat_csr, k=embedding_dim, which='LM')
    # Use U * sqrt(S) as embeddings
    embeddings = U * np.sqrt(S)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    embeddings = embeddings / norms

    # Build result dict
    result = {}
    for i, item in enumerate(top_items):
        result[int(item)] = embeddings[i].astype(np.float32)

    print(f"  Built {len(result)} item embeddings, dim={embedding_dim}")
    return result


def build_simple_cooccurrence_embeddings(
    user_sequences: Dict[str, List[int]],
    embedding_dim: int = 128,
) -> Dict[int, np.ndarray]:
    """
    Simpler fallback: use item frequency + neighbor averaging.

    For each item, its embedding is the average frequency-weighted vector
    of its co-occurring neighbors. Much faster than full SVD for large datasets.
    """
    all_items = set()
    for seq in user_sequences.values():
        all_items.update(seq)
    item_list = sorted(all_items)
    n_items = len(item_list)
    print(f"  Items: {n_items}")

    # Count item frequencies and co-occurrences
    item_freq = defaultdict(int)
    cooc = defaultdict(lambda: defaultdict(float))

    for seq in tqdm(user_sequences.values(), desc="  Building co-occurrence"):
        for i, item in enumerate(seq):
            item_freq[item] += 1
            for j in range(max(0, i - 3), min(len(seq), i + 4)):
                if i != j:
                    cooc[item][seq[j]] += 1.0 / abs(i - j)  # distance-weighted

    # Build embeddings for top items by frequency
    top_items = sorted(item_freq, key=item_freq.get, reverse=True)[:5000]
    n_top = len(top_items)
    print(f"  Using top {n_top} items")

    # Each item gets a random base, then we average neighbor vectors
    rng = np.random.RandomState(42)
    base_emb = rng.randn(n_top, embedding_dim).astype(np.float32) * 0.01

    # Iteratively smooth: each item's emb = weighted avg of neighbor embs
    for iteration in range(10):
        new_emb = np.zeros_like(base_emb)
        for i, item in enumerate(tqdm(top_items, desc=f"  Iteration {iteration+1}/10",
                                      leave=False)):
            neighbors = cooc[item]
            if not neighbors:
                new_emb[i] = base_emb[i]
                continue
            total_w = 0.0
            for neighbor, weight in neighbors.items():
                if neighbor in top_items:
                    j = top_items.index(neighbor)
                    new_emb[i] += weight * base_emb[j]
                    total_w += weight
            if total_w > 0:
                new_emb[i] /= total_w
            else:
                new_emb[i] = base_emb[i]

        base_emb = new_emb
        # Normalize
        norms = np.linalg.norm(base_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        base_emb = base_emb / norms

    result = {}
    for i, item in enumerate(top_items):
        result[int(item)] = base_emb[i].astype(np.float32)

    print(f"  Built {len(result)} item embeddings, dim={embedding_dim}")
    return result
