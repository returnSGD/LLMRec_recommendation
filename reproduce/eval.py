"""
P5 Full Evaluation Script — All 5 Task Families + Zero-shot Transfer.

Evaluates a trained P5 checkpoint on all tasks following the paper's
exact protocols (Tables 2-7 in P5 RecSys 2022).

Usage:
    python eval.py --checkpoint output/beauty-small_xxx/BEST_EVAL_LOSS.pth \
                   --dataset beauty --backbone t5-small

Protocols by task:
  Rating:        Greedy decode → parse float → RMSE/MAE
  Sequential:    Beam search B=20 → all-item ranking → HR@k, NDCG@k
  Explanation:   Greedy decode → BLEU-4, ROUGE-1/2/L
  Review:        Greedy decode → BLEU/ROUGE (summ) + RMSE/MAE (rating)
  Direct:        P("yes") ranking (discriminative) + Beam search (generative) → HR@k, NDCG@k
  Zero-shot:     Cross-domain transfer evaluation (Z-1 ~ Z-7)
"""
import os
import sys
import json
import gzip
import pickle
import argparse
import time
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Use local copies of baseline_model/src files
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modeling_p5 import P5
from tokenization import P5Tokenizer
from all_amazon_templates import all_tasks as AMAZON_TEMPLATES
try:
    from all_yelp_templates import all_tasks as YELP_TEMPLATES
except ImportError:
    YELP_TEMPLATES = None

# ── Constants ───────────────────────────────────────────
PROMPTS = {
    # (task_family, prompt_id, type): seen or unseen per paper Section 5.3
    "rating_seen":     ("1-6", "rating", "seen"),
    "rating_unseen":   ("1-10", "rating", "unseen"),
    "seq_seen":        ("2-3", "sequential", "seen"),
    "seq_unseen":      ("2-13", "sequential", "unseen"),
    "exp_direct":      ("3-3", "explanation", "seen"),
    "exp_feat_seen":   ("3-9", "explanation", "seen"),
    "exp_feat_unseen": ("3-12", "explanation", "unseen"),
    "review_summ":     ("4-1", "review", "seen"),
    "review_rating_seen":   ("4-2", "review", "seen"),
    "review_rating_unseen": ("4-4", "review", "unseen"),
    "direct_disc_seen":     ("5-1", "traditional", "seen"),
    "direct_disc_unseen":   ("5-4", "traditional", "unseen"),
    "direct_gen_seen":      ("5-5", "traditional", "seen"),
    "direct_gen_unseen":    ("5-8", "traditional", "unseen"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to trained P5 checkpoint (.pth)')
    p.add_argument('--dataset', type=str, default='beauty',
                   choices=['beauty', 'sports', 'toys', 'yelp'])
    p.add_argument('--backbone', type=str, default='t5-small')
    p.add_argument('--beam_size', type=int, default=20,
                   help='Beam size for sequential/direct (paper uses 20)')
    p.add_argument('--batch_size', type=int, default=8,
                   help='Evaluation batch size (smaller for beam search)')
    p.add_argument('--max_text_length', type=int, default=512)
    p.add_argument('--gen_max_length', type=int, default=64)
    p.add_argument('--output_file', type=str, default=None,
                   help='JSON file to save results (default: auto-generated)')
    p.add_argument('--tasks', type=str, default='all',
                   help='Comma-separated: rating,sequential,explanation,review,direct,zeroshot')
    p.add_argument('--zero_shot_target', type=str, default=None,
                   help='Target dataset for zero-shot (e.g., toys). If set, evaluates on this dataset.')
    p.add_argument('--zero_shot_source', type=str, default=None,
                   help='Source dataset the model was trained on (for Z-shot prompts)')
    return p.parse_args()


# ── Data Helpers ────────────────────────────────────────
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def read_lines(path):
    with open(path, 'r') as f:
        return [l.rstrip('\n') for l in f]


def load_dataset(dataset: str, mode: str = 'test'):
    """Load all data splits for a dataset."""
    base = os.path.join('data', dataset)
    review = load_pickle(os.path.join(base, 'review_splits.pkl'))[mode]
    exp = load_pickle(os.path.join(base, 'exp_splits.pkl'))[mode]
    try:
        rating = load_pickle(os.path.join(base, 'rating_splits_augmented.pkl'))[mode]
    except FileNotFoundError:
        rating = review
    sequential = read_lines(os.path.join(base, 'sequential_data.txt'))
    negative = read_lines(os.path.join(base, 'negative_samples.txt'))
    datamaps = load_json(os.path.join(base, 'datamaps.json'))
    user_id2name = load_pickle(os.path.join(base, 'user_id2name.pkl'))

    # Parse meta
    meta_dict = {}
    try:
        for meta in parse_gz(os.path.join(base, 'meta.json.gz')):
            if dataset == 'yelp':
                meta_dict[meta.get('business_id', '')] = meta
            else:
                meta_dict[meta.get('asin', '')] = meta
    except:
        meta_dict = {}

    # Parse sequential data into user → items mapping
    user_items = {}
    for line in sequential:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            user_items[parts[0]] = [int(i) for i in parts[1].split(' ')]

    # All items
    all_items = list(datamaps['item2id'].values())  # These are the item IDs as stored

    return {
        'review': review, 'exp': exp, 'rating': rating,
        'sequential': sequential, 'negative': negative,
        'datamaps': datamaps,
        'user_id2name': user_id2name,
        'meta_dict': meta_dict,
        'user_items': user_items,
        'all_items': all_items,
    }


def parse_gz(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


def get_template(templates, task_family: str, prompt_id: str):
    """Get a specific prompt template."""
    fam = templates[task_family]
    return fam[prompt_id]


def fill_template(template: dict, **kwargs) -> str:
    """Fill a template's source text with values."""
    return template['source'].format(**kwargs)


# ── Evaluation Classes ──────────────────────────────────

class RatingEvaluator:
    """Evaluate rating prediction (RMSE/MAE)."""

    def __init__(self, model, tokenizer, device, data, templates, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.data = data
        self.templates = templates
        self.dataset = dataset

    def evaluate(self, prompt_id: str) -> Dict:
        template = get_template(self.templates, 'rating', prompt_id)
        review_data = self.data['rating']  # test split
        datamaps = self.data['datamaps']
        meta_dict = self.data['meta_dict']

        preds, truths = [], []
        for datum in tqdm(review_data[:min(5000, len(review_data))],
                          desc=f"Rating {prompt_id}"):
            try:
                user_id = datamaps['user2id'].get(datum.get('reviewerID', ''), '0')
                item_id = datamaps['item2id'].get(datum.get('asin', ''), '0')

                if prompt_id == '1-6':
                    user_name = datum.get('reviewerName', user_id)
                    source = template['source'].format(user_name, item_id)
                elif prompt_id == '1-10':
                    asin = datum.get('asin', '')
                    user_name = datum.get('reviewerName', user_id)
                    title = meta_dict.get(asin, {}).get('title', 'unknown title')
                    source = template['source'].format(user_name, title)
                else:
                    continue

                truth = float(datum.get('overall', 3))
                pred = self._generate(source)
                pred_val = self._parse_rating(pred)
                if pred_val is not None:
                    preds.append(pred_val)
                    truths.append(truth)
            except Exception:
                continue

        if not preds:
            return {'RMSE': float('nan'), 'MAE': float('nan'), 'count': 0}

        preds = np.array(preds)
        truths = np.array(truths)
        rmse = np.sqrt(np.mean((preds - truths) ** 2))
        mae = np.mean(np.abs(preds - truths))
        return {'RMSE': round(float(rmse), 4), 'MAE': round(float(mae), 4),
                'count': len(preds)}

    def _generate(self, source: str) -> str:
        input_ids = self.tokenizer.encode(source, truncation=True,
                                           max_length=self.tokenizer.model_max_length)
        input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids=input_tensor,
                                         max_length=self.model.config.task_specific_params.get(
                                             'max_length', 10) if hasattr(self.model.config,
                                             'task_specific_params') else 10,
                                         num_beams=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def _parse_rating(self, text: str) -> Optional[float]:
        try:
            return float(text.split()[0].replace(',', '.'))
        except ValueError:
            nums = re.findall(r'\d+\.?\d*', text)
            return float(nums[0]) if nums else None


class SequentialEvaluator:
    """Evaluate sequential recommendation (HR@k, NDCG@k) — all-item setting."""

    def __init__(self, model, tokenizer, device, data, templates, dataset,
                 beam_size=20):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.data = data
        self.templates = templates
        self.dataset = dataset
        self.beam_size = beam_size

    def evaluate(self, prompt_id: str) -> Dict:
        template = get_template(self.templates, 'sequential', prompt_id)
        sequential_data = self.data['sequential']
        user_id2name = self.data['user_id2name']
        datamaps = self.data['datamaps']

        hr = {1: 0, 5: 0, 10: 0}
        ndcg = {5: 0.0, 10: 0.0}
        total = 0

        # Build item_id → text mapping for beam search output matching
        id2item = datamaps.get('id2item', {})
        item2id = datamaps.get('item2id', {})
        all_item_ids = list(id2item.keys())  # these are the string item IDs used in text

        # For each user, last item is test target (leave-last-out)
        for line in tqdm(sequential_data[:min(5000, len(sequential_data))],
                         desc=f"Sequential {prompt_id}"):
            try:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                user_id = parts[0]
                user_desc = user_id2name.get(user_id, user_id)
                # Test item = last item (paper Section 5.1)
                target_item = parts[-1]
                history = parts[1:-1]

                if len(history) == 0:
                    continue

                # Build source
                if prompt_id in ('2-1', '2-2', '2-3'):
                    source = template['source'].format(user_id, ' , '.join(history))
                elif prompt_id in ('2-4', '2-5', '2-6', '2-13'):
                    source = template['source'].format(user_desc, ' , '.join(history))
                else:
                    continue

                # Beam search
                input_ids = self.tokenizer.encode(source, truncation=True,
                                                   max_length=512)
                input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_tensor,
                        max_length=20,
                        num_beams=self.beam_size,
                        num_return_sequences=min(self.beam_size, 20),
                        early_stopping=True,
                    )

                # Decode beam outputs → match to item IDs
                predicted_items = []
                for output in outputs:
                    text = self.tokenizer.decode(output, skip_special_tokens=True).strip()
                    # Extract potential item ID from generated text
                    item_match = re.findall(r'\b\d+\b', text)
                    for m in item_match:
                        if m not in predicted_items:
                            predicted_items.append(m)
                            break

                # Compute ranking metrics
                if target_item in predicted_items:
                    rank = predicted_items.index(target_item) + 1
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
            except Exception as e:
                continue

        if total == 0:
            return {f'HR@{k}': 0.0 for k in [1, 5, 10]} | \
                   {f'NDCG@{k}': 0.0 for k in [5, 10]} | {'count': 0}

        result = {f'HR@{k}': round(hr[k] / total, 4) for k in [1, 5, 10]}
        result.update({f'NDCG@{k}': round(ndcg[k] / total, 4) for k in [5, 10]})
        result['count'] = total
        return result


class TextGenEvaluator:
    """Evaluate text generation tasks (BLEU, ROUGE)."""

    def __init__(self, model, tokenizer, device, data, templates, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.data = data
        self.templates = templates
        self.dataset = dataset

    def evaluate_explanation(self, prompt_id: str) -> Dict:
        """Evaluate explanation generation."""
        template = get_template(self.templates, 'explanation', prompt_id)
        exp_data = self.data['exp']
        datamaps = self.data['datamaps']
        meta_dict = self.data['meta_dict']

        all_preds, all_refs = [], []
        for datum in tqdm(exp_data[:min(1000, len(exp_data))],
                          desc=f"Explanation {prompt_id}"):
            try:
                user_id = datamaps['user2id'].get(datum.get('reviewerID', ''), '0')
                item_id = datamaps['item2id'].get(datum.get('asin', ''), '0')
                asin = datum.get('asin', '')
                user_name = datum.get('reviewerName', user_id)
                title = meta_dict.get(asin, {}).get('title', 'unknown title')
                overall = int(datum.get('overall', 3))
                feature = datum.get('feature', 'quality')

                if prompt_id == '3-3':
                    source = template['source'].format(user_id, str(overall), title)
                elif prompt_id == '3-9':
                    source = template['source'].format(feature, user_id, title)
                elif prompt_id == '3-12':
                    source = template['source'].format(feature, overall, user_name, item_id)
                else:
                    continue

                target = datum.get('explanation', '')

                pred = self._generate(source)
                all_preds.append(pred)
                all_refs.append(target)
            except Exception:
                continue

        return self._compute_text_metrics(all_preds, all_refs)

    def evaluate_review_summ(self, prompt_id: str) -> Dict:
        """Evaluate review summarization."""
        template = get_template(self.templates, 'review', prompt_id)
        review_data = self.data['review']

        all_preds, all_refs = [], []
        for datum in tqdm(review_data[:min(1000, len(review_data))],
                          desc=f"Review summ {prompt_id}"):
            try:
                user_id = datum.get('reviewerID', '0')
                review_text = datum.get('reviewText', '')
                summary = datum.get('summary', '')

                if not review_text or not summary:
                    continue

                source = template['source'].format(user_id, review_text)
                pred = self._generate(source)
                all_preds.append(pred)
                all_refs.append(summary)
            except Exception:
                continue

        return self._compute_text_metrics(all_preds, all_refs)

    def evaluate_review_rating(self, prompt_id: str) -> Dict:
        """Evaluate rating prediction from review text."""
        template = get_template(self.templates, 'review', prompt_id)
        review_data = self.data['review']

        preds, truths = [], []
        for datum in tqdm(review_data[:min(5000, len(review_data))],
                          desc=f"Review rating {prompt_id}"):
            try:
                user_id = datum.get('reviewerID', '0')
                review_text = datum.get('reviewText', '')
                truth = float(datum.get('overall', 3))
                user_name = datum.get('reviewerName', user_id)

                if not review_text:
                    continue

                if prompt_id == '4-2':
                    source = template['source'].format(user_id, review_text)
                elif prompt_id == '4-4':
                    source = template['source'].format(user_name, review_text)
                else:
                    continue

                pred_text = self._generate(source)
                try:
                    pred_val = float(pred_text.split()[0].replace(',', '.'))
                except (ValueError, IndexError):
                    nums = re.findall(r'\d+\.?\d*', pred_text)
                    pred_val = float(nums[0]) if nums else 3.0

                preds.append(pred_val)
                truths.append(truth)
            except Exception:
                continue

        if not preds:
            return {'RMSE': float('nan'), 'MAE': float('nan'), 'count': 0}

        preds = np.array(preds)
        truths = np.array(truths)
        rmse = np.sqrt(np.mean((preds - truths) ** 2))
        mae = np.mean(np.abs(preds - truths))
        return {'RMSE': round(float(rmse), 4), 'MAE': round(float(mae), 4),
                'count': len(preds)}

    def _generate(self, source: str) -> str:
        input_ids = self.tokenizer.encode(source, truncation=True, max_length=512)
        input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids=input_tensor,
                                         max_length=self.model.config.gen_max_length
                                         if hasattr(self.model.config, 'gen_max_length') else 64,
                                         num_beams=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def _compute_text_metrics(self, preds: List[str], refs: List[str]) -> Dict:
        """Compute BLEU and ROUGE scores."""
        if not preds:
            return {'BLEU-4': 0.0, 'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0,
                    'count': 0}

        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smooth = SmoothingFunction().method1
            bleu_scores = []
            for ref, hyp in zip(refs, preds):
                bleu_scores.append(
                    sentence_bleu([ref.split()], hyp.split(),
                                  weights=(0.25, 0.25, 0.25, 0.25),
                                  smoothing_function=smooth)
                )
            bleu4 = np.mean(bleu_scores) * 100
        except ImportError:
            bleu4 = 0.0

        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'],
                                             use_stemmer=True)
            r1, r2, rl = [], [], []
            for ref, hyp in zip(refs, preds):
                scores = scorer.score(ref, hyp)
                r1.append(scores['rouge1'].fmeasure * 100)
                r2.append(scores['rouge2'].fmeasure * 100)
                rl.append(scores['rougeL'].fmeasure * 100)
            return {
                'BLEU-4': round(float(bleu4), 4),
                'ROUGE-1': round(float(np.mean(r1)), 4),
                'ROUGE-2': round(float(np.mean(r2)), 4),
                'ROUGE-L': round(float(np.mean(rl)), 4),
                'count': len(preds),
            }
        except ImportError:
            return {'BLEU-4': round(float(bleu4), 4),
                    'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0,
                    'count': len(preds)}


class DirectEvaluator:
    """Evaluate direct recommendation (1-out-of-100 candidate pool)."""

    def __init__(self, model, tokenizer, device, data, templates, dataset,
                 beam_size=20):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.data = data
        self.templates = templates
        self.dataset = dataset
        self.beam_size = beam_size

    def evaluate_discriminative(self, prompt_id: str) -> Dict:
        """Evaluate yes/no discriminative prompts (5-1, 5-4)."""
        template = get_template(self.templates, 'traditional', prompt_id)
        sequential_data = self.data['sequential']
        user_items = self.data['user_items']
        negative_data = self.data['negative']
        user_id2name = self.data['user_id2name']
        datamaps = self.data['datamaps']

        hr = {1: 0, 5: 0, 10: 0}
        ndcg = {5: 0.0, 10: 0.0}
        total = 0
        max_users = min(500, len(sequential_data))

        for line in tqdm(sequential_data[:max_users],
                          desc=f"Direct disc {prompt_id}"):
            try:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                user_id = parts[0]
                user_desc = user_id2name.get(user_id, user_id)
                target_item = parts[-1]
                history = user_items.get(user_id, [])

                # Get negative candidates + positive
                candidates = self._get_candidates(user_id, target_item, negative_data)
                if len(candidates) < 2:
                    continue

                # Score each candidate with P("yes")
                scores = {}
                for cand in candidates:
                    if prompt_id == '5-1':
                        source = template['source'].format(user_id, cand)
                    elif prompt_id == '5-4':
                        source = template['source'].format(user_id, cand)
                    else:
                        source = template['source'].format(user_desc, cand)

                    score = self._get_yes_prob(source)
                    scores[cand] = score

                # Rank by score (higher P("yes") = better)
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                ranked_items = [r[0] for r in ranked]

                if target_item in ranked_items:
                    rank = ranked_items.index(target_item) + 1
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

        result = {f'HR@{k}': round(hr[k] / total, 4) for k in [1, 5, 10]}
        result.update({f'NDCG@{k}': round(ndcg[k] / total, 4) for k in [5, 10]})
        result['count'] = total
        return result

    def evaluate_generative(self, prompt_id: str) -> Dict:
        """Evaluate generative prompts (5-5, 5-8) with beam search."""
        template = get_template(self.templates, 'traditional', prompt_id)
        sequential_data = self.data['sequential']
        user_items = self.data['user_items']
        negative_data = self.data['negative']
        user_id2name = self.data['user_id2name']

        hr = {1: 0, 5: 0, 10: 0}
        ndcg = {5: 0.0, 10: 0.0}
        total = 0
        max_users = min(500, len(sequential_data))

        for line in tqdm(sequential_data[:max_users],
                          desc=f"Direct gen {prompt_id}"):
            try:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                user_id = parts[0]
                user_desc = user_id2name.get(user_id, user_id)
                target_item = parts[-1]

                candidates = self._get_candidates(user_id, target_item, negative_data)
                if len(candidates) < 2:
                    continue

                # Build source with candidate list
                cand_str = ' , '.join(map(str, candidates))
                if prompt_id in ('5-5', '5-6'):
                    source = template['source'].format(user_desc, cand_str)
                elif prompt_id in ('5-7', '5-8'):
                    source = template['source'].format(user_id, cand_str)
                else:
                    continue

                # Beam search generation
                input_ids = self.tokenizer.encode(source, truncation=True,
                                                   max_length=512)
                input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_tensor,
                        max_length=20,
                        num_beams=self.beam_size,
                        num_return_sequences=min(self.beam_size, 10),
                        early_stopping=True,
                    )

                predicted_items = []
                for output in outputs:
                    text = self.tokenizer.decode(output, skip_special_tokens=True).strip()
                    item_match = re.findall(r'\d+', text)
                    for m in item_match:
                        if m not in predicted_items:
                            predicted_items.append(m)
                            break

                if target_item in predicted_items:
                    rank = predicted_items.index(target_item) + 1
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

        result = {f'HR@{k}': round(hr[k] / total, 4) for k in [1, 5, 10]}
        result.update({f'NDCG@{k}': round(ndcg[k] / total, 4) for k in [5, 10]})
        result['count'] = total
        return result

    def _get_yes_prob(self, source: str) -> float:
        """Get P('yes') score for a discriminative prompt."""
        input_ids = self.tokenizer.encode(source, truncation=True, max_length=512)
        # Append the target token "yes" or "no" — we want P("yes")
        yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]

        input_tensor = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
        with torch.no_grad():
            decoder_start = torch.LongTensor([[self.tokenizer.pad_token_id]]).to(self.device)
            outputs = self.model(input_ids=input_tensor, decoder_input_ids=decoder_start,
                               return_dict=True)
            logits = outputs.logits[0, 0, :]  # first token logits
            probs = F.softmax(logits, dim=-1)
            return float(probs[yes_id].cpu().item())

    def _get_candidates(self, user_id: str, target_item: str,
                        negative_data: List[str]) -> List[str]:
        """Build 1-positive + 99-negative candidate pool."""
        # Try to use pre-computed negatives
        try:
            user_int = int(user_id) - 1  # user IDs are 1-indexed in data
            if 0 <= user_int < len(negative_data):
                parts = negative_data[user_int].split(' ', 1)
                if len(parts) == 2:
                    negatives = parts[1].split(' ')[:99]
                    pool = negatives + [target_item]
                    import random
                    random.shuffle(pool)
                    return pool
        except (ValueError, IndexError):
            pass

        return [target_item]


# ── Main Evaluation ─────────────────────────────────────

def load_model(checkpoint_path: str, backbone: str, device: str):
    """Load trained P5 model."""
    from transformers import T5Config
    config = T5Config.from_pretrained(backbone)
    config.losses = 'rating,sequential,explanation,review,traditional'
    config.gen_max_length = 64

    tokenizer = P5Tokenizer.from_pretrained(backbone, max_length=512, do_lower_case=True)

    from train import P5Pretraining
    model = P5Pretraining.from_pretrained(backbone, config=config)
    model.resize_token_embeddings(tokenizer.vocab_size)

    # Fix: whole_word_embeddings may be corrupted during from_pretrained loading
    if hasattr(model.encoder, 'whole_word_embeddings'):
        model.encoder.whole_word_embeddings.weight.data.normal_(mean=0.0, std=1.0)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model', ckpt)
    # Handle DDP wrapping
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    model = model.to(device)
    model.eval()
    model.tokenizer = tokenizer

    return model, tokenizer


def run_all_evals(model, tokenizer, device, data, templates, dataset,
                  beam_size=20) -> Dict:
    """Run full evaluation suite."""
    results = {}

    # 1. Rating Prediction
    print("\n" + "=" * 60)
    print("1. Rating Prediction (Table 2)")
    print("=" * 60)
    rater = RatingEvaluator(model, tokenizer, device, data, templates, dataset)
    results['rating_1-6'] = rater.evaluate('1-6')
    print(f"  1-6 (seen): RMSE={results['rating_1-6'].get('RMSE','?')}, "
          f"MAE={results['rating_1-6'].get('MAE','?')}")
    results['rating_1-10'] = rater.evaluate('1-10')
    print(f"  1-10 (unseen): RMSE={results['rating_1-10'].get('RMSE','?')}, "
          f"MAE={results['rating_1-10'].get('MAE','?')}")

    # 2. Sequential Recommendation
    print("\n" + "=" * 60)
    print("2. Sequential Recommendation (Table 3)")
    print("=" * 60)
    seq_eval = SequentialEvaluator(model, tokenizer, device, data, templates,
                                    dataset, beam_size)
    results['seq_2-3'] = seq_eval.evaluate('2-3')
    print(f"  2-3 (seen): HR@5={results['seq_2-3'].get('HR@5','?')}, "
          f"HR@10={results['seq_2-3'].get('HR@10','?')}, "
          f"NDCG@10={results['seq_2-3'].get('NDCG@10','?')}")
    results['seq_2-13'] = seq_eval.evaluate('2-13')
    print(f"  2-13 (unseen): HR@5={results['seq_2-13'].get('HR@5','?')}, "
          f"HR@10={results['seq_2-13'].get('HR@10','?')}, "
          f"NDCG@10={results['seq_2-13'].get('NDCG@10','?')}")

    # 3. Explanation Generation
    print("\n" + "=" * 60)
    print("3. Explanation Generation (Table 4)")
    print("=" * 60)
    txt_eval = TextGenEvaluator(model, tokenizer, device, data, templates, dataset)
    results['exp_3-3'] = txt_eval.evaluate_explanation('3-3')
    print(f"  3-3 (direct): BLEU-4={results['exp_3-3'].get('BLEU-4','?')}, "
          f"ROUGE-1={results['exp_3-3'].get('ROUGE-1','?')}")
    results['exp_3-9'] = txt_eval.evaluate_explanation('3-9')
    print(f"  3-9 (feat, seen): BLEU-4={results['exp_3-9'].get('BLEU-4','?')}, "
          f"ROUGE-1={results['exp_3-9'].get('ROUGE-1','?')}")
    results['exp_3-12'] = txt_eval.evaluate_explanation('3-12')
    print(f"  3-12 (feat, unseen): BLEU-4={results['exp_3-12'].get('BLEU-4','?')}, "
          f"ROUGE-1={results['exp_3-12'].get('ROUGE-1','?')}")

    # 4. Review Related
    print("\n" + "=" * 60)
    print("4. Review Related (Tables 5 & 6)")
    print("=" * 60)
    results['review_summ_4-1'] = txt_eval.evaluate_review_summ('4-1')
    print(f"  4-1 (summ): BLEU-4={results['review_summ_4-1'].get('BLEU-4','?')}, "
          f"ROUGE-1={results['review_summ_4-1'].get('ROUGE-1','?')}")
    results['review_rating_4-2'] = txt_eval.evaluate_review_rating('4-2')
    print(f"  4-2 (rating, seen): RMSE={results['review_rating_4-2'].get('RMSE','?')}, "
          f"MAE={results['review_rating_4-2'].get('MAE','?')}")
    results['review_rating_4-4'] = txt_eval.evaluate_review_rating('4-4')
    print(f"  4-4 (rating, unseen): RMSE={results['review_rating_4-4'].get('RMSE','?')}, "
          f"MAE={results['review_rating_4-4'].get('MAE','?')}")

    # 5. Direct Recommendation
    print("\n" + "=" * 60)
    print("5. Direct Recommendation (Table 7)")
    print("=" * 60)
    direct_eval = DirectEvaluator(model, tokenizer, device, data, templates,
                                   dataset, beam_size)
    results['direct_5-1'] = direct_eval.evaluate_discriminative('5-1')
    print(f"  5-1 (disc, seen): HR@5={results['direct_5-1'].get('HR@5','?')}, "
          f"HR@10={results['direct_5-1'].get('HR@10','?')}")
    results['direct_5-4'] = direct_eval.evaluate_discriminative('5-4')
    print(f"  5-4 (disc, unseen): HR@5={results['direct_5-4'].get('HR@5','?')}, "
          f"HR@10={results['direct_5-4'].get('HR@10','?')}")
    results['direct_5-5'] = direct_eval.evaluate_generative('5-5')
    print(f"  5-5 (gen, seen): HR@5={results['direct_5-5'].get('HR@5','?')}, "
          f"HR@10={results['direct_5-5'].get('HR@10','?')}")
    results['direct_5-8'] = direct_eval.evaluate_generative('5-8')
    print(f"  5-8 (gen, unseen): HR@5={results['direct_5-8'].get('HR@5','?')}, "
          f"HR@10={results['direct_5-8'].get('HR@10','?')}")

    return results


def print_report(results: Dict, dataset: str, backbone: str):
    """Print formatted evaluation report."""
    print("\n\n")
    print("=" * 80)
    print(f"  P5 Reproducibility Report")
    print(f"  Dataset: {dataset}  |  Backbone: {backbone}")
    print("=" * 80)

    # Rating
    print("\n── Rating Prediction (Table 2) ──")
    print(f"  {'':25s} {'RMSE':>8s} {'MAE':>8s}")
    for k, label in [('rating_1-6', '1-6 (seen)'), ('rating_1-10', '1-10 (unseen)')]:
        r = results.get(k, {})
        print(f"  {label:25s} {r.get('RMSE', 'N/A'):>8.4f} {r.get('MAE', 'N/A'):>8.4f}")

    # Sequential
    print("\n── Sequential Recommendation (Table 3) ──")
    print(f"  {'':25s} {'HR@5':>8s} {'HR@10':>8s} {'NDCG@5':>8s} {'NDCG@10':>8s}")
    for k, label in [('seq_2-3', '2-3 (seen)'), ('seq_2-13', '2-13 (unseen)')]:
        r = results.get(k, {})
        print(f"  {label:25s} {r.get('HR@5', 'N/A'):>8.4f} "
              f"{r.get('HR@10', 'N/A'):>8.4f} {r.get('NDCG@5', 'N/A'):>8.4f} "
              f"{r.get('NDCG@10', 'N/A'):>8.4f}")

    # Explanation
    print("\n── Explanation Generation (Table 4) ──")
    print(f"  {'':25s} {'BLEU-4':>8s} {'ROUGE-1':>8s} {'ROUGE-2':>8s} {'ROUGE-L':>8s}")
    for k, label in [('exp_3-3', '3-3 (direct)'), ('exp_3-9', '3-9 (feat, seen)'),
                     ('exp_3-12', '3-12 (feat, unseen)')]:
        r = results.get(k, {})
        print(f"  {label:25s} {r.get('BLEU-4', 'N/A'):>8.4f} "
              f"{r.get('ROUGE-1', 'N/A'):>8.4f} {r.get('ROUGE-2', 'N/A'):>8.4f} "
              f"{r.get('ROUGE-L', 'N/A'):>8.4f}")

    # Review
    print("\n── Review Related (Tables 5 & 6) ──")
    print(f"  {'Review Summarization':>30s} {'BLEU-4':>8s} {'ROUGE-1':>8s} {'ROUGE-L':>8s}")
    r = results.get('review_summ_4-1', {})
    print(f"  {'4-1 (summarization)':>30s} {r.get('BLEU-4', 'N/A'):>8.4f} "
          f"{r.get('ROUGE-1', 'N/A'):>8.4f} {r.get('ROUGE-L', 'N/A'):>8.4f}")

    print(f"\n  {'Review Rating Prediction':>30s} {'RMSE':>8s} {'MAE':>8s}")
    for k, label in [('review_rating_4-2', '4-2 (seen)'),
                     ('review_rating_4-4', '4-4 (unseen)')]:
        r = results.get(k, {})
        print(f"  {label:30s} {r.get('RMSE', 'N/A'):>8.4f} {r.get('MAE', 'N/A'):>8.4f}")

    # Direct
    print("\n── Direct Recommendation (Table 7) ──")
    print(f"  {'':25s} {'HR@1':>8s} {'HR@5':>8s} {'HR@10':>8s} {'NDCG@5':>8s} {'NDCG@10':>8s}")
    for k, label in [('direct_5-1', '5-1 (disc, seen)'),
                     ('direct_5-4', '5-4 (disc, unseen)'),
                     ('direct_5-5', '5-5 (gen, seen)'),
                     ('direct_5-8', '5-8 (gen, unseen)')]:
        r = results.get(k, {})
        print(f"  {label:25s} {r.get('HR@1', 'N/A'):>8.4f} "
              f"{r.get('HR@5', 'N/A'):>8.4f} {r.get('HR@10', 'N/A'):>8.4f} "
              f"{r.get('NDCG@5', 'N/A'):>8.4f} {r.get('NDCG@10', 'N/A'):>8.4f}")

    print("\n" + "=" * 80)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")

    # Load templates
    if args.dataset == 'yelp' and YELP_TEMPLATES:
        templates = YELP_TEMPLATES
    else:
        templates = AMAZON_TEMPLATES

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(args.checkpoint, args.backbone, str(device))
    print("Model loaded.")

    # Load data
    print("Loading evaluation data...")
    data = load_dataset(args.dataset, mode='test')
    print(f"  Reviews: {len(data['review'])}, Explanations: {len(data['exp'])}, "
          f"Users: {len(data['sequential'])}")

    # Run evaluation
    task_filter = set(args.tasks.split(',')) if args.tasks != 'all' else None
    results = {}

    if task_filter is None or 'rating' in task_filter:
        rater = RatingEvaluator(model, tokenizer, device, data, templates, args.dataset)
        for pid, label in [('1-6', 'Rating 1-6 (seen)'), ('1-10', 'Rating 1-10 (unseen)')]:
            print(f"\n--- {label} ---")
            results[f'rating_{pid}'] = rater.evaluate(pid)
            print(results[f'rating_{pid}'])

    if task_filter is None or 'sequential' in task_filter:
        seq_eval = SequentialEvaluator(model, tokenizer, device, data, templates,
                                        args.dataset, args.beam_size)
        for pid, label in [('2-3', 'Seq 2-3 (seen)'), ('2-13', 'Seq 2-13 (unseen)')]:
            print(f"\n--- {label} ---")
            results[f'seq_{pid}'] = seq_eval.evaluate(pid)
            print(results[f'seq_{pid}'])

    if task_filter is None or 'explanation' in task_filter:
        txt_eval = TextGenEvaluator(model, tokenizer, device, data, templates,
                                     args.dataset)
        for pid, label in [('3-3', 'Exp 3-3 (direct)'),
                          ('3-9', 'Exp 3-9 (feat, seen)'),
                          ('3-12', 'Exp 3-12 (feat, unseen)')]:
            print(f"\n--- {label} ---")
            results[f'exp_{pid}'] = txt_eval.evaluate_explanation(pid)
            print(results[f'exp_{pid}'])

    if task_filter is None or 'review' in task_filter:
        txt_eval = TextGenEvaluator(model, tokenizer, device, data, templates,
                                     args.dataset)
        print(f"\n--- Review Summ 4-1 ---")
        results['review_summ_4-1'] = txt_eval.evaluate_review_summ('4-1')
        print(results['review_summ_4-1'])
        for pid, label in [('4-2', 'Review Rating 4-2 (seen)'),
                          ('4-4', 'Review Rating 4-4 (unseen)')]:
            print(f"\n--- {label} ---")
            results[f'review_rating_{pid}'] = txt_eval.evaluate_review_rating(pid)
            print(results[f'review_rating_{pid}'])

    if task_filter is None or 'direct' in task_filter:
        de = DirectEvaluator(model, tokenizer, device, data, templates,
                             args.dataset, args.beam_size)
        for pid, label in [('5-1', 'Direct 5-1 (disc, seen)'),
                          ('5-4', 'Direct 5-4 (disc, unseen)'),
                          ('5-5', 'Direct 5-5 (gen, seen)'),
                          ('5-8', 'Direct 5-8 (gen, unseen)')]:
            print(f"\n--- {label} ---")
            if pid in ('5-1', '5-4'):
                results[f'direct_{pid}'] = de.evaluate_discriminative(pid)
            else:
                results[f'direct_{pid}'] = de.evaluate_generative(pid)
            print(results[f'direct_{pid}'])

    # Print final report
    print_report(results, args.dataset, args.backbone)

    # Save results
    if args.output_file:
        out_path = args.output_file
    else:
        ckpt_name = Path(args.checkpoint).stem
        out_path = f"eval_results_{args.dataset}_{ckpt_name}.json"

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
