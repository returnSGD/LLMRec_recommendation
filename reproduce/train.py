"""
Single-GPU P5 Training Script (adapted from baseline_model multi-GPU DDP).
Usage: python train.py --dataset beauty --backbone t5-small --epochs 10

Key changes from original:
  - Removed all DDP (DistributedDataParallel, dist.barrier, reduce_dict)
  - Single GPU training with AMP mixed precision
  - Maintains same training logic and hyperparameters as paper
"""
import os
import sys
import time
import argparse
import random
from pathlib import Path
from datetime import datetime
from packaging import version

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Use local copies of baseline_model/src files
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modeling_p5 import P5
from tokenization import P5Tokenizer
from pretrain_data import get_loader
from utils import LossMeter

_use_native_amp = False
if version.parse(torch.__version__) >= version.parse("1.6"):
    _use_native_amp = True
    from torch.cuda.amp import autocast, GradScaler


class P5Pretraining(P5):
    """P5 model with multi-task training step (same as baseline)."""

    def __init__(self, config):
        super().__init__(config)
        self.losses = self.config.losses.split(',')

    def train_step(self, batch):
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        whole_word_ids = batch['whole_word_ids'].to(device)
        lm_labels = batch["target_ids"].to(device)
        loss_weights = batch["loss_weights"].to(device)

        output = self(
            input_ids=input_ids,
            whole_word_ids=whole_word_ids,
            labels=lm_labels,
            return_dict=True
        )
        lm_mask = (lm_labels != -100).float()
        B, L = lm_labels.size()
        loss = output['loss'].view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)

        results = {}
        results['loss'] = (loss * loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}
        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]
        return results

    @torch.no_grad()
    def valid_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        lm_labels = batch["target_ids"].to(device)
        loss_weights = batch["loss_weights"].to(device)

        output = self(input_ids=input_ids, labels=lm_labels, return_dict=True)
        lm_mask = (lm_labels != -100).float()
        B, L = lm_labels.size()
        loss = output['loss'].view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)

        results = {}
        results['loss'] = (loss * loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}
        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1
        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]
        return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beauty',
                        choices=['beauty', 'sports', 'toys', 'yelp'])
    parser.add_argument('--backbone', type=str, default='t5-small',
                        choices=['t5-small', 't5-base'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--max_text_length', type=int, default=512)
    parser.add_argument('--gen_max_length', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use FP16 mixed precision')
    parser.add_argument('--no_fp16', action='store_true',
                        help='Disable FP16 mixed precision')
    parser.add_argument('--whole_word_embed', action='store_true', default=True,
                        help='Use whole-word embeddings (P5 paper default)')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='Fraction of training data to use (e.g. 0.05 for 5%%)')
    parser.add_argument('--losses', type=str,
                        default='rating,sequential,explanation,review,traditional')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_fp16:
        args.fp16 = False

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Output dir
    timestamp = datetime.now().strftime('%b%d_%H-%M')
    run_name = f"{args.dataset}-{args.backbone.replace('t5-','')}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Tokenizer
    tokenizer = P5Tokenizer.from_pretrained(
        args.backbone,
        max_length=args.max_text_length,
        do_lower_case=True,
    )

    # Loss names
    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')] + ['total_loss']

    # ---- Data Loaders ----
    # Task lists (same as original pretrain.py)
    if args.dataset == 'yelp':
        train_task_list = {
            'rating': ['1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9'],
            'sequential': ['2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9',
                          '2-10','2-11','2-12'],
            'explanation': ['3-1','3-2','3-3','3-4','3-5','3-6','3-7','3-8','3-9'],
            'review': ['4-1','4-2'],
            'traditional': ['5-1','5-2','5-3','5-4','5-5','5-6','5-7']
        }
    else:
        train_task_list = {
            'rating': ['1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9'],
            'sequential': ['2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9',
                          '2-10','2-11','2-12'],
            'explanation': ['3-1','3-2','3-3','3-4','3-5','3-6','3-7','3-8','3-9',
                           '3-10','3-11'],
            'review': ['4-1','4-2','4-3'],
            'traditional': ['5-1','5-2','5-3','5-4','5-5','5-6','5-7']
        }

    train_sample_numbers = {'rating': 1, 'sequential': (5, 5, 10),
                           'explanation': 1, 'review': 1, 'traditional': (10, 5)}

    val_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1),
                         'explanation': 1, 'review': 1, 'traditional': (1, 1)}

    assert args.whole_word_embed, "P5 requires --whole_word_embed"

    print("Loading training data...")
    train_loader = get_loader(
        argparse.Namespace(
            backbone=args.backbone, max_text_length=args.max_text_length,
            gen_max_length=args.gen_max_length, do_lower_case=True,
            tokenizer='p5',
        ),
        train_task_list, train_sample_numbers,
        split=args.dataset, mode='train',
        batch_size=args.batch_size, workers=args.num_workers,
        distributed=False,
        sample_ratio=args.sample_ratio,
    )

    print("Loading validation data...")
    if args.dataset == 'yelp':
        val_task_list = train_task_list
    else:
        val_task_list = {
            'rating': ['1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9'],
            'sequential': ['2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9',
                          '2-10','2-11','2-12'],
            'explanation': ['3-1','3-2','3-3','3-4','3-5','3-6','3-7','3-8','3-9',
                           '3-10','3-11'],
            'review': ['4-1','4-2','4-3'],
            'traditional': ['5-1','5-2','5-3','5-4','5-5','5-6','5-7']
        }

    val_loader = get_loader(
        argparse.Namespace(
            backbone=args.backbone, max_text_length=args.max_text_length,
            gen_max_length=args.gen_max_length, do_lower_case=True,
            tokenizer='p5',
        ),
        val_task_list, val_sample_numbers,
        split=args.dataset, mode='val',
        batch_size=args.batch_size, workers=args.num_workers,
        distributed=False,
        sample_ratio=1.0,  # never sample validation set
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---- Model ----
    from transformers import T5Config
    config = T5Config.from_pretrained(args.backbone)
    config.dropout_rate = args.dropout
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    config.activation_dropout = args.dropout
    config.losses = args.losses

    model = P5Pretraining.from_pretrained(args.backbone, config=config)
    model.resize_token_embeddings(tokenizer.vocab_size)
    model.tokenizer = tokenizer
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}: {args.resume}")

    # ---- Optimizer & Scheduler ----
    from torch.optim import AdamW
    from transformers.optimization import get_linear_schedule_with_warmup

    batch_per_epoch = len(train_loader)
    t_total = batch_per_epoch * args.epochs
    warmup_iters = int(t_total * args.warmup_ratio)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_iters, t_total)

    scaler = GradScaler() if (args.fp16 and _use_native_amp) else None

    print(f"Training: {t_total} steps, {warmup_iters} warmup, {args.epochs} epochs")

    # ---- Training Loop ----
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_results = {name: 0.0 for name in LOSSES_NAME}
        for name in LOSSES_NAME:
            epoch_results[f'{name}_count'] = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+args.epochs}",
                    ncols=150)
        loss_meters = {name: LossMeter() for name in LOSSES_NAME}

        for step, batch in enumerate(pbar):
            if args.fp16 and scaler:
                with autocast():
                    results = model.train_step(batch)
            else:
                results = model.train_step(batch)

            loss = results['loss']

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

            scheduler.step()
            model.zero_grad()
            global_step += 1

            for k, v in results.items():
                if k in epoch_results:
                    if isinstance(v, (int, float)):
                        epoch_results[k] += v
                    elif isinstance(v, torch.Tensor):
                        epoch_results[k] += v.item()

            # Update progress bar
            if step % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                desc_parts = [f"LR {lr:.2e}"]
                for name in LOSSES_NAME:
                    if name[:-5] in results and f"{name}_count" in results:
                        cnt = results[f"{name}_count"]
                        if cnt > 0:
                            loss_meters[name].update(
                                results[name[:-5]] / cnt if isinstance(results[name[:-5]], torch.Tensor)
                                else results[f"{name}_loss"] / cnt
                            )
                desc_parts.append(
                    f"Loss {loss_meters['total_loss'].val:.3f}"
                    if len(loss_meters['total_loss']) > 0 else ""
                )
                pbar.set_postfix_str(" | ".join(p for p in desc_parts if p))

        # Epoch summary
        train_loss = epoch_results['total_loss'] / max(epoch_results['total_loss_count'], 1)
        print(f"\nEpoch {epoch+1} train_loss: {train_loss:.4f}")

        # Save checkpoint every epoch
        ckpt_path = output_dir / f"Epoch{epoch+1:02d}.pth"
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'epoch': epoch + 1,
            'global_step': global_step,
            'args': vars(args),
        }, ckpt_path)
        print(f"Saved: {ckpt_path}")

        # Validation (skip first 10 epochs to match original behavior)
        if epoch >= 10:
            model.eval()
            val_results = {name: 0.0 for name in LOSSES_NAME}
            for name in LOSSES_NAME:
                val_results[f'{name}_count'] = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating", ncols=120):
                    results = model.valid_step(batch)
                    for k, v in results.items():
                        if k in val_results:
                            if isinstance(v, (int, float)):
                                val_results[k] += v
                            elif isinstance(v, torch.Tensor):
                                val_results[k] += v.item()

            val_loss = val_results['total_loss'] / max(val_results['total_loss_count'], 1)
            print(f"Validation loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / "BEST_EVAL_LOSS.pth"
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                }, best_path)
                print(f"New best! Saved: {best_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Outputs: {output_dir}")
    return str(output_dir)


if __name__ == '__main__':
    main()
