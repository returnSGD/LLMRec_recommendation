"""
Quick validation script to find the best checkpoint among saved epochs.
Usage: python validate_checkpoints.py --dataset beauty --backbone <path> --checkpoint_dir <path>
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import numpy as np
from tqdm import tqdm

from modeling_p5 import P5
from tokenization import P5Tokenizer
from pretrain_data import get_loader

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='beauty')
    p.add_argument('--backbone', type=str, required=True)
    p.add_argument('--checkpoint_dir', type=str, required=True)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_text_length', type=int, default=512)
    p.add_argument('--gen_max_length', type=int, default=64)
    p.add_argument('--max_val_batches', type=int, default=200,
                   help='Max validation batches per checkpoint')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Tokenizer
    tokenizer = P5Tokenizer.from_pretrained(args.backbone, max_length=args.max_text_length, do_lower_case=True)

    # Validation data (same as train.py)
    val_task_list = {
        'rating': ['1-1','1-2','1-3','1-4','1-5','1-6','1-7','1-8','1-9'],
        'sequential': ['2-1','2-2','2-3','2-4','2-5','2-6','2-7','2-8','2-9',
                      '2-10','2-11','2-12'],
        'explanation': ['3-1','3-2','3-3','3-4','3-5','3-6','3-7','3-8','3-9',
                       '3-10','3-11'],
        'review': ['4-1','4-2','4-3'],
        'traditional': ['5-1','5-2','5-3','5-4','5-5','5-6','5-7']
    }
    val_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1),
                         'explanation': 1, 'review': 1, 'traditional': (1, 1)}

    print("Loading validation data...")
    val_loader = get_loader(
        argparse.Namespace(
            backbone=args.backbone, max_text_length=args.max_text_length,
            gen_max_length=args.gen_max_length, do_lower_case=True,
            tokenizer='p5',
        ),
        val_task_list, val_sample_numbers,
        split=args.dataset, mode='val',
        batch_size=args.batch_size, workers=4,
        distributed=False, sample_ratio=1.0,
    )
    print(f"Val batches: {len(val_loader)}")

    # Find checkpoints
    ckpt_dir = Path(args.checkpoint_dir)
    ckpts = sorted(ckpt_dir.glob("Epoch*.pth"))
    print(f"Found {len(ckpts)} checkpoints: {[c.name for c in ckpts]}")

    from train import P5Pretraining
    from transformers import T5Config

    results = {}
    for ckpt_path in ckpts:
        print(f"\n{'='*50}")
        print(f"Evaluating: {ckpt_path.name}")

        config = T5Config.from_pretrained(args.backbone)
        config.dropout_rate = 0.0
        config.dropout = 0.0
        config.attention_dropout = 0.0
        config.activation_dropout = 0.0
        config.losses = 'rating,sequential,explanation,review,traditional'

        model = P5Pretraining.from_pretrained(args.backbone, config=config)
        model.resize_token_embeddings(tokenizer.vocab_size)

        # Fix whole_word_embeddings
        if hasattr(model.encoder, 'whole_word_embeddings'):
            model.encoder.whole_word_embeddings.weight.data.normal_(mean=0.0, std=1.0)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        model = model.to(device)
        model.eval()

        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Val {ckpt_path.name}", ncols=100)):
                if args.max_val_batches > 0 and i >= args.max_val_batches:
                    break
                device_batch = {
                    'input_ids': batch['input_ids'].to(device),
                    'whole_word_ids': batch['whole_word_ids'].to(device),
                    'target_ids': batch['target_ids'].to(device),
                    'loss_weights': batch['loss_weights'].to(device),
                }
                output = model(
                    input_ids=device_batch['input_ids'],
                    whole_word_ids=device_batch['whole_word_ids'],
                    labels=device_batch['target_ids'],
                    return_dict=True,
                )
                lm_labels = device_batch['target_ids']
                lm_mask = (lm_labels != -100).float()
                B, L = lm_labels.size()
                loss = output['loss'].view(B, L) * lm_mask
                loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)
                total_loss += (loss * device_batch['loss_weights']).sum().item()
                total_count += len(loss)

        avg_loss = total_loss / max(total_count, 1)
        results[ckpt_path.name] = avg_loss
        print(f"  Val loss: {avg_loss:.6f}")

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print("Summary:")
    for name, loss in sorted(results.items(), key=lambda x: x[1]):
        marker = " <-- BEST" if loss == min(results.values()) else ""
        print(f"  {name}: {loss:.6f}{marker}")

    best = min(results, key=results.get)
    print(f"\nBest checkpoint: {best} (loss={results[best]:.6f})")

    # Save results
    import json
    with open(str(ckpt_dir / "validation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {ckpt_dir / 'validation_results.json'}")

if __name__ == '__main__':
    main()
