"""
Pre-download T5 models before training (with mirror support).

Usage:
    python download_models.py                    # download t5-small and t5-base
    python download_models.py --model t5-small   # download only t5-small
    HF_ENDPOINT=https://hf-mirror.com python download_models.py  # use mirror

Or using modelscope (no HF needed):
    pip install modelscope
    python -c "from modelscope import snapshot_download; \
               snapshot_download('AI-ModelScope/t5-small', cache_dir='./pretrained_models')"
"""
import os
import sys
import argparse


def download_hf(model_name: str, cache_dir: str = None):
    """Download a HuggingFace model via transformers API."""
    from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"Downloading {model_name} from {endpoint}...")
    print(f"  (set HF_ENDPOINT=https://hf-mirror.com for mainland China)")

    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    print(f"  Downloading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, **kwargs)
    print(f"  Downloading model config...")
    config = T5Config.from_pretrained(model_name, **kwargs)
    print(f"  Downloading model weights (this may take a few minutes)...")
    model = T5ForConditionalGeneration.from_pretrained(model_name, **kwargs)

    print(f"  {model_name} downloaded successfully!")
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params/1e6:.2f}M")
    return model, tokenizer


def download_via_modelscope(model_name: str, cache_dir: str = "./pretrained_models"):
    """Download model from ModelScope (works in mainland China without VPN)."""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("Error: modelscope not installed. Run: pip install modelscope")
        sys.exit(1)

    # Map HF names to ModelScope names
    ms_map = {
        "t5-small": "AI-ModelScope/t5-small",
        "t5-base": "AI-ModelScope/t5-base",
    }
    ms_name = ms_map.get(model_name, model_name)

    print(f"Downloading {ms_name} from modelscope...")
    local_dir = snapshot_download(ms_name, cache_dir=cache_dir)
    print(f"Downloaded to: {local_dir}")
    return local_dir


def main():
    parser = argparse.ArgumentParser(description="Pre-download T5 models")
    parser.add_argument("--model", type=str, default="all",
                        choices=["t5-small", "t5-base", "all"],
                        help="Model to download (default: all)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for models")
    parser.add_argument("--use_modelscope", action="store_true",
                        help="Use ModelScope instead of HuggingFace")
    args = parser.parse_args()

    models = {
        "t5-small": "t5-small",
        "t5-base": "t5-base",
    }

    if args.model == "all":
        to_download = list(models.values())
    else:
        to_download = [args.model]

    for m in to_download:
        print(f"\n{'='*60}")
        if args.use_modelscope:
            download_via_modelscope(m, args.cache_dir or "./pretrained_models")
        else:
            download_hf(m, args.cache_dir)
        print(f"{'='*60}")

    print("\nAll models downloaded. Ready for training!")


if __name__ == "__main__":
    main()
