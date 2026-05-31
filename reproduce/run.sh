#!/bin/bash
# =============================================================================
# P5 Reproducibility Pipeline
# =============================================================================
# One-click script to reproduce ALL P5 paper tasks and generate a report.
#
# Usage:
#   bash run.sh                                    # Default: P5-Small on Beauty
#   bash run.sh --dataset beauty --backbone t5-small --epochs 10
#   bash run.sh --dataset all --backbone t5-small  # All 3 Amazon datasets
#   bash run.sh --help
#
# Server specs: RTX PRO 6000 (96GB), PyTorch 2.8.0, Python 3.12, CUDA 12.8
# =============================================================================

set -euo pipefail

# ── HuggingFace Mirror Configuration ───────────────────────
# Use HF mirror if in mainland China (set HF_MIRROR=1 or uncomment below)
if [ "${HF_MIRROR:-0}" = "1" ]; then
    export HF_ENDPOINT="https://hf-mirror.com"
    echo "Using HF mirror: $HF_ENDPOINT"
fi
# Alternative: use modelscope to download models
# pip install modelscope
# Then models are auto-downloaded from modelscope.cn

# ── Defaults ──────────────────────────────────────────────
DATASET="beauty"
BACKBONE="t5-small"
EPOCHS=10
BATCH_SIZE=16
LR=1e-3
SEED=2022
TASKS="all"
SKIP_TRAIN=false
SKIP_EVAL=false
OUTPUT_DIR="./output"
EXTRA_ARGS=""

# ── Parse args ────────────────────────────────────────────
show_help() {
    cat << 'EOF'
P5 Reproducibility Pipeline

USAGE:
  bash run.sh [OPTIONS]

OPTIONS:
  --dataset NAME       Dataset: beauty, sports, toys, yelp, or all (default: beauty)
  --backbone NAME      Model: t5-small or t5-base (default: t5-small)
  --epochs N           Training epochs (default: 10)
  --batch_size N       Batch size per GPU (default: 16 for small, 8 for base)
  --lr FLOAT           Peak learning rate (default: 1e-3)
  --seed N             Random seed (default: 2022)
  --output_dir DIR     Output directory (default: ./output)
  --tasks LIST         Eval tasks: all,rating,sequential,explanation,review,direct (default: all)
  --skip_train         Skip training, only evaluate existing checkpoint
  --skip_eval          Skip evaluation, only train
  --resume PATH        Resume from checkpoint
  --fp16               Use FP16 mixed precision (default: auto-detect)
  --no_fp16            Disable FP16
  --help               Show this help

EXAMPLES:
  bash run.sh                                                # P5-Small on Beauty (full pipeline)
  bash run.sh --dataset sports --backbone t5-small           # P5-Small on Sports
  bash run.sh --dataset all --backbone t5-small --epochs 5   # Quick all-dataset run
  bash run.sh --backbone t5-base --batch_size 8              # P5-Base on Beauty
  bash run.sh --skip_train --resume output/xxx/BEST_EVAL_LOSS.pth  # Eval only
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --backbone) BACKBONE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        --skip_train) SKIP_TRAIN=true; shift ;;
        --skip_eval) SKIP_EVAL=true; shift ;;
        --resume) EXTRA_ARGS="$EXTRA_ARGS --resume $2"; shift 2 ;;
        --fp16) EXTRA_ARGS="$EXTRA_ARGS --fp16"; shift ;;
        --no_fp16) EXTRA_ARGS="$EXTRA_ARGS --no_fp16"; shift ;;
        --help) show_help ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Environment checks ─────────────────────────────────────
REPRODUCE_DIR="$(cd "$(dirname "$0")/reproduce" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "  P5 Reproducibility Pipeline"
echo "============================================================"
echo "  Project root:  $PROJECT_ROOT"
echo "  Reproduce dir: $REPRODUCE_DIR"
echo "  Dataset:       $DATASET"
echo "  Backbone:      $BACKBONE"
echo "  Epochs:        $EPOCHS"
echo "  Batch size:    $BATCH_SIZE"
echo "  Output dir:    $OUTPUT_DIR"
echo "  Tasks:         $TASKS"
echo "============================================================"
echo ""

# Check Python and CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || {
    echo "ERROR: PyTorch not found. Install: pip install -r $REPRODUCE_DIR/requirements.txt"
    exit 1
}

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "VRAM: $(python -c "import torch; p=torch.cuda.get_device_properties(0); print(f'{p.total_memory/1024**3:.1f} GB')")"
else
    echo "WARNING: CUDA not available, running on CPU!"
fi

# Check data
for d in beauty sports toys; do
    if [ -d "$PROJECT_ROOT/data/$d" ]; then
        n=$(ls "$PROJECT_ROOT/data/$d/"*.pkl "$PROJECT_ROOT/data/$d/"*.txt "$PROJECT_ROOT/data/$d/"*.json 2>/dev/null | wc -l)
        echo "  data/$d/: $n files"
    else
        echo "  data/$d/: MISSING!"
    fi
done

echo ""

# ── Handle "all" datasets ─────────────────────────────────
if [ "$DATASET" = "all" ]; then
    DATASETS=("beauty" "sports" "toys")
else
    DATASETS=("$DATASET")
fi

# ── Main Pipeline ──────────────────────────────────────────
final_status=0

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "############################################################"
    echo "  Dataset: $ds"
    echo "############################################################"

    # ── Phase 1: Training ──────────────────────────────────
    if [ "$SKIP_TRAIN" = false ]; then
        echo ""
        echo ">>> Phase 1: Training P5 on $ds ($BACKBONE)"

        # Auto-detect best checkpoint
        TIMESTAMP=$(date +%b%d_%H-%M)
        RUN_NAME="${ds}-${BACKBONE#t5-}_${TIMESTAMP}"

        cd "$PROJECT_ROOT"
        python reproduce/train.py \
            --dataset "$ds" \
            --backbone "$BACKBONE" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --seed "$SEED" \
            --output_dir "$OUTPUT_DIR" \
            $EXTRA_ARGS

        echo ">>> Training complete for $ds"
    else
        echo ">>> SKIPPING training (--skip_train)"
    fi

    # ── Phase 2: Find best checkpoint ──────────────────────
    if [ "$SKIP_EVAL" = false ]; then
        # Find the most recent run directory
        if [ -n "${RUN_NAME:-}" ] && [ -d "$OUTPUT_DIR/$RUN_NAME" ]; then
            CKPT_DIR="$OUTPUT_DIR/$RUN_NAME"
        else
            CKPT_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "${ds}-*" | sort -r | head -1)
        fi

        if [ -z "${CKPT_DIR:-}" ] || [ ! -d "${CKPT_DIR:-}" ]; then
            echo "ERROR: No checkpoint directory found for $ds in $OUTPUT_DIR"
            echo "  Directories found:"
            find "$OUTPUT_DIR" -maxdepth 1 -type d 2>/dev/null || echo "  (none)"
            final_status=1
            continue
        fi

        BEST_CKPT="$CKPT_DIR/BEST_EVAL_LOSS.pth"
        if [ ! -f "$BEST_CKPT" ]; then
            # Fall back to last epoch checkpoint
            BEST_CKPT=$(find "$CKPT_DIR" -name "Epoch*.pth" | sort -V | tail -1)
        fi

        if [ -z "${BEST_CKPT:-}" ] || [ ! -f "${BEST_CKPT:-}" ]; then
            echo "ERROR: No checkpoint .pth found in $CKPT_DIR"
            echo "  Contents:"
            ls -la "$CKPT_DIR/" 2>/dev/null || echo "  (empty)"
            final_status=1
            continue
        fi

        echo ""
        echo ">>> Phase 2: Evaluation using $BEST_CKPT"

        cd "$PROJECT_ROOT"
        python reproduce/eval.py \
            --checkpoint "$BEST_CKPT" \
            --dataset "$ds" \
            --backbone "$BACKBONE" \
            --beam_size 20 \
            --batch_size 4 \
            --tasks "$TASKS" \
            --output_file "$CKPT_DIR/eval_results.json"

        echo ">>> Evaluation complete for $ds"

        # ── Phase 3: Generate Report ────────────────────────
        echo ""
        echo ">>> Phase 3: Generating reproducibility report"

        cd "$PROJECT_ROOT"
        python reproduce/generate_report.py \
            --results "$CKPT_DIR/eval_results.json" \
            --dataset "$ds" \
            --backbone "$BACKBONE" \
            --checkpoint "$BEST_CKPT" \
            --output "$CKPT_DIR/reproducibility_report.md"

        # Print report to console
        echo ""
        cat "$CKPT_DIR/reproducibility_report.md"

        echo ""
        echo ">>> Results saved to: $CKPT_DIR/"
        echo "    - eval_results.json"
        echo "    - reproducibility_report.md"

    else
        echo ">>> SKIPPING evaluation (--skip_eval)"
    fi
done

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "============================================================"

if [ $final_status -eq 0 ]; then
    echo "  Status: SUCCESS"
else
    echo "  Status: COMPLETED WITH ERRORS"
fi

echo "  Output directory: $OUTPUT_DIR"
echo "============================================================"

exit $final_status
