#!/usr/bin/env bash
set -euo pipefail

# One optimizer-step smoke for the Future-L1 TwiFF training path on one GPU.
# This intentionally uses stateless SGD instead of the paper's AdamW/ZeRO setup
# so an 8B model can test Trainer and parameter-update plumbing on one 80 GB GPU.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/root/local-checkpoints/Qwen3-VL-8B-Instruct}"
DATA_PATH="${DATA_PATH:-/workspace/data/future_l1_tiny/train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/results/future_l1_train_step_smoke}"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "ERROR: model checkpoint not found: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: TwiFF smoke dataset not found: $DATA_PATH" >&2
  exit 1
fi
if [[ ! -f "$REPO_ROOT/chat_template.json" ]]; then
  echo "ERROR: Future-L1 chat template not found: $REPO_ROOT/chat_template.json" >&2
  exit 1
fi

# Qwen's stock template drops assistant-side image placeholders. Training needs
# the repository template so latent image features and latent tokens stay aligned.
if ! cmp -s "$REPO_ROOT/chat_template.json" "$MODEL_PATH/chat_template.json"; then
  echo "Installing Future-L1 chat template into the local checkpoint."
  cp "$REPO_ROOT/chat_template.json" "$MODEL_PATH/chat_template.json"
fi

mkdir -p "$OUTPUT_DIR" "$REPO_ROOT/reports"

export PYTHONPATH="$REPO_ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_DISABLED=true
export FUTURE_L1_SKIP_FINAL_SAVE=1

echo "model_path=$MODEL_PATH"
echo "data_path=$DATA_PATH"
echo "output_dir=$OUTPUT_DIR"
echo "optimizer=sgd (smoke only; not the paper training optimizer)"

cd "$REPO_ROOT"
python -u src/train/train.py \
  --model_id "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --run_name future_l1_train_step_smoke \
  --remove_unused_columns False \
  --use_twiff_dataset True \
  --freeze_vision_tower True \
  --freeze_merger True \
  --freeze_llm False \
  --latent_loss mse \
  --latent_lambda 0.2 \
  --max_latent_token 4 \
  --learning_rate 1e-5 \
  --optim sgd \
  --weight_decay 0.0 \
  --max_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --bf16 True \
  --fp16 False \
  --disable_flash_attn2 True \
  --gradient_checkpointing True \
  --image_min_pixels 3136 \
  --image_max_pixels 4096 \
  --video_max_pixels 100352 \
  --nframes 4 \
  --logging_steps 1 \
  --save_strategy no \
  --dataloader_num_workers 0 \
  --random_seed 0 \
  --report_to none

echo "OVERALL PASS (one Trainer/SGD update; no model checkpoint saved)"
