#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# ================= 配置区域 =================
# 人类代码检测结果的保存目录
OUTPUT_ROOT="outputs/sweet_human_baseline"
export CUDA_VISIBLE_DEVICES=0
echo "========================================================"
echo "Start detecting HUMAN code (Control Group)..."
echo "Results will be saved to: $OUTPUT_ROOT"
echo "========================================================"

# ---------------------------------------------------------
# 1. HumanEval (Human Code)
# ---------------------------------------------------------
echo "[1/3] Detecting HumanEval Human Code..."

task="humaneval"
max_len=512
batch_size=20
top_p=0.95
n_sample=40
accelerate launch main.py \
    --model bigcode/starcoder \
    --use_auth_token \
    --task $task \
    --temperature 0.2 \
    --precision bf16 \
    --batch_size $batch_size \
    --allow_code_execution \
    --do_sample \
    --top_p $top_p \
    --n_samples $n_sample \
    --max_length_generation $max_len \
    --detect_human_code \
    --outputs_dir "$OUTPUT_ROOT/humaneval" \
    --sweet \
    --gamma 0.25 \
    --delta 2.0 \
    --entropy_threshold 0.9

echo "--------------------------------------------------------"

# ---------------------------------------------------------
# 2. MBPP (Human Code)
# ---------------------------------------------------------
echo "[2/3] Detecting MBPP Human Code..."

task="mbpp"
max_len=2048
batch_size=5
top_p=0.95
n_sample=20

accelerate launch main.py \
      --model bigcode/starcoder \
    --use_auth_token \
    --task $task \
    --temperature 0.2 \
    --precision bf16 \
    --batch_size $batch_size \
    --allow_code_execution \
    --do_sample \
    --top_p $top_p \
    --n_samples $n_sample \
    --max_length_generation $max_len \
    --detect_human_code \
    --outputs_dir "$OUTPUT_ROOT/mbpp" \
    --sweet \
    --gamma 0.25 \
    --delta 2.0 \
    --entropy_threshold 0.9

echo "--------------------------------------------------------"

# ---------------------------------------------------------
# 3. DS-1000 (Human Code)
# ---------------------------------------------------------
# echo "[3/3] Detecting DS-1000 Human Code..."

# task="ds1000-all-completion"
# max_len=1024
# batch_size=10
# top_p=0.5
# n_sample=40

# accelerate launch main.py \
#   --model bigcode/starcoder \
    # --use_auth_token \
    # --task $task \
    # --temperature 0.2 \
    # --precision bf16 \
    # --batch_size $batch_size \
    # --allow_code_execution \
    # --do_sample \
    # --top_p $top_p \
    # --n_samples $n_sample \
    # --max_length_generation $max_len \
    # --detect_human_code \
    # --outputs_dir "$OUTPUT_ROOT/ds1000" \
    # --sweet \
    # --gamma 0.25 \
    # --delta 2.0 \
    # --entropy_threshold 0.9

echo "========================================================"
echo "All human detection tasks finished."