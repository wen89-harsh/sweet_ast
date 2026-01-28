#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com  # 强制使用镜像，防止联网报错

# ================= 配置区域 =================
# 必须和你“生成脚本”里的 OUTPUT_ROOT 保持一致！
# 这样脚本才知道去哪里找生成的代码文件。
GENERATION_ROOT="outputs/sweet_full_experiment"
MODEL="bigcode/starcoder"
export CUDA_VISIBLE_DEVICES=0
# ---------------------------------------------------------
# 1. 检测 HumanEval (生成的代码)
# ---------------------------------------------------------
echo "[1/3] Detecting HumanEval generations..."

task="humaneval"
max_len=512
batch_size=20
top_p=0.95
n_sample=40
# 拼接生成文件的路径
GEN_PATH="$GENERATION_ROOT/humaneval/generations.json"
OUTPUT_DIR="$GENERATION_ROOT/humaneval"

if [ -f "$GEN_PATH" ]; then
    accelerate launch main.py \
        --model $MODEL \
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
        --load_generations_path $GEN_PATH \
        --outputs_dir $OUTPUT_DIR \
        --sweet \
        --gamma 0.25 \
        --delta 2.0 \
        --entropy_threshold 0.9
else
    echo "Error: File $GEN_PATH not found! Skipping..."
fi

echo "--------------------------------------------------------"

# ---------------------------------------------------------
# 2. 检测 MBPP (生成的代码)
# ---------------------------------------------------------
echo "[2/3] Detecting MBPP generations..."

task="mbpp"
max_len=2048
batch_size=5
top_p=0.95
n_sample=20
GEN_PATH="$GENERATION_ROOT/mbpp/generations.json"
OUTPUT_DIR="$GENERATION_ROOT/mbpp"

if [ -f "$GEN_PATH" ]; then
    accelerate launch main.py \
        --model $MODEL \
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
        --load_generations_path $GEN_PATH \
        --outputs_dir $OUTPUT_DIR \
        --sweet \
        --gamma 0.25 \
        --delta 2.0 \
        --entropy_threshold 0.9
else
    echo "Error: File $GEN_PATH not found! Skipping..."
fi

echo "--------------------------------------------------------"

# ---------------------------------------------------------
# 3. 检测 DS-1000 (生成的代码)
# ---------------------------------------------------------
# echo "[3/3] Detecting DS-1000 generations..."

# task="ds1000-all-completion"
# max_len=1024
# batch_size=10
# top_p=0.5
# n_sample=40
# GEN_PATH="$GENERATION_ROOT/ds1000/generations.json"
# OUTPUT_DIR="$GENERATION_ROOT/ds1000"

# if [ -f "$GEN_PATH" ]; then
#     accelerate launch main.py \
#         --model $MODEL \
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
            # --load_generations_path $GEN_PATH \
            # --outputs_dir $OUTPUT_DIR \
            # --sweet \
            # --gamma 0.25 \
            # --delta 2.0 \
            # --entropy_threshold 0.9
# else
#     echo "Error: File $GEN_PATH not found! Skipping..."
# fi

echo "========================================================"
echo "All detection tasks finished."