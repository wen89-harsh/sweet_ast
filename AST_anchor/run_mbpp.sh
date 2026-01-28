#!/bin/bash
# =================================================================
# Sweet-AST Grid Search Script for MBPP (No Analysis)
# 目标: 仅执行生成和检测，不进行 AUROC 计算
# 配置: MaxLen=2048, Batch=5, Samples=20
# =================================================================

# 0. 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate wenv

# 1. 基础环境设置
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export HF_ENDPOINT=https://hf-mirror.com
# 请根据实际空闲显卡修改
export CUDA_VISIBLE_DEVICES=3

# 【显存优化】MBPP 长度为 2048，建议开启
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. 核心配置 (MBPP 特有)
MODEL="bigcode/starcoder"
TASK="mbpp"
BASE_OUTPUT_DIR="outputs/grid_search_ast_mbpp"

# MBPP 标准参数
MAX_LEN=2048
BATCH_SIZE=5
N_SAMPLES=20
TOP_P=0.95

mkdir -p "$BASE_OUTPUT_DIR"

# =================================================================
# 3. 参数网格
# =================================================================

# Gamma
GAMMAS=(0.5)

# Delta
DELTAS=(3.0)

# Threshold
THRESHOLDS=(0.5)

# Weight
WEIGHTS=(0.7)

# =================================================================
# 4. 执行循环
# =================================================================

count=0
total_steps=$((${#GAMMAS[@]} * ${#DELTAS[@]} * ${#THRESHOLDS[@]} * ${#WEIGHTS[@]}))

echo "========================================================"
echo "Starting MBPP Grid Search (Generation Only)..."
echo "Task: $TASK | Max Len: $MAX_LEN | Batch: $BATCH_SIZE"
echo "Total experiments: $total_steps"
echo "========================================================"

for gamma in "${GAMMAS[@]}"; do
  for delta in "${DELTAS[@]}"; do
    for threshold in "${THRESHOLDS[@]}"; do
      for weight in "${WEIGHTS[@]}"; do
      
        count=$((count+1))
        EXP_NAME="g${gamma}_d${delta}_t${threshold}_w${weight}"
        EXP_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"
        
        mkdir -p "$EXP_DIR"
        MACHINE_DIR="${EXP_DIR}/machine"
        HUMAN_DIR="${EXP_DIR}/human"
        
        echo ">>> [$count/$total_steps] Running: $EXP_NAME"

        # ------------------------------------------------------------------
        # Step 1: Human Baseline (MBPP test set)
        # ------------------------------------------------------------------
        if [ ! -f "${HUMAN_DIR}/evaluation_results.json" ]; then
            echo "  [1/2] Detecting Human Code..."
            accelerate launch AST_anchor/main_ast.py \
              --model $MODEL \
              --use_auth_token \
              --task $TASK \
              --precision bf16 \
              --batch_size 1 \
              --allow_code_execution \
              --max_length_generation $MAX_LEN \
              --detect_human_code \
              --outputs_dir "$HUMAN_DIR" \
              --sweet_ast \
              --gamma $gamma \
              --delta $delta \
              --entropy_threshold $threshold \
              --anchor_weight $weight > "${EXP_DIR}/human.log" 2>&1
        fi

        # ------------------------------------------------------------------
        # Step 2: Machine Generation
        # ------------------------------------------------------------------
        if [ ! -f "${MACHINE_DIR}/evaluation_results.json" ]; then
            echo "  [2/2] Generating & Detecting..."
            accelerate launch AST_anchor/main_ast.py \
              --model $MODEL \
              --use_auth_token \
              --task $TASK \
              --temperature 0.2 \
              --precision bf16 \
              --batch_size $BATCH_SIZE \
              --allow_code_execution \
              --do_sample \
              --top_p $TOP_P \
              --n_samples $N_SAMPLES \
              --max_length_generation $MAX_LEN \
              --save_generations \
              --outputs_dir "$MACHINE_DIR" \
              --sweet_ast \
              --gamma $gamma \
              --delta $delta \
              --entropy_threshold $threshold \
              --anchor_weight $weight > "${EXP_DIR}/machine.log" 2>&1
        else
            echo "  [2/2] Already exists, skipping."
        fi
      done
    done
  done
done

echo "MBPP Grid Search Generation Finished!"