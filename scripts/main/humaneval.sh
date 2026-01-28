#!/bin/bash
# 1. 基础环境设置
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=3

# ================= 配置区域 =================
MODEL="bigcode/starcoder"
BASE_OUTPUT_DIR="outputs/grid_search_full_fix"
TASK="humaneval"

GAMMAS=(0.25 0.5)
DELTAS=(2.0 3.0)
THRESHOLDS=(0.9 1.2)

SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary.csv"
mkdir -p "$BASE_OUTPUT_DIR"

if [ ! -f "$SUMMARY_FILE" ]; then
    echo "Gamma,Delta,Threshold,Pass@1,TPR(Detection_Rate),AUROC" > "$SUMMARY_FILE"
fi

echo "========================================================"
echo "Starting Robust Grid Search..."
echo "Results will be saved to: $BASE_OUTPUT_DIR"
echo "========================================================"

count=0
total_steps=$((${#GAMMAS[@]} * ${#DELTAS[@]} * ${#THRESHOLDS[@]}))

for gamma in "${GAMMAS[@]}"; do
  for delta in "${DELTAS[@]}"; do
    for threshold in "${THRESHOLDS[@]}"; do
      
      count=$((count+1))
      EXP_NAME="g${gamma}_d${delta}_t${threshold}"
      EXP_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"
      
      mkdir -p "$EXP_DIR"
      MACHINE_DIR="${EXP_DIR}/machine"
      HUMAN_DIR="${EXP_DIR}/human"
      
      echo "----------------------------------------------------------------"
      echo "[$count/$total_steps] Parameters: Gamma=$gamma, Delta=$delta, Threshold=$threshold"
      echo "----------------------------------------------------------------"

      # 1. 机器生成与检测
      if [ -f "${MACHINE_DIR}/evaluation_results.json" ]; then
          echo "  [1/3] Machine generation found, skipping..."
      else
          echo "  [1/3] Running Generation & Detection..."
          accelerate launch main.py \
            --model $MODEL \
            --use_auth_token \
            --task $TASK \
            --temperature 0.2 \
            --precision bf16 \
            --batch_size 20 \
            --allow_code_execution \
            --do_sample \
            --top_p 0.95 \
            --n_samples 20 \
            --max_length_generation 512 \
            --save_generations \
            --outputs_dir "$MACHINE_DIR" \
            --sweet \
            --gamma $gamma \
            --delta $delta \
            --entropy_threshold $threshold > "${EXP_DIR}/machine.log" 2>&1
      fi

      # 2. 人类代码检测
      if [ -f "${HUMAN_DIR}/evaluation_results.json" ]; then
          echo "  [2/3] Human baseline found, skipping..."
      else
          echo "  [2/3] Detecting Human Code (Control Group)..."
          accelerate launch main.py \
            --model $MODEL \
            --use_auth_token \
            --task $TASK \
            --precision bf16 \
            --batch_size 20 \
            --allow_code_execution \
            --max_length_generation 512 \
            --detect_human_code \
            --outputs_dir "$HUMAN_DIR" \
            --sweet \
            --gamma $gamma \
            --delta $delta \
            --entropy_threshold $threshold > "${EXP_DIR}/human.log" 2>&1
      fi
    done
  done
done

echo "Done! Check output at: $SUMMARY_FILE"