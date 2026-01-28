#!/bin/bash
# 1. 基础环境设置
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export HF_ENDPOINT=https://hf-mirror.com
# 指定显卡 (保持你原来的设置)
export CUDA_VISIBLE_DEVICES=2

# ================= 配置区域 =================
MODEL="bigcode/starcoder"
# 【修改】区分输出目录，避免覆盖 HumanEval 的结果
BASE_OUTPUT_DIR="outputs/grid_search_mbpp_full_60"
# 【修改】任务名称
TASK="mbpp"

# 【修改】MBPP 专属参数设置
MAX_LEN=2048
BATCH_SIZE=5      # MBPP 序列长，BatchSize 需调小
TOP_P=0.95
N_SAMPLES=20

# 全量参数 (3 * 5 * 4 = 60 组)
GAMMAS=(0.1 0.25 0.5)
DELTAS=(0.5 1.0 2.0 3.0 4.0)
THRESHOLDS=(0.3 0.6 0.9 1.2)

SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary.csv"
mkdir -p "$BASE_OUTPUT_DIR"

if [ ! -f "$SUMMARY_FILE" ]; then
    echo "Gamma,Delta,Threshold,Pass@1,TPR(Detection_Rate),AUROC" > "$SUMMARY_FILE"
fi

echo "========================================================"
echo "Starting MBPP Grid Search (Full 60 combinations)..."
echo "Results will be saved to: $BASE_OUTPUT_DIR"
echo "========================================================"

count=0
total_steps=$((${#GAMMAS[@]} * ${#DELTAS[@]} * ${#THRESHOLDS[@]}))

for gamma in "${GAMMAS[@]}"; do
  for delta in "${DELTAS[@]}"; do
    for threshold in "${THRESHOLDS[@]}"; do
      
      count=$((count+1))
      
      # 【注意】这里移除了跳过逻辑，因为这是新的 MBPP 实验，你需要跑全所有组合。
      
      EXP_NAME="g${gamma}_d${delta}_t${threshold}"
      EXP_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"
      
      mkdir -p "$EXP_DIR"
      MACHINE_DIR="${EXP_DIR}/machine"
      HUMAN_DIR="${EXP_DIR}/human"
      
      echo "----------------------------------------------------------------"
      echo "[$count/$total_steps] Parameters: Gamma=$gamma, Delta=$delta, Threshold=$threshold"
      echo "----------------------------------------------------------------"

      # ==========================================
      # 1. 机器生成与检测 (这一步是你原代码里缺失的，我补全了)
      # ==========================================
      if [ -f "${MACHINE_DIR}/evaluation_results.json" ]; then
          echo "  [1/3] Machine generation found, skipping..."
      else
          echo "  [1/3] Running Generation & Detection..."
          # 使用 MBPP 的参数：MAX_LEN, BATCH_SIZE 等
          accelerate launch main.py \
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
            --sweet \
            --gamma $gamma \
            --delta $delta \
            --entropy_threshold $threshold > "${EXP_DIR}/machine.log" 2>&1
      fi

      # ==========================================
      # 2. 人类代码检测
      # ==========================================
      if [ -f "${HUMAN_DIR}/evaluation_results.json" ]; then
          echo "  [2/3] Human baseline found, skipping..."
      else
          echo "  [2/3] Detecting Human Code (Control Group)..."
          accelerate launch main.py \
            --model $MODEL \
            --use_auth_token \
            --task $TASK \
            --precision bf16 \
            --batch_size $BATCH_SIZE \
            --allow_code_execution \
            --max_length_generation $MAX_LEN \
            --detect_human_code \
            --outputs_dir "$HUMAN_DIR" \
            --sweet \
            --gamma $gamma \
            --delta $delta \
            --entropy_threshold $threshold > "${EXP_DIR}/human.log" 2>&1
      fi

      # ==========================================
      # 3. 计算 AUROC
      # ==========================================
      echo "  [3/3] Calculating AUROC..."
      
      python3 scripts/main/extract_metrics.py \
          "${MACHINE_DIR}/evaluation_results.json" \
          "${HUMAN_DIR}/evaluation_results.json" \
          "$TASK" \
          "$gamma" \
          "$delta" \
          "$threshold" >> "$SUMMARY_FILE"

    done
  done
done

echo "Done! Check output at: $SUMMARY_FILE"