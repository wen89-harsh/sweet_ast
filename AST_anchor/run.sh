#!/bin/bash
# =================================================================
# Sweet-AST Full Grid Search Script (Integrated Analysis)
# 目标: 遍历参数 -> 生成/检测 -> 立即计算 AUROC/TPR
#注意n_samples到时候要调到40以完整验证
# =================================================================
# 0. 激活conda环境（修改为你的环境名）
source ~/anaconda3/etc/profile.d/conda.sh
conda activate wenv

# 1. 基础环境设置
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1

# 2. 模型与路径配置
MODEL="bigcode/starcoder"
BASE_OUTPUT_DIR="outputs/grid_search_ast_full"
TASK="humaneval"
SUMMARY_FILE="${BASE_OUTPUT_DIR}/summary.csv"

mkdir -p "$BASE_OUTPUT_DIR"

# 初始化汇总文件 (注意：如果 calculate_auroc_tpr.py 会追加内容，这里只需建个头)
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "Gamma,Delta,Threshold,Weight,Pass@1,Detection_Rate,Mean_Z,Green_Fraction" > "$SUMMARY_FILE"
fi

# =================================================================
# 3. 参数网格定义 (共 1*5*4*3 = 60 组实验)
# =================================================================

# Gamma: 固定 0.25
# 【建议配置】
GAMMAS=(0.4)

# Delta: 覆盖从"轻微干扰"到"强力干扰"
# 1.5/2.0 (保质量), 2.5/3.0 (平衡), 4.0 (保检测)
DELTAS=(2.5 3.0 3.5)

# Threshold: 覆盖从"激进计分"到"保守计分"
# 0.4/0.5 (解决样本少的问题), 0.6 (推荐), 0.8 (论文原设)
THRESHOLDS=(0.5 0.6 0.8)

# Anchor Weight: 覆盖"结构依赖"的强度
# 0.5 (增加随机性), 0.7 (折中), 0.9 (强结构)
WEIGHTS=(0.7 0.8)
# =================================================================
# 4. 执行循环
# =================================================================

count=0
total_steps=$((${#GAMMAS[@]} * ${#DELTAS[@]} * ${#THRESHOLDS[@]} * ${#WEIGHTS[@]}))

echo "========================================================"
echo "Starting Full Grid Search with Analysis..."
echo "Total experiments: $total_steps"
echo "========================================================"

for gamma in "${GAMMAS[@]}"; do
  for delta in "${DELTAS[@]}"; do
    for threshold in "${THRESHOLDS[@]}"; do
      for weight in "${WEIGHTS[@]}"; do
      
        count=$((count+1))
        # 实验命名规范: g(gamma)_d(delta)_t(threshold)_w(weight)
        EXP_NAME="g${gamma}_d${delta}_t${threshold}_w${weight}"
        EXP_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"
        
        mkdir -p "$EXP_DIR"
        MACHINE_DIR="${EXP_DIR}/machine"
        HUMAN_DIR="${EXP_DIR}/human"
        
        echo "================================================================================"
        echo "[$count/$total_steps] Config: Gamma=$gamma | Delta=$delta | Thr=$threshold | W=$weight"
        echo "================================================================================"

        # ------------------------------------------------------------------
        # Step 1: Human Detection (人类代码检测 - 负样本)
        # ------------------------------------------------------------------
        if [ -f "${HUMAN_DIR}/evaluation_results.json" ]; then
            echo "  [1/3] Human baseline found, skipping..."
        else
            echo "  [1/3] Detecting Human Code (AST Mode)..."
            accelerate launch AST_anchor/main_ast.py \
              --model $MODEL \
              --use_auth_token \
              --task $TASK \
              --precision bf16 \
              --batch_size 1 \
              --allow_code_execution \
              --max_length_generation 512 \
              --detect_human_code \
              --outputs_dir "$HUMAN_DIR" \
              --sweet_ast \
              --gamma $gamma \
              --delta $delta \
              --entropy_threshold $threshold \
              --anchor_weight $weight > "${EXP_DIR}/human.log" 2>&1
        fi

        # ------------------------------------------------------------------
        # Step 2: Machine Generation (机器生成与检测 - 正样本)
        # ------------------------------------------------------------------
        if [ -f "${MACHINE_DIR}/evaluation_results.json" ]; then
            echo "  [2/3] Machine generation found, skipping..."
        else
            echo "  [2/3] Running Generation & Detection (AST Mode)..."
            accelerate launch AST_anchor/main_ast.py \
              --model $MODEL \
              --use_auth_token \
              --task $TASK \
              --temperature 0.2 \
              --precision bf16 \
              --batch_size 8 \
              --allow_code_execution \
              --do_sample \
              --top_p 0.95 \
              --n_samples 10 \
              --max_length_generation 512 \
              --save_generations \
              --outputs_dir "$MACHINE_DIR" \
              --sweet_ast \
              --gamma $gamma \
              --delta $delta \
              --entropy_threshold $threshold \
              --anchor_weight $weight > "${EXP_DIR}/machine.log" 2>&1
        fi
        
        # ------------------------------------------------------------------
        # Step 3: Calculate AUROC & TPR (整合分析)
        # ------------------------------------------------------------------
        MACHINE_JSON="${MACHINE_DIR}/evaluation_results.json"
        HUMAN_JSON="${HUMAN_DIR}/evaluation_results.json"
        METRICS_LOG="${EXP_DIR}/metrics_report.txt"

        if [ -f "$MACHINE_JSON" ] && [ -f "$HUMAN_JSON" ]; then
            echo "  [3/3] Calculating AUROC & TPR..."
            
            # 运行分析脚本，并将输出保存到实验目录的 metrics_report.txt
            python3 calculate_auroc_tpr.py \
                --task $TASK \
                --machine_fname "$MACHINE_JSON" \
                --human_fname "$HUMAN_JSON" > "$METRICS_LOG" 2>&1
            
            # 将关键结果打印到屏幕，方便您实时看（从日志中提取最后几行）
            tail -n 3 "$METRICS_LOG"
            
            # 【可选】如果您希望 calculate_auroc_tpr.py 的结果写入总表 SUMMARY_FILE
            # 您需要确保该 python 脚本输出的是符合 CSV 格式的一行，如果是，取消下面注释：
            # python3 calculate_auroc_tpr.py ... >> "$SUMMARY_FILE"
        else
            echo "  [3/3] Missing evaluation files, skipping analysis."
        fi
        
      done
    done
  done
done

echo "All grid search experiments completed! Check $SUMMARY_FILE"