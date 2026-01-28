#!/bin/bash

# =============================================================================
# SWEET 密集死代码攻击脚本 V4.2
# =============================================================================

# 1. 安装必要的依赖
echo "检查依赖..."
pip install libcst nltk datasets transformers > /dev/null 2>&1

# 2. 设置环境变量
# 注意: 当前目录已经包含 sweet.py 和 watermark.py，无需额外路径
export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=3

# 3. 数据集路径
DATASET_DIR="outputs/grid_search_full_fix/g0.25_d3.0_t0.9"

# =============================================================================
# 模式 1: 使用 evaluation_results.json 的 baseline（推荐，快速）
# =============================================================================
echo ""
echo "========================================"
echo " 模式 1: 使用 Baseline 快速攻击"
echo "========================================"
python sweer_cst_attack.py \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "outputs/attack_v4_baseline" \
  --use_eval_baseline \
  --prob 0.4 \
  --attack_mode mixed \
  --gamma 0.25 \
  --delta 3.0 \
  --entropy_threshold 0.9 \
  --z_threshold 4.0

# =============================================================================
# 模式 2: 完整重新计算（慢但准确）
# =============================================================================
# 如果需要完整重新计算，取消下面注释
# echo ""
# echo "========================================"
# echo " 模式 2: 完整重新计算"
# echo "========================================"
# python sweer_cst_attack.py \
#   --dataset_dir "$DATASET_DIR" \
#   --output_dir "outputs/attack_v4_full" \
#   --prob 0.5 \
#   --attack_mode mixed \
#   --gamma 0.25 \
#   --delta 3.0 \
#   --entropy_threshold 0.9 \
#   --z_threshold 4.0

# =============================================================================
# 模式 3: 调试模式（快速验证）
# =============================================================================
# 如果需要快速验证，取消下面注释
# echo ""
# echo "========================================"
# echo " 模式 3: 调试模式 (前100个样本)"
# echo "========================================"
# python sweer_cst_attack.py \
#   --dataset_dir "$DATASET_DIR" \
#   --output_dir "outputs/attack_v4_debug" \
#   --use_eval_baseline \
#   --prob 0.5 \
#   --attack_mode mixed \
#   --limit 100 \
#   --debug

echo ""
echo "========================================"
echo " 攻击完成！"
echo "========================================"