#!/bin/bash

# 定义通用的参数（如果大家都一样，就放在这里；不一样的放在下面各自的板块里）
# 比如模型路径通常是一样的
MODEL="bigcode/starcoder"
OUTPUT_ROOT="outputs/sweet_full_experiment"  # 所有结果的总目录
#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_VISIBLE_DEVICES=0
# ... 下面是你原本的代码 ...
echo "========================================================"
echo "Start running all 3 tasks sequentially..."
echo "Results will be saved to: $OUTPUT_ROOT"
echo "========================================================"

# ---------------------------------------------------------
# 1. 运行 HumanEval
# ---------------------------------------------------------
# echo "[1/3] Starting HumanEval..."

# # 设置 HumanEval 特有的参数
# task="humaneval"
# max_len=512
# batch_size=20
# top_p=0.95
# n_sample=40         # 论文通常生成20或40个样本

# # 执行命令
# accelerate launch main.py \
#     --model $MODEL \
#     --use_auth_token \
#     --task $task \
#     --temperature 0.2 \
#     --precision bf16 \
#     --batch_size $batch_size \
#     --allow_code_execution \
#     --do_sample \
#     --top_p $top_p \
#     --n_samples $n_sample \
#     --max_length_generation $max_len \
#     --save_generations \
#     --outputs_dir "$OUTPUT_ROOT/humaneval" \
#     --sweet \
#     --gamma 0.25 \
#     --delta 2.0 \
#     --entropy_threshold 0.9

# echo "[1/3] HumanEval Finished!"
# echo "--------------------------------------------------------"

# # ---------------------------------------------------------
# # 2. 运行 MBPP
# # ---------------------------------------------------------
# echo "[2/3] Starting MBPP..."

# # 设置 MBPP 特有的参数 (注意 max_len 和 n_sample 变了)
# task="mbpp"
# max_len=2048
# batch_size=5         # MBPP 比较长，batch_size 建议调小
# top_p=0.95
# n_sample=20

# # 执行命令
# accelerate launch main.py \
#     --model $MODEL \
#     --use_auth_token \
#     --task $task \
#     --temperature 0.2 \
#     --precision bf16 \
#     --batch_size $batch_size \
#     --allow_code_execution \
#     --do_sample \
#     --top_p $top_p \
#     --n_samples $n_sample \
#     --max_length_generation $max_len \
#     --save_generations \
#     --outputs_dir "$OUTPUT_ROOT/mbpp" \
#     --sweet \
#     --gamma 0.25 \
#     --delta 2.0 \
#     --entropy_threshold 0.9

# echo "[2/3] MBPP Finished!"
# echo "--------------------------------------------------------"

# ---------------------------------------------------------
# 3. 运行 DS-1000
# ---------------------------------------------------------
echo "[3/3] Starting DS-1000..."

# 设置 DS-1000 特有的参数 (注意 top_p 变了)
task="ds1000-all-completion"
max_len=2048
batch_size=10
top_p=0.5            # 注意：DS-1000 的 top_p 通常设置得比较低
n_sample=40

# 执行命令
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
    --save_generations \
    --outputs_dir "$OUTPUT_ROOT/ds1000" \
    --sweet \
    --gamma 0.25 \
    --delta 2.0 \
    --entropy_threshold 0.9

echo "[3/3] DS-1000 Finished!"
echo "========================================================"