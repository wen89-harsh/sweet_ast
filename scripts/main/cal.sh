#!/bin/bash
# =================================================================
# 批量结果分析脚本
# 功能: 遍历所有实验结果，计算 AUROC/TPR，并提取 Pass@1
# =================================================================

# 1. 基础设置
export PYTHONIOENCODING=utf-8
BASE_DIR="outputs/grid_search_ast_full"
REPORT_FILE="final_analysis_report.csv"
export CUDA_VISIBLE_DEVICES=2
# 激活环境 (确保使用您的 wenv python)
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate wenv
PYTHON_CMD="/home/shanjianping/anaconda3/envs/wenv/bin/python" 

echo "========================================================"
echo "开始批量分析所有实验结果..."
echo "结果将保存至: $REPORT_FILE"
echo "========================================================"

# 初始化 CSV 表头
# 格式: 实验名, Pass@1, AUROC, TPR(Low), TPR(Med), TPR(High)
echo "Config_Name,Pass@1,AUROC,TPR(FPR<0.1%),TPR(FPR<1%),TPR(FPR<5%)" > "$REPORT_FILE"

count=0

# 2. 遍历目录
for exp_dir in "$BASE_DIR"/g*; do
    # 检查是不是目录
    if [ ! -d "$exp_dir" ]; then continue; fi

    # 获取实验名称 (例如 g0.25_d2.0_t0.4_w0.5)
    exp_name=$(basename "$exp_dir")
    
    # 定义文件路径
    machine_file="${exp_dir}/machine/evaluation_results.json"
    human_file="${exp_dir}/human/evaluation_results.json"

    # 3. 检查文件是否存在
    if [ -f "$machine_file" ] && [ -f "$human_file" ]; then
        count=$((count+1))
        echo -n "[$count] Analyzing $exp_name ... "

        # --- A. 提取 Pass@1 (从 Machine JSON 中读取) ---
        # 使用 python 单行脚本快速提取，比 jq 更通用
        pass_at_1=$($PYTHON_CMD -c "import json; print(round(json.load(open('$machine_file'))['humaneval']['pass@1'], 4))" 2>/dev/null)
        
        if [ -z "$pass_at_1" ]; then pass_at_1="N/A"; fi

        # --- B. 计算 AUROC / TPR ---
        # 运行您的 calculate_auroc_tpr.py 脚本
        metrics_output=$($PYTHON_CMD calculate_auroc_tpr.py \
            --task humaneval \
            --machine_fname "$machine_file" \
            --human_fname "$human_file" 2>/dev/null)

        # --- C. 格式化输出 ---
        # 您的脚本输出是竖着的4行数字，我们用 tr 把换行符变成逗号
        # 输入:
        # 0.76
        # 0.16
        # ...
        # 输出: 0.76,0.16,...
        csv_metrics=$(echo "$metrics_output" | tr '\n' ',' | sed 's/,$//')

        # 检查是否计算成功（防止 python 报错导致空结果）
        if [ -z "$csv_metrics" ]; then
            echo "Failed (Script Error)"
        else
            # 写入 CSV
            echo "$exp_name,$pass_at_1,$csv_metrics" >> "$REPORT_FILE"
            echo "Done. (AUROC: $(echo $csv_metrics | cut -d',' -f1))"
        fi
    else
        # echo "Skipping $exp_name (Files missing)"
        :
    fi
done

echo "========================================================"
echo "分析完成！共处理 $count 个实验。"
echo "报表已保存: $REPORT_FILE"
echo "========================================================"

# 4. 自动展示最强王者 (按 AUROC 排序)
echo ""
echo "🏆 AUROC 最高的 TOP 5 实验:"
echo "--------------------------------------------------------"
# 使用 column 命令美化显示 CSV
# sort -t, -k3 -nr : 按第3列(AUROC) 降序排序
head -n 1 "$REPORT_FILE" | column -t -s, 
grep -v "Config_Name" "$REPORT_FILE" | sort -t, -k3 -nr | head -n 5 | column -t -s, 

echo ""
echo "💡 提示: 您可以使用 Excel 打开 $REPORT_FILE 进行更详细的筛选。"