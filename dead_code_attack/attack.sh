#!/bin/bash
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# AST感知的水印攻击 - 运行脚本
# 修复：支持SWEET原始方法攻击

echo "=========================================="
echo "SWEET水印死代码攻击实验"
echo "=========================================="
echo ""

# 选择模式
echo "请选择运行模式:"
echo "  1. 快速测试 (50样本，安全策略)"
echo "  2. 标准模式 (全部样本，安全策略) [推荐]"
echo "  3. 对比模式 (原始SWEET vs AST增强)"
echo "  4. 自定义模式"
echo ""

read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "运行快速测试..."
        python ast_aware_attack.py --dataset_path ../outputs/grid_search_full_fix/g0.25_d3.0_t0.9 --output_dir attack_results_sweet_original --only_first_sample --num_samples 50
        ;;
    2)
        echo ""
        echo "运行标准模式（推荐）..."
        python ast_aware_attack.py --dataset_path ../outputs/grid_search_full_fix/g0.25_d3.0_t0.9 --output_dir attack_results_sweet_original
        ;;
    3)
        echo ""
        echo "运行对比模式(原始SWEET vs AST增强)..."
        
        # 1. 攻击原始SWEET
        echo "步骤1: 攻击原始SWEET方法..."
        python ast_aware_attack.py --dataset_path ../outputs/grid_search_full_fix/g0.25_d3.0_t0.9 --output_dir attack_results_sweet_original --only_first_sample --num_samples 100
        
        # 2. 如果存在AST结果，也攻击它
        if [ -d "../outputs/sweet_ast_results" ]; then
            echo "步骤2: 攻击SWEET+AST方法..."
            python ast_aware_attack.py --dataset_path ../outputs/sweet_ast_results --output_dir attack_results_sweet_ast --only_first_sample --num_samples 100
            
            echo "对比结果生成中..."
            python compare_attacks.py
        else
            echo "未找到SWEET+AST结果，请先生成"
        fi
        ;;
    4)
        echo ""
        echo "自定义模式"
        read -p "样本数量 (默认=全部): " num_samples
        read -p "是否使用死代码块? (y/n, 默认n): " use_dead
        
        cmd="python ast_aware_attack.py --dataset_path ../outputs/grid_search_full_fix/g0.25_d3.0_t0.9 --output_dir attack_results_sweet_original"
        
        if [ ! -z "$num_samples" ]; then
            cmd="$cmd --num_samples $num_samples --only_first_sample"
        fi
        
        if [ "$use_dead" = "y" ]; then
            cmd="$cmd --use_dead_code"
        fi
        
        echo ""
        echo "执行命令: $cmd"
        $cmd
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "? 实验失败"
    exit $EXIT_CODE
fi

echo ""
echo "=========================================="
echo "? 实验完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  结果目录: attack_results_sweet_original/"
echo "  攻击报告: cat attack_results_sweet_original/attack_report.json"
echo "  实验日志: cat attack_results_sweet_original/attack_experiment.log"
echo ""
echo "关键指标解读:"
echo "  - TPR@FPR<5% 下降幅度: 表示攻击效果，下降越多越脆弱"
echo "  - 预期原始SWEET: 从~0.8降至~0.3-0.5"
echo "  - Pass@1 保持率: 应接近100%（使用安全策略时）"
echo ""
echo "对比AST增强版（如存在）："
echo "  结果目录: attack_results_sweet_ast/"
echo "=========================================="