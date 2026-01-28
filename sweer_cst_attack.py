import os
import json
import torch
import logging
import argparse
import random
import re
import keyword
import libcst as cst
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. 依赖库检查 ---
try:
    import nltk
    from nltk.corpus import wordnet
    wordnet.ensure_loaded()
except:
    os.system("pip install nltk")
    import nltk
    nltk.download('wordnet')
    from nltk.corpus import wordnet

try:
    # 加载 HumanEval 数据集用于剥离 Prompt
    from datasets import load_dataset
except ImportError:
    os.system("pip install datasets")
    from datasets import load_dataset

try:
    from sweet import SweetDetector
    SWEET_AVAILABLE = True
except ImportError:
    print("错误: 未找到 sweet 库。")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 攻击模块 (V3.0 混合密集攻击)
# ==========================================
WORDS = list(wordnet.all_synsets())

def generate_random_identifier(psi=5):
    identifier = []
    while len(identifier) < psi:
        try:
            word = random.choice(WORDS).lemmas()[0].name()
            if re.search(r'[^a-zA-Z0-9]', word) or word in keyword.kwlist: continue
            identifier.append(word)
        except: continue
    return '_'.join(identifier)

def t_add_dense_dead_code(module_code, prob=0.3, attack_mode='mixed'):
    """密集死代码攻击: 混合多种模式"""
    try:
        module = cst.parse_module(module_code)
    except: return False, module_code

    if len(module.body) == 0: return False, module.code

    def get_dead_pattern():
        v1 = generate_random_identifier(3)
        v2 = generate_random_identifier(3)
        patterns = [
            f"{v1} = {v1}", 
            f"if {v1} != {v1}: {v2} = {random.randint(0,100)}",
            f"{v1} = [i for i in []]",
            f"{v1} = str({v2}) if False else {v1}"
        ]
        if attack_mode == 'simple': return patterns[0]
        if attack_mode == 'complex': return random.choice(patterns[1:])
        return random.choice(patterns) # mixed

    class DenseTransformer(cst.CSTTransformer):
        def __init__(self):
            self.inserted = 0
        def leave_SimpleStatementLine(self, node, updated_node):
            if random.random() < prob:
                self.inserted += 1
                try:
                    dead = get_dead_pattern()
                    new_body = cst.parse_module(dead).body
                    return cst.FlattenSentinel([*new_body, updated_node])
                except: return updated_node
            return updated_node

    transformer = DenseTransformer()
    new_module = module.visit(transformer)
    return transformer.inserted > 0, new_module.code

# ==========================================
# 检测模块 (带 Prompt 掩码 - V4 核心修复)
# ==========================================

def calculate_entropy(model, input_ids, device):
    """计算熵"""
    if input_ids.shape[1] < 2: return None
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        # 补齐首位
        entropy = torch.cat([torch.full((1, 1), 5.0).to(device), entropy], dim=1)
    return entropy.squeeze().cpu().tolist()

def detect_aware(detector, model, tokenizer, code, prompt, device, z_thresh, entropy_thresh):
    """
    上下文感知检测: 
    1. 拼接 Prompt + Code 计算正确的熵
    2. 设置 prefix_len 忽略 Prompt 部分的水印检测
    """
    try:
        # 编码 Prompt 以获取前缀长度
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prefix_len = len(prompt_ids)
        
        # 智能拼接：如果 code 已经包含 prompt，就不重复拼接
        if code.strip().startswith(prompt.strip()[:20]):
            full_code = code
        else:
            full_code = prompt + "\n" + code
            
        full_ids = tokenizer.encode(full_code, add_special_tokens=False, truncation=True, max_length=1536)
        
        if len(full_ids) <= prefix_len: 
            return False, 0.0 # 生成内容太短
            
        # 计算完整熵 (包含 Prompt 上下文)
        entropy = calculate_entropy(model, torch.tensor(full_ids).unsqueeze(0).to(device), device)
        
        # 核心修复: 传入 prefix_len，告诉检测器跳过 Prompt 部分
        # 注意：这里需要确保 detector 初始化时传入了正确的 gamma/delta
        score = detector._score_sequence(
            input_ids=torch.tensor(full_ids),
            prefix_len=prefix_len,  # <--- 关键！只检测生成部分
            entropy=entropy 
        )
        z = score.get('z_score', 0.0)
        return z > z_thresh, z
    except Exception as e:
        # logger.error(e)
        return False, 0.0

# ==========================================
# 主程序 (支持所有参数)
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="SWEET 攻击 V4.2 (对比evaluation_results.json)")
    # V4 基础参数
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="outputs/attack_v4_result")
    parser.add_argument('--prob', type=float, default=0.4, help="攻击强度")
    parser.add_argument('--limit', type=int, default=None)
    
    # V3 兼容参数
    parser.add_argument('--model_name', type=str, default="bigcode/starcoder")
    parser.add_argument('--attack_mode', type=str, default='mixed')
    parser.add_argument('--entropy_threshold', type=float, default=0.9)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--delta', type=float, default=3.0)
    parser.add_argument('--z_threshold', type=float, default=4.0)
    
    # 新增: 与 evaluation_results.json 对比
    parser.add_argument('--use_eval_baseline', action='store_true', 
                       help="从evaluation_results.json读取原始z-score作为baseline")
    
    args = parser.parse_args()

    # 1. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"加载模型 {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # 2. 加载 HumanEval 数据集 (获取原始 Prompt)
    logger.info("加载 HumanEval 数据集...")
    humaneval = load_dataset("openai_humaneval", split="test")
    prompts_map = {item['task_id']: item['prompt'] for item in humaneval}

    # 3. 初始化检测器
    detector = SweetDetector(
        vocab=list(tokenizer.get_vocab().values()), 
        gamma=args.gamma, 
        delta=args.delta, 
        entropy_threshold=args.entropy_threshold, 
        tokenizer=tokenizer
    )

    # 4. 加载生成结果 + 可选的评估基线
    data_path = Path(args.dataset_dir) / "machine" / "generations.json"
    eval_path = Path(args.dataset_dir) / "machine" / "evaluation_results.json"
    
    baseline_z_scores = None
    baseline_stats = None
    
    if args.use_eval_baseline and eval_path.exists():
        logger.info(f"加载评估基线: {eval_path}")
        with open(eval_path, 'r') as f:
            eval_data = json.load(f)
            if 'humaneval' in eval_data and 'watermark_detection' in eval_data['humaneval']:
                wd = eval_data['humaneval']['watermark_detection']
                baseline_stats = {
                    'total': wd['total_samples'],
                    'detected': wd['positive_samples'],
                    'detection_rate': wd['detection_rate'],
                    'tpr_fpr5': wd.get('TPR (FPR < 5%)', 0),
                    'roc_auc': wd.get('roc_auc', 0)
                }
                # 提取每个样本的 z_score
                baseline_z_scores = [r.get('z_score', 0) for r in wd.get('raw_detection_results', [])]
                logger.info(f"  原始检测统计: {baseline_stats['detected']}/{baseline_stats['total']} = {baseline_stats['detection_rate']:.4f}")
                logger.info(f"  TPR (FPR < 5%): {baseline_stats['tpr_fpr5']:.4f}")
    
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    # 展平数据
    samples = []
    sample_idx = 0
    if isinstance(raw_data, list):
        for i, task_codes in enumerate(raw_data):
            task_id = f"HumanEval/{i}"
            prompt = prompts_map.get(task_id, "")
            if isinstance(task_codes, list):
                for code in task_codes:
                    baseline_z = baseline_z_scores[sample_idx] if baseline_z_scores and sample_idx < len(baseline_z_scores) else None
                    samples.append({
                        'task_id': task_id, 
                        'prompt': prompt, 
                        'code': code,
                        'baseline_z': baseline_z
                    })
                    sample_idx += 1
            elif isinstance(task_codes, str):
                baseline_z = baseline_z_scores[sample_idx] if baseline_z_scores and sample_idx < len(baseline_z_scores) else None
                samples.append({
                    'task_id': task_id, 
                    'prompt': prompt, 
                    'code': task_codes,
                    'baseline_z': baseline_z
                })
                sample_idx += 1

    if args.limit: samples = samples[:args.limit]
    
    logger.info(f"开始测试 {len(samples)} 个样本 (Prompt感知模式)...")
    logger.info(f"配置: prob={args.prob}, mode={args.attack_mode}")
    
    stats = {'orig_det': 0, 'att_det': 0, 'success': 0, 'baseline_match': 0}
    results = []

    for item in tqdm(samples):
        # A. 原始检测 (排除 Prompt 干扰)
        # 如果有 baseline，可以选择直接使用或重新计算
        if args.use_eval_baseline and item.get('baseline_z') is not None:
            z_orig = item['baseline_z']
            is_w = z_orig > args.z_threshold
            if is_w: 
                stats['orig_det'] += 1
                stats['baseline_match'] += 1
        else:
            is_w, z_orig = detect_aware(
                detector, model, tokenizer, item['code'], item['prompt'], 
                device, args.z_threshold, args.entropy_threshold
            )
            if is_w: stats['orig_det'] += 1
        
        # B. 攻击 (密集插入)
        success, att_code = t_add_dense_dead_code(
            item['code'], prob=args.prob, attack_mode=args.attack_mode
        )
        if success: stats['success'] += 1
        else: att_code = item['code']
        
        # C. 攻击后检测
        is_w_att, z_att = detect_aware(
            detector, model, tokenizer, att_code, item['prompt'], 
            device, args.z_threshold, args.entropy_threshold
        )
        if is_w_att: stats['att_det'] += 1
        
        results.append({'z_orig': z_orig, 'z_att': z_att, 'orig_detected': is_w, 'att_detected': is_w_att})

    # 报告
    total = len(samples)
    orig_detection_rate = stats['orig_det'] / total if total > 0 else 0
    att_detection_rate = stats['att_det'] / total if total > 0 else 0
    
    # 计算 ROC 相关指标需要正负样本，这里我们只能计算简单检测率
    print("\n" + "="*70)
    print(" V4.2 实验报告 (Prompt Aware + 密集攻击)")
    print("="*70)
    
    # 如果有 baseline，显示对比
    if baseline_stats:
        print(f" 【Baseline - evaluation_results.json】")
        print(f"   检测率 (z > {args.z_threshold}): {baseline_stats['detection_rate']:.4f}")
        print(f"   TPR (FPR < 5%): {baseline_stats['tpr_fpr5']:.4f} (ROC优化后)")
        print(f"   ROC AUC: {baseline_stats['roc_auc']:.4f}")
        print("-"*70)
    
    print(f" 【当前实验配置】")
    print(f"   样本数: {total}")
    print(f"   Z-score阈值: {args.z_threshold}")
    print(f"   熵阈值: {args.entropy_threshold}")
    print(f"   攻击强度: prob={args.prob}, mode={args.attack_mode}")
    if args.use_eval_baseline:
        print(f"   使用baseline z-score: {stats['baseline_match']}/{total}")
    print("-"*70)
    
    print(f" 【检测结果】")
    print(f"   原始检测率: {orig_detection_rate:.4f} ({stats['orig_det']}/{total})")
    print(f"   攻击后检测率: {att_detection_rate:.4f} ({stats['att_det']}/{total})")
    print(f"   检测率下降: {orig_detection_rate - att_detection_rate:.4f}")
    print(f"   相对下降: {(1 - att_detection_rate/orig_detection_rate)*100 if orig_detection_rate > 0 else 0:.1f}%")
    
    # 与 baseline TPR 对比
    if baseline_stats:
        baseline_tpr = baseline_stats['tpr_fpr5']
        tpr_drop_from_baseline = baseline_tpr - att_detection_rate
        print("-"*70)
        print(f" 【与Baseline TPR对比】")
        print(f"   Baseline TPR (FPR<5%): {baseline_tpr:.4f}")
        print(f"   攻击后检测率: {att_detection_rate:.4f}")
        print(f"   相对Baseline下降: {tpr_drop_from_baseline:.4f} ({tpr_drop_from_baseline/baseline_tpr*100:.1f}%)")
    
    print("="*70)
    print()
    print(" 💡 说明:")
    print("   - evaluation_results.json 中的 'TPR (FPR < 5%)' 是通过ROC曲线")
    print("     优化阈值得到的，需要负样本(无水印代码)才能计算")
    print("   - 当前脚本使用固定阈值 z > 4.0 计算检测率")
    print("   - 使用 --use_eval_baseline 可以直接复用原始z-score避免重复计算")
    print("="*70)
    
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    with open(f"{args.output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"结果保存至 {args.output_dir}")

if __name__ == "__main__":
    main()