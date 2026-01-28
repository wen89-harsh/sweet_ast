"""
AST感知型水印攻击
基于AST结构的水印攻击，旨在保持pass@1（代码功能正确性）的前提下进行攻击。

主要思路：
1. 只在不影响AST结构的位置插入噪声（注释、无效赋值等）
2. 不修改函数签名或关键AST节点
3. 通过增加代码熵来干扰水印检测，防止检测器识别
"""

import os
import sys
import json
import ast
import copy
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random

import torch

# 尝试导入水印检测模块
try:
    from watermark import WatermarkDetector
    from sweet import SweetDetector
    WATERMARK_AVAILABLE = True
except ImportError:
    WATERMARK_AVAILABLE = False


class ASTAwareAttacker:
    """AST感知的攻击者 - 在不破坏AST关键结构的前提下插入噪声"""
    
    def __init__(self, psi: int = 5):
        self.psi = psi
        self.logger = logging.getLogger(__name__)
        
    def generate_random_identifier(self, length: int = 3) -> str:
        """生成随机标识符"""
        import keyword
        chars = 'abcdefghijklmnopqrstuvwxyz'
        identifier = random.choice(chars)
        for _ in range(length - 1):
            identifier += random.choice(chars + '0123456789')
        while identifier in keyword.kwlist:
            identifier += random.choice(chars)
        return identifier
    
    def insert_comment_noise(self, code: str, num_comments: int = 3) -> str:
        """
        策略1: 插入随机注释
        修复：不应在块起始行（如def/if）之后立即插入，也不应插入0缩进的注释干扰结构。
        """
        lines = code.split('\n')
        if len(lines) < 2:
            return code
        
        inserted = 0
        new_lines = []
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            
            line_strip = line.strip()
            # 计算缩进
            indent = len(line) - len(line.lstrip())
            
            # 安全检查：
            # 1. 还没有插满
            # 2. 当前行不是空行
            # 3. 当前行不是注释
            # 4. 关键修复：不要在 'def:', 'if:', 'else:' 等块起始语句后直接插入，除非能在下一行正确处理缩进
            # 5. 关键修复：只在有缩进的地方插入（indent > 0），避免破坏函数外部结构或函数头
            is_block_starter = line_strip.endswith(':')
            
            if (inserted < num_comments and 
                len(line_strip) > 0 and 
                not line_strip.startswith('#') and
                not is_block_starter and 
                indent > 0): 
                
                if random.random() < 0.3:
                    random_text = self.generate_random_identifier(8)
                    # 使用与当前行相同的缩进
                    comment = ' ' * indent + f'# {random_text}'
                    new_lines.append(comment)
                    inserted += 1
        
        return '\n'.join(new_lines)
    
    def insert_noop_assignments(self, code: str, num_assignments: int = 2) -> str:
        """策略2: 插入无效赋值"""
        try:
            tree = ast.parse(code)
        except:
            return code
        
        function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if not function_defs:
            return code
        
        lines = code.split('\n')
        first_func = function_defs[0]
        func_line = first_func.lineno - 1
        
        # 寻找合适的插入点：跳过函数定义和文档字符串
        insert_line = func_line + 1
        found_safe_spot = False
        
        for i in range(func_line + 1, len(lines)):
            line_strip = lines[i].strip()
            # 跳过空行、文档字符串开头/结尾
            if (line_strip and 
                not line_strip.startswith('"""') and 
                not line_strip.startswith("'''") and
                not line_strip.startswith('#')):
                insert_line = i
                found_safe_spot = True
                break
        
        # 如果没找到安全插入点（例如函数只有docstring），则放弃
        if not found_safe_spot or insert_line >= len(lines):
            return code

        # 获取插入点的缩进
        indent = len(lines[insert_line]) - len(lines[insert_line].lstrip())
        
        # 关键修复：如果计算出的缩进为0（可能跳到了函数外），则不插入，防止破坏语法
        if indent == 0:
            return code
            
        indent_str = ' ' * indent
        
        noop_lines = []
        for _ in range(num_assignments):
            var_name = '_' + self.generate_random_identifier(6)
            value = random.choice([0, 1, 'None', '""', '[]'])
            noop_lines.append(f'{indent_str}{var_name} = {value}')
        
        new_lines = lines[:insert_line] + noop_lines + lines[insert_line:]
        return '\n'.join(new_lines)
    
    def insert_conditional_dead_code(self, code: str, num_blocks: int = 1) -> str:
        """策略3: 插入条件死代码"""
        try:
            tree = ast.parse(code)
        except:
            return code
        
        lines = code.split('\n')
        function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if not function_defs or len(lines) < 3:
            return code
        
        first_func = function_defs[0]
        func_start = first_func.lineno - 1
        func_end = first_func.end_lineno if hasattr(first_func, 'end_lineno') else len(lines)
        
        if func_end - func_start < 3:
            return code
        
        # 尝试寻找中间位置
        try:
            insert_pos = random.randint(func_start + 2, min(func_end - 1, len(lines) - 1))
        except ValueError:
            return code
            
        # 检查插入位置是否有效
        if insert_pos >= len(lines):
            return code

        indent = len(lines[insert_pos]) - len(lines[insert_pos].lstrip())
        
        # 关键修复：缩进保护
        if indent == 0:
            return code
            
        indent_str = ' ' * indent
        
        var1 = self.generate_random_identifier(5)
        var2 = self.generate_random_identifier(5)
        
        dead_block = [
            f'{indent_str}if False:  # Dead code',
            f'{indent_str}    {var1} = {random.randint(0, 100)}',
            f'{indent_str}    {var2} = "{self.generate_random_identifier(8)}"'
        ]
        
        new_lines = lines[:insert_pos] + dead_block + lines[insert_pos:]
        return '\n'.join(new_lines)
    
    def add_string_noise(self, code: str, num_strings: int = 2) -> str:
        """策略4: 添加字符串噪声"""
        try:
            tree = ast.parse(code)
        except:
            return code
        
        lines = code.split('\n')
        function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if not function_defs:
            return code
        
        first_func = function_defs[0]
        func_line = first_func.lineno
        
        insert_line = func_line
        found_safe_spot = False
        
        # 寻找插入点，类似 noop assignment
        for i in range(func_line, min(func_line + 10, len(lines))):
            line_strip = lines[i].strip()
            if (i < len(lines) and line_strip and 
                not line_strip.startswith('"""') and 
                not line_strip.startswith("'''") and
                not line_strip.startswith('#')):
                insert_line = i
                found_safe_spot = True
                break
        
        if not found_safe_spot or insert_line >= len(lines):
            return code

        indent = len(lines[insert_line]) - len(lines[insert_line].lstrip())
        
        # 关键修复：缩进保护
        if indent == 0:
            return code

        indent_str = ' ' * indent
        
        noise_lines = []
        for _ in range(num_strings):
            random_str = self.generate_random_identifier(10)
            noise_lines.append(f'{indent_str}_ = "{random_str}"')
        
        new_lines = lines[:insert_line] + noise_lines + lines[insert_line:]
        return '\n'.join(new_lines)
    
    def attack_code(self, code: str, 
                   use_comments: bool = True,
                   use_noop: bool = True, 
                   use_dead_code: bool = False,
                   use_string_noise: bool = True) -> Tuple[bool, str, Dict]:
        """执行攻击"""
        metadata = {
            'strategies_used': [],
            'success': False,
            'syntax_valid': False
        }
        
        try:
            current_code = code
            
            # 策略应用顺序
            if use_comments:
                current_code = self.insert_comment_noise(current_code, num_comments=3)
                metadata['strategies_used'].append('comment_noise')
            
            if use_noop:
                current_code = self.insert_noop_assignments(current_code, num_assignments=2)
                metadata['strategies_used'].append('noop_assignments')
            
            if use_string_noise:
                current_code = self.add_string_noise(current_code, num_strings=2)
                metadata['strategies_used'].append('string_noise')
            
            if use_dead_code:
                current_code = self.insert_conditional_dead_code(current_code, num_blocks=1)
                metadata['strategies_used'].append('dead_code_block')
            
            # 最终语法验证
            try:
                compile(current_code, '<string>', 'exec')
                metadata['syntax_valid'] = True
                metadata['success'] = True
            except SyntaxError as e:
                # 记录具体的语法错误，但不终止程序
                # self.logger.debug(f"语法验证失败: {e}") 
                return False, code, metadata
            except Exception as e:
                return False, code, metadata
            
            return True, current_code, metadata
            
        except Exception as e:
            self.logger.error(f"攻击过程异常: {e}")
            return False, code, metadata


class WatermarkAttackExperiment:
    """水印攻击实验"""
    
    def __init__(self, dataset_path: str, output_dir: str,
                 gamma: float = 0.25, delta: float = 3.0, temperature: float = 0.9):
        
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.gamma = gamma
        self.delta = delta
        self.temperature = temperature
        
        self.setup_logging()
        self.attacker = ASTAwareAttacker(psi=5)
        
        if WATERMARK_AVAILABLE:
            from transformers import AutoTokenizer
            model_name = "codellama/CodeLlama-7b-hf"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.logger.info(f"Tokenizer加载成功: {model_name}")
            except:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.logger.warning("使用GPT-2 tokenizer作为备用")
            
            vocab = list(self.tokenizer.get_vocab().values())
            self.detector = SweetDetector(
                vocab=vocab,
                gamma=gamma,
                delta=delta,
                entropy_threshold=0.9
            )
        else:
            self.detector = None
            self.tokenizer = None
            self.logger.warning("水印模块不可用 (将无法计算TPR)")
    
    def setup_logging(self):
        log_file = self.output_dir / "attack_experiment.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, only_first_sample=False):
        """加载数据 - 支持标准JSON和JSONL格式"""
        results_file = self.dataset_path / "machine" / "evaluation_results.json"
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                self.original_results = json.load(f)
        except Exception as e:
            self.logger.error(f"无法读取 evaluation_results.json: {e}")
            self.original_results = {'humaneval': {'pass@1': 0.0}}
        
        gen_file = self.dataset_path / "machine" / "generations.json"
        self.generations = []
        
        try:
            with open(gen_file, 'r', encoding='utf-8') as f:
                # 尝试一次性读取（标准JSON列表格式）
                try:
                    content = json.load(f)
                    if isinstance(content, list):
                        all_tasks = content
                    else:
                        all_tasks = []
                    
                    for task_idx, samples in enumerate(all_tasks):
                        self._process_samples(task_idx, samples, only_first_sample)
                        
                except json.JSONDecodeError:
                    # 回退到逐行读取（JSONL格式）
                    f.seek(0)
                    for task_idx, line in enumerate(f):
                        if line.strip():
                            samples = json.loads(line)
                            self._process_samples(task_idx, samples, only_first_sample)
                            
        except Exception as e:
            self.logger.error(f"无法读取 generations.json: {e}")

        self.logger.info(f"加载了 {len(self.generations)} 个生成样本")
        self.logger.info(f"原始 pass@1: {self.original_results['humaneval']['pass@1']:.4f}")
        
        if 'watermark_detection' in self.original_results['humaneval']:
            wd = self.original_results['humaneval']['watermark_detection']
            self.logger.info(f"原始 TPR@FPR<5%: {wd.get('TPR (FPR < 5%)', 0.0):.4f}")

    def _process_samples(self, task_idx, samples, only_first_sample):
        if isinstance(samples, list):
            samples_to_process = [samples[0]] if only_first_sample and len(samples) > 0 else samples
            for sample_idx, code in enumerate(samples_to_process):
                if isinstance(code, str):
                    self.generations.append({
                        'task_id': f'HumanEval/{task_idx}',
                        'sample_idx': sample_idx,
                        'completion': code
                    })
        elif isinstance(samples, dict):
            self.generations.append(samples)
    
    def detect_watermark(self, code: str) -> Dict:
        if not self.detector or not self.tokenizer:
            return {'prediction': False, 'z_score': 0.0}
        
        try:
            input_ids = self.tokenizer.encode(code, add_special_tokens=False)
            if len(input_ids) < 10:
                return {'prediction': False, 'z_score': 0.0}
            
            entropy = [1.5] * len(input_ids)
            result = self.detector._score_sequence(
                input_ids=torch.tensor(input_ids),
                prefix_len=0,
                entropy=entropy
            )
            z_score = result.get('z_score', 0.0)
            result['prediction'] = z_score > 4.0
            return result
        except Exception as e:
            return {'prediction': False, 'z_score': 0.0, 'error': str(e)}
    
    def run_experiment(self, num_samples: int = None,
                      use_comments: bool = True,
                      use_noop: bool = True,
                      use_dead_code: bool = False,
                      use_string_noise: bool = True,
                      only_first_sample: bool = False):
        
        self.logger.info("=" * 80)
        self.logger.info("开始AST感知型水印攻击实验")
        self.logger.info(f"配置: 注释={use_comments}, 无操作={use_noop}, "
                        f"死代码={use_dead_code}, 字符串={use_string_noise}")
        self.logger.info("=" * 80)
        
        self.load_data(only_first_sample=only_first_sample)
        
        generations = self.generations[:num_samples] if num_samples else self.generations
        self.logger.info(f"处理 {len(generations)} 个样本")
        
        attacked_generations = []
        stats = {'total': len(generations), 'success': 0, 'failed': 0}
        
        for idx, gen in enumerate(generations):
            if (idx + 1) % 100 == 0:
                self.logger.info(f"进度: {idx + 1}/{len(generations)}")
            
            success, attacked_code, metadata = self.attacker.attack_code(
                gen['completion'],
                use_comments=use_comments,
                use_noop=use_noop,
                use_dead_code=use_dead_code,
                use_string_noise=use_string_noise
            )
            
            if success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
                attacked_code = gen['completion']
            
            attacked_gen = copy.deepcopy(gen)
            attacked_gen['completion'] = attacked_code
            attacked_gen['attack_metadata'] = metadata
            attacked_generations.append(attacked_gen)
        
        self.logger.info(f"攻击统计: 成功={stats['success']}/{stats['total']}")
        
        output_file = self.output_dir / "attacked_generations.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for gen in attacked_generations:
                f.write(json.dumps(gen, ensure_ascii=False) + '\n')
        
        self.logger.info("检测水印...")
        detection_results = []
        for gen in attacked_generations:
            detection = self.detect_watermark(gen['completion'])
            detection_results.append(detection)
        
        detected = sum(1 for d in detection_results if d.get('prediction', False))
        attacked_tpr = detected / len(detection_results) if detection_results else 0.0
        
        wd = self.original_results['humaneval'].get('watermark_detection', {})
        original_tpr = wd.get('TPR (FPR < 5%)', 0.0)
        
        report = {
            'metrics': {
                'original': {'TPR': original_tpr},
                'attacked': {'TPR': attacked_tpr, 'success_rate': stats['success']/stats['total']}
            }
        }
        
        report_file = self.output_dir / "attack_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"实验结束。攻击后 TPR: {attacked_tpr:.4f} (原始: {original_tpr:.4f})")
        return report

def main():
    parser = argparse.ArgumentParser(description='AST感知水印攻击')
    parser.add_argument('--dataset_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--output_dir', type=str, default='attack_results', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--use_comments', action='store_true', default=True)
    parser.add_argument('--use_noop', action='store_true', default=True)
    parser.add_argument('--use_dead_code', action='store_true', default=False)
    parser.add_argument('--use_string_noise', action='store_true', default=True)
    parser.add_argument('--only_first_sample', action='store_true', default=False)
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--delta', type=float, default=3.0)
    parser.add_argument('--temperature', type=float, default=0.9)
    
    args = parser.parse_args()
    
    experiment = WatermarkAttackExperiment(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        gamma=args.gamma,
        delta=args.delta,
        temperature=args.temperature
    )
    
    experiment.run_experiment(
        num_samples=args.num_samples,
        use_comments=args.use_comments,
        use_noop=args.use_noop,
        use_dead_code=args.use_dead_code,
        use_string_noise=args.use_string_noise,
        only_first_sample=args.only_first_sample
    )

if __name__ == "__main__":
    main()