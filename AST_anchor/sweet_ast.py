# -*- coding: utf-8 -*-
"""
SWEET Watermark with Structure-Based AST Anchors
彻底修复同步问题版本：放弃Token位置估算，使用字节范围精准定位
"""

from __future__ import annotations
import torch
from transformers import LogitsProcessor
import hashlib
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

try:
    from tree_sitter import Language, Parser
    import tree_sitter_python
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    print("Warning: tree-sitter not available, falling back to ast module")

from watermark import WatermarkBase

# ---------------------------------------------------------------------------
# 核心解析器：不再返回列表，而是提供查询服务
# ---------------------------------------------------------------------------

class TreeSitterASTParser:
    """基于Tree-sitter的AST结构解析器"""
    
    # 扩充后的锚点类型
    ANCHOR_NODE_TYPES = {
        'function_definition', 'class_definition', 
        'for_statement', 'while_statement', 'if_statement', 'with_statement', 'try_statement',
        'assignment', 'augmented_assignment', 'expression_statement', 
        'call', 'return_statement', 'import_statement', 'import_from_statement'
    }
    
    def __init__(self):
        if not TREE_SITTER_AVAILABLE:
            raise ImportError("tree-sitter required")
        
        PY_LANGUAGE = Language(tree_sitter_python.language())
        try:
            self.parser = Parser(PY_LANGUAGE)
        except TypeError:
            self.parser = Parser()
            self.parser.set_language(PY_LANGUAGE)
        
        self.tree = None

    def parse(self, code: str):
        """解析代码更新内部树结构"""
        if not code.strip(): return
        try:
            code_bytes = bytes(code, "utf8")
            if self.tree is None:
                self.tree = self.parser.parse(code_bytes)
            else:
                self.tree = self.parser.parse(code_bytes, self.tree)
        except Exception:
            pass

    def get_anchor_hash_at_byte_pos(self, byte_pos: int) -> str:
        """
        【核心逻辑】给定一个字节位置，返回它所属的最深层锚点的哈希。
        如果不在任何锚点内，返回空字符串。
        """
        if self.tree is None: return ""

        # 1. 查找覆盖该字节位置的最小节点
        try:
            node = self.tree.root_node.descendant_for_byte_range(byte_pos, byte_pos)
        except:
            return ""

        # 2. 向上寻找最近的“合法锚点”
        anchor_node = None
        while node:
            if node.type in self.ANCHOR_NODE_TYPES:
                anchor_node = node
                break
            node = node.parent
        
        if anchor_node is None:
            return ""

        # 3. 计算该锚点的结构哈希
        # 提取名字
        node_name = ""
        for child in anchor_node.children:
            if child.type == 'identifier':
                # 修复：遇到非法字节时用占位符替换，不要崩
                node_name = child.text.decode('utf8', errors='replace')
                break
        # 计算特征：类型 + 名字 + 父节点路径 (增强唯一性)
        parent_path = []
        curr = anchor_node.parent
        depth = 0
        while curr and depth < 3:
            parent_path.append(curr.type)
            curr = curr.parent
            depth += 1
            
        # 签名：不需要位置信息，只需要结构信息
        # 只要代码结构不变，这个签名在生成端和检测端就是 100% 一致的
        signature = f"{anchor_node.type}:{node_name}|{'->'.join(parent_path)}"
        
        return hashlib.sha256(signature.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 生成端：Logits Processor
# ---------------------------------------------------------------------------

class SweetASTAnchoredLogitsProcessor(WatermarkBase, LogitsProcessor):
    def __init__(self, tokenizer, vocab: list[int] = None, gamma: float = 0.25, delta: float = 2.0, 
                 entropy_threshold: float = 0.0, anchor_weight: float = 0.9, hash_key: int = 15485863, **kwargs):
        super().__init__(vocab=vocab, gamma=gamma, delta=delta, entropy_threshold=entropy_threshold, hash_key=hash_key, **kwargs)
        self.tokenizer = tokenizer
        self.anchor_weight = anchor_weight # 建议设高一点，比如 0.9
        self.ast_parser = TreeSitterASTParser()
        self._last_decoded_text = ""

    def _get_seed_from_structure(self, input_ids: torch.LongTensor) -> int:
        """从当前AST结构获取种子"""
        # 1. 获取当前代码的字节长度（即当前光标位置）
        try:
            # 注意：我们要获取的是"即将生成的Token"的位置，即当前文本的末尾
            current_code = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            byte_pos = len(current_code.encode('utf-8'))
        except:
            return self._fallback_seed(input_ids)

        # 2. 更新AST
        if current_code != self._last_decoded_text:
            self.ast_parser.parse(current_code)
            self._last_decoded_text = current_code

        # 3. 询问解析器：当前光标在哪个锚点里？
        # 我们查询 byte_pos - 1，确保如果在语句末尾也能匹配到
        query_pos = max(0, byte_pos - 1)
        anchor_hash = self.ast_parser.get_anchor_hash_at_byte_pos(query_pos)

        if not anchor_hash:
            return self._fallback_seed(input_ids)

        # 4. 计算种子 (不再依赖 distance)
        anchor_seed = int(anchor_hash, 16) % (2**32)
        token_seed = self._fallback_seed(input_ids)
        
        # 混合：主要依赖结构种子
        mixed_seed = int(self.anchor_weight * anchor_seed + (1 - self.anchor_weight) * token_seed)
        return mixed_seed % (2**32)

    def _fallback_seed(self, input_ids):
        if len(input_ids) >= 1:
            return (self.hash_key * input_ids[-1].item()) % (2**32)
        return self.hash_key % (2**32)

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        seed = self._get_seed_from_structure(input_ids)
        self.rng.manual_seed(seed)
        greenlist_size = int(self.vocab_size * self.gamma)
        return torch.randperm(self.vocab_size, generator=self.rng)[:greenlist_size].tolist()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.rng is None: self.rng = torch.Generator()
        
        batched_greenlist_ids = [self._get_greenlist_ids(seq) for seq in input_ids]
        
        # 标准 SWEET 逻辑
        green_mask = self._calc_greenlist_mask(scores, batched_greenlist_ids)
        raw_probs = torch.softmax(scores, dim=-1)
        ent = -torch.where(raw_probs > 0, raw_probs * raw_probs.log(), raw_probs.new([0.0])).sum(dim=-1)
        entropy_mask = (ent > self.entropy_threshold).view(-1, 1)
        
        scores[green_mask * entropy_mask] += self.delta
        return scores
    
    def _calc_greenlist_mask(self, scores, greenlist_token_ids):
        mask = torch.zeros_like(scores)
        for i, ids in enumerate(greenlist_token_ids):
            mask[i, ids] = 1
        return mask.bool()


# ---------------------------------------------------------------------------
# 检测端：Detector
# ---------------------------------------------------------------------------

class SweetASTAnchoredDetector(WatermarkBase):
    def __init__(self, tokenizer, vocab: list[int] = None, gamma: float = 0.25, delta: float = 2.0, 
                 entropy_threshold: float = 0.0, anchor_weight: float = 0.9, z_threshold: float = 4.0, 
                 hash_key: int = 15485863, **kwargs):
        super().__init__(vocab=vocab, gamma=gamma, delta=delta, entropy_threshold=entropy_threshold, hash_key=hash_key, **kwargs)
        self.tokenizer = tokenizer
        self.anchor_weight = anchor_weight
        self.z_threshold = z_threshold
        self.ast_parser = TreeSitterASTParser()

    def detect(self, input_ids: torch.Tensor = None, entropy: List[float] = None, **kwargs) -> Dict:
        # 初始化 RNG
        if self.rng is None: 
            self.rng = torch.Generator()
            
        # 兼容性处理
        if input_ids is None and 'tokenized_text' in kwargs: input_ids = kwargs['tokenized_text']
        if 'tokenized_prefix' in kwargs:
            prefix_len = len(kwargs['tokenized_prefix'])
        else:
            prefix_len = kwargs.get('prefix_len', 0)
        if prefix_len is None: prefix_len = 0
        prefix_len = max(1, prefix_len)
        
        num_tokens_generated = len(input_ids) - prefix_len
        if num_tokens_generated < 1:
            return {"invalid": True}

        # 解析完整代码 (用于构建 AST)
        full_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        self.ast_parser.parse(full_text)
        
        # 统计锚点数量（用于调试）
        num_anchors = 0
        if self.ast_parser.tree:
            def count_anchors(node):
                count = 1 if node.type in self.ast_parser.ANCHOR_NODE_TYPES else 0
                for child in node.children:
                    count += count_anchors(child)
                return count
            try:
                num_anchors = count_anchors(self.ast_parser.tree.root_node)
            except:
                pass
        
        # -----------------------------------------------------------
        # 【安全增强版】性能优化逻辑
        # -----------------------------------------------------------
        offset_mapping = None
        try:
            # 尝试获取快速映射
            # 注意：这里我们使用 input_ids 对应的 full_text
            encoded = self.tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
            offset_mapping = encoded['offset_mapping']
        except Exception:
            offset_mapping = None

        num_scored = 0
        green_count = 0
        
        # 用于慢速回退模式的累加器
        # 预计算前缀长度
        prefix_text = self.tokenizer.decode(input_ids[:prefix_len], skip_special_tokens=True)
        current_byte_len_slow = len(prefix_text.encode('utf-8'))

        for idx in range(prefix_len, len(input_ids)):
            # -------------------------------------------------------------
            # 混合模式：优先尝试快速查表，失败则回退到慢速解码
            # -------------------------------------------------------------
            current_byte_len = 0
            use_fast = False
            
            # 检查1: offset_mapping 是否存在
            # 检查2: 索引是否越界 (关键修复!)
            map_idx = idx - 1
            if offset_mapping is not None and map_idx < len(offset_mapping):
                if map_idx >= 0:
                    current_byte_len = offset_mapping[map_idx][1]
                else:
                    current_byte_len = 0
                use_fast = True
            
            if not use_fast:
                # 【慢速回退】模拟逐个 Token 生成 (确保绝对准确且不报错)
                # 当遇到特殊字符或越界时，会自动走到这里
                token_text = self.tokenizer.decode([input_ids[idx]], skip_special_tokens=True)
                # 更新累加器
                current_byte_len_slow += len(token_text.encode('utf-8'))
                current_byte_len = current_byte_len_slow
            
            # -------------------------------------------------------------
            # 核心检测逻辑 (保持不变)
            # -------------------------------------------------------------
            query_pos = max(0, current_byte_len - 1)
            anchor_hash = self.ast_parser.get_anchor_hash_at_byte_pos(query_pos)
            
            # 计算种子（与生成时完全一致）
            if anchor_hash:
                anchor_seed = int(anchor_hash, 16) % (2**32)
                prev_token = input_ids[idx-1].item() if idx > 0 else 0
                token_seed = (self.hash_key * prev_token) % (2**32)
                seed = int(self.anchor_weight * anchor_seed + (1 - self.anchor_weight) * token_seed) % (2**32)
            else:
                prev_token = input_ids[idx-1].item() if idx > 0 else 0
                seed = (self.hash_key * prev_token) % (2**32)
            
            # 检查是否计分 (熵阈值)
            if entropy[idx] <= self.entropy_threshold:
                continue
                
            num_scored += 1
            
            # 检查绿名单
            self.rng.manual_seed(seed)
            perm = torch.randperm(self.vocab_size, generator=self.rng)
            greenlist = perm[:int(self.vocab_size * self.gamma)]
            
            if input_ids[idx] in greenlist:
                green_count += 1

        z_score = self._compute_z_score(green_count, num_scored)
        p_value = self._compute_p_value(z_score)
        
        return {
            "num_tokens_generated": num_tokens_generated,
            "num_tokens_scored": num_scored,
            "num_green_tokens": green_count,
            "watermarking_fraction": num_scored / num_tokens_generated if num_tokens_generated > 0 else 0.0,
            "green_fraction": green_count / num_scored if num_scored > 0 else 0.0,
            "z_score": z_score,
            "p_value": p_value,
            "prediction": z_score > self.z_threshold,
            "num_anchors": num_anchors,
        }

    def _compute_z_score(self, green, total):
        if total == 0: return -100.0
        p = self.gamma
        variance = p * (1 - p) * total
        if variance == 0: return 0.0
        return (green - p * total) / (variance ** 0.5)
    
    def _compute_p_value(self, z_score: float) -> float:
        try:
            import scipy.stats
            return 1 - scipy.stats.norm.cdf(z_score)
        except:
            return 0.0