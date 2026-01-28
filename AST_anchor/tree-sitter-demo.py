# -*- coding: utf-8 -*-
"""
Tree-sitter增量解析演示
展示如何使用Tree-sitter进行实时代码解析和AST构建
"""

import sys

# 尝试导入tree-sitter
try:
    import tree_sitter
    from tree_sitter import Language, Parser
except ImportError as e:
    print("=" * 80)
    print("错误: 未安装tree-sitter")
    print("=" * 80)
    print()
    print("请运行以下命令安装:")
    print("  pip install tree-sitter tree-sitter-python")
    print()
    print(f"详细错误: {e}")
    sys.exit(1)

# 尝试加载Python语言
PY_LANGUAGE = None
try:
    # 方法1: 如果已经编译好的.so文件
    PY_LANGUAGE = Language('build/my-languages.so', 'python')
    print("✓ 使用编译的语言文件")
except:
    try:
        # 方法2: 如果有预编译的包
        import tree_sitter_python
        PY_LANGUAGE = Language(tree_sitter_python.language())
        print("✓ 使用tree-sitter-python包")
    except ImportError as e:
        print("=" * 80)
        print("错误: 未安装tree-sitter-python")
        print("=" * 80)
        print()
        print("请运行以下命令安装:")
        print("  pip install tree-sitter-python")
        print()
        print(f"详细错误: {e}")
        sys.exit(1)
    except Exception as e:
        print("=" * 80)
        print("错误: 无法加载Python语言支持")
        print("=" * 80)
        print()
        print(f"详细错误: {e}")
        sys.exit(1)


class TreeSitterIncrementalParser:
    """
    Tree-sitter增量解析器演示
    展示实时解析和AST遍历
    """
    
    def __init__(self):
        # 兼容新旧API
        try:
            # 新API (tree-sitter >= 0.21.0)
            self.parser = Parser(PY_LANGUAGE)
        except TypeError:
            # 旧API (tree-sitter < 0.21.0)
            self.parser = Parser()
            self.parser.set_language(PY_LANGUAGE)
        self.tree = None
        
    def parse(self, code: str):
        """解析代码"""
        code_bytes = bytes(code, "utf8")
        self.tree = self.parser.parse(code_bytes)
        return self.tree
    
    def incremental_parse(self, old_code: str, new_code: str, edit_start: int, edit_end: int):
        """
        增量解析演示
        
        Args:
            old_code: 旧代码
            new_code: 新代码
            edit_start: 编辑起始位置（字节）
            edit_end: 编辑结束位置（字节）
        """
        # 先解析旧代码
        old_tree = self.parser.parse(bytes(old_code, "utf8"))
        
        print(f"旧代码AST节点数: {self._count_nodes(old_tree.root_node)}")
        
        # 告诉parser发生了编辑
        # 这允许parser重用未改变部分的AST
        old_tree.edit(
            start_byte=edit_start,
            old_end_byte=edit_end,
            new_end_byte=edit_start + (len(new_code) - len(old_code)),
            start_point=(0, edit_start),  # 简化，实际需要计算行列
            old_end_point=(0, edit_end),
            new_end_point=(0, edit_start + (len(new_code) - len(old_code)))
        )
        
        # 增量解析新代码
        new_tree = self.parser.parse(bytes(new_code, "utf8"), old_tree)
        
        print(f"新代码AST节点数: {self._count_nodes(new_tree.root_node)}")
        print(f"增量解析完成！Tree-sitter重用了未改变的AST节点")
        
        return new_tree
    
    def _count_nodes(self, node):
        """递归计数节点"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def print_tree(self, node=None, indent=0, max_depth=10):
        """打印AST树结构"""
        if node is None:
            node = self.tree.root_node
        
        # 限制深度，避免输出过长
        if indent >= max_depth:
            print("  " * indent + "├─ ...")
            return
        
        try:
            # 打印当前节点
            node_type = node.type
            node_text = node.text.decode('utf8')[:50] if node.text else ""
            
            print("  " * indent + f"├─ {node_type:20s}", end="")
            
            # 对于某些节点显示文本内容
            if node_type in ['identifier', 'string', 'integer', 'float']:
                print(f" = '{node_text}'")
            else:
                print()
            
            # 递归打印子节点
            for child in node.children:
                self.print_tree(child, indent + 1, max_depth)
        except Exception as e:
            print(f"  (打印节点时出错: {e})")
    
    def extract_function_definitions(self, node=None):
        """提取所有函数定义（递归遍历AST）"""
        if node is None:
            node = self.tree.root_node
        
        functions = []
        
        # 检查当前节点是否是函数定义
        if node.type == 'function_definition':
            # 查找函数名
            func_name = None
            for child in node.children:
                if child.type == 'identifier':
                    func_name = child.text.decode('utf8')
                    break
            
            functions.append({
                'name': func_name,
                'start_byte': node.start_byte,
                'end_byte': node.end_byte,
                'start_point': node.start_point,
                'end_point': node.end_point
            })
        
        # 递归处理子节点
        for child in node.children:
            functions.extend(self.extract_function_definitions(child))
        
        return functions
    
    def extract_control_structures(self, node=None):
        """提取控制结构（for, while, if等）"""
        if node is None:
            node = self.tree.root_node
        
        structures = []
        
        # 关键控制结构类型
        control_types = {
            'for_statement': 'For循环',
            'while_statement': 'While循环',
            'if_statement': 'If条件',
            'with_statement': 'With语句',
            'try_statement': 'Try语句'
        }
        
        if node.type in control_types:
            structures.append({
                'type': control_types[node.type],
                'node_type': node.type,
                'start_byte': node.start_byte,
                'start_point': node.start_point,
                'text': node.text.decode('utf8')[:100]  # 前100字符
            })
        
        # 递归
        for child in node.children:
            structures.extend(self.extract_control_structures(child))
        
        return structures


def demo_basic_parsing():
    """演示1: 基本解析"""
    print("=" * 80)
    print("演示1: Tree-sitter基本解析")
    print("=" * 80)
    print()
    
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    print("源代码:")
    print(code)
    print()
    
    try:
        parser = TreeSitterIncrementalParser()
        tree = parser.parse(code)
        
        print("AST树结构:")
        print("-" * 80)
        parser.print_tree()
        print()
        print("✓ 解析成功！")
        print()
    except Exception as e:
        print(f"✗ 解析失败: {e}")
        import traceback
        traceback.print_exc()
        print()


def demo_extract_functions():
    """演示2: 提取函数定义"""
    print("=" * 80)
    print("演示2: 提取函数定义")
    print("=" * 80)
    print()
    
    code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    return quick_sort([x for x in arr if x < pivot]) + [pivot] + quick_sort([x for x in arr if x > pivot])
"""
    
    print("源代码:")
    print(code)
    print()
    
    parser = TreeSitterIncrementalParser()
    parser.parse(code)
    
    functions = parser.extract_function_definitions()
    
    print(f"找到 {len(functions)} 个函数定义:")
    print("-" * 80)
    for func in functions:
        print(f"函数名: {func['name']}")
        print(f"  位置: 字节 {func['start_byte']}-{func['end_byte']}")
        print(f"  行列: ({func['start_point'][0]}, {func['start_point'][1]}) - "
              f"({func['end_point'][0]}, {func['end_point'][1]})")
        print()


def demo_extract_control_structures():
    """演示3: 提取控制结构（水印锚点候选）"""
    print("=" * 80)
    print("演示3: 提取控制结构（AST锚点）")
    print("=" * 80)
    print()
    
    code = """
def process_data(data):
    result = []
    
    for item in data:
        if item > 0:
            result.append(item * 2)
    
    while len(result) < 10:
        result.append(0)
    
    try:
        with open('output.txt', 'w') as f:
            f.write(str(result))
    except IOError:
        pass
    
    return result
"""
    
    print("源代码:")
    print(code)
    print()
    
    parser = TreeSitterIncrementalParser()
    parser.parse(code)
    
    structures = parser.extract_control_structures()
    
    print(f"找到 {len(structures)} 个控制结构（可作为AST锚点）:")
    print("-" * 80)
    for i, struct in enumerate(structures):
        print(f"[{i+1}] {struct['type']} ({struct['node_type']})")
        print(f"    位置: 字节 {struct['start_byte']}, 行 {struct['start_point'][0]}")
        print(f"    内容预览: {struct['text'][:60]}...")
        print()


def demo_incremental_parsing():
    """演示4: 增量解析（最重要！）"""
    print("=" * 80)
    print("演示4: Tree-sitter增量解析")
    print("=" * 80)
    print()
    
    # 原始代码
    old_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    # 在中间插入一行注释
    new_code = """
def fibonacci(n):
    # This is a comment
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    print("旧代码:")
    print(old_code)
    print()
    
    print("新代码（插入了注释）:")
    print(new_code)
    print()
    
    parser = TreeSitterIncrementalParser()
    
    print("执行增量解析...")
    print("-" * 80)
    
    # 找到编辑位置（简化，实际应该精确计算）
    edit_start = old_code.find("if n <= 1:")
    edit_end = edit_start
    
    new_tree = parser.incremental_parse(old_code, new_code, edit_start, edit_end)
    
    print()
    print("增量解析的优势：")
    print("  ✓ 只重新解析改变的部分")
    print("  ✓ 重用未改变部分的AST节点")
    print("  ✓ 解析速度快（适合实时生成场景）")
    print()


def demo_watermark_anchor_extraction():
    """演示5: 为水印提取AST锚点"""
    print("=" * 80)
    print("演示5: 为水印系统提取AST锚点")
    print("=" * 80)
    print()
    
    code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""
    
    print("源代码:")
    print(code)
    print()
    
    parser = TreeSitterIncrementalParser()
    parser.parse(code)
    
    # 提取函数和控制结构
    functions = parser.extract_function_definitions()
    structures = parser.extract_control_structures()
    
    print("可用作水印锚点的AST节点:")
    print("-" * 80)
    
    print("\n📍 函数定义锚点:")
    for func in functions:
        print(f"  • FunctionDef: {func['name']} (字节位置: {func['start_byte']})")
    
    print("\n📍 控制结构锚点:")
    for struct in structures:
        print(f"  • {struct['type']} (字节位置: {struct['start_byte']})")
    
    print()
    print("这些锚点的特点:")
    print("  ✓ 位置相对稳定（即使插入死代码）")
    print("  ✓ 结构明确（函数、循环、条件）")
    print("  ✓ 可用于同步水印种子")
    print()
    
    # 模拟水印种子计算
    print("模拟水印种子计算:")
    print("-" * 80)
    
    import hashlib
    
    all_anchors = []
    
    for func in functions:
        context = f"function_definition:{func['name']}"
        seed = int(hashlib.sha256(context.encode()).hexdigest()[:16], 16) % (2**32)
        all_anchors.append({
            'type': 'FunctionDef',
            'name': func['name'],
            'position': func['start_byte'],
            'seed': seed
        })
    
    for struct in structures:
        context = f"{struct['node_type']}:{struct['start_byte']}"
        seed = int(hashlib.sha256(context.encode()).hexdigest()[:16], 16) % (2**32)
        all_anchors.append({
            'type': struct['node_type'],
            'position': struct['start_byte'],
            'seed': seed
        })
    
    # 按位置排序
    all_anchors.sort(key=lambda x: x['position'])
    
    for anchor in all_anchors:
        if 'name' in anchor:
            print(f"位置 {anchor['position']:4d} | {anchor['type']:20s} | "
                  f"名称: {anchor['name']:15s} | 种子: {anchor['seed']}")
        else:
            print(f"位置 {anchor['position']:4d} | {anchor['type']:20s} | "
                  f"种子: {anchor['seed']}")
    
    print()
    print("在生成过程中，每到达一个锚点位置，就用该锚点的种子重置随机数生成器")
    print("这样即使token序列改变，只要AST结构不变，水印就能保持！")
    print()


def main():
    """运行所有演示"""
    demos = [
        ("基本解析", demo_basic_parsing),
        ("提取函数定义", demo_extract_functions),
        ("提取控制结构", demo_extract_control_structures),
        ("增量解析", demo_incremental_parsing),
        ("水印锚点提取", demo_watermark_anchor_extraction)
    ]
    
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Tree-sitter 演示程序" + " " * 38 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    for i, (name, demo_func) in enumerate(demos):
        print(f"{i+1}. {name}")
    print(f"{len(demos)+1}. 运行全部演示")
    print("0. 退出")
    print()
    
    choice = input("请选择演示 (0-6): ").strip()
    
    if choice == '0':
        return
    elif choice == str(len(demos) + 1):
        # 运行全部
        for name, demo_func in demos:
            demo_func()
            input("\n按回车继续下一个演示...")
            print("\n\n")
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(demos):
                demos[idx][1]()
            else:
                print("无效选择")
        except ValueError:
            print("无效输入: 请输入数字")
        except Exception as e:
            print(f"执行演示时出错: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

