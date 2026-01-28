# -*- coding: utf-8 -*-
import os
import pathlib

# DS-1000 ���ݼ�·��
DATA_DIR = pathlib.Path("lm_eval/tasks/ds/ds1000_data")

def main():
    if not DATA_DIR.exists():
        print(f"? �Ҳ���Ŀ¼: {DATA_DIR}")
        return

    print(f"? �����޸������ļ�...")
    count = 0
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file == "test_code.py":
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # �滻��ʱ�� parser ģ��
                if "import parser" in content:
                    new_content = content.replace("import parser", "import ast")
                    new_content = new_content.replace("parser.suite", "ast.parse")
                    
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    count += 1
    
    print(f"? �޸���ɣ����޸��� {count} ���ļ���")

if __name__ == "__main__":
    main()