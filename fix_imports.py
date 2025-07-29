#!/usr/bin/env python3
"""
批量修复相对导入脚本
将所有相对导入改为绝对导入
"""
import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """修复单个文件中的相对导入"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复相对导入模式
    patterns = [
        (r'from \.\.shared\.', 'from shared.'),
        (r'from \.\.\.shared\.', 'from shared.'),
        (r'from \.\.', 'from '),
        (r'from \.', 'from '),
    ]
    
    original_content = content
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 修复: {file_path}")
        return True
    return False

def main():
    """主函数"""
    backend_path = Path("backend")
    
    # 需要修复的文件列表
    files_to_fix = [
        "auth_service/models/user.py",
        "auth_service/services/user_service.py", 
        "auth_service/routes.py",
        "auth_service/utils/auth.py",
        "feast_service/manager.py",
        "feast_service/routes.py",
        "feast_service/training_set_manager.py",
        "pipeline_service/routes.py",
        "pipeline_service/models.py",
        "doris_connector/connection.py",
        "doris_connector/routes.py",
        "doris_connector/query_service.py",
        "monitor_service/routes.py",
        "shared/database.py"
    ]
    
    fixed_count = 0
    for file_path in files_to_fix:
        full_path = backend_path / file_path
        if full_path.exists():
            if fix_imports_in_file(full_path):
                fixed_count += 1
    
    print(f"\n🎉 修复完成！共修复了 {fixed_count} 个文件")

if __name__ == "__main__":
    main() 