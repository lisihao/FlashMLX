#!/usr/bin/env python3
"""
批量更新所有测试脚本的模型路径到 Toshiba 盘
"""

import os
import re
from pathlib import Path

# 模型路径映射
MODEL_PATH_MAPPING = {
    # MLX Community models
    "mlx-community/Qwen3.5-35B-A3B-6bit": "/Volumes/toshiba/models/qwen3.5-35b-mlx",
    "mlx-community/Qwen3.5-35B-A3B": "/Volumes/toshiba/models/qwen3.5-35b-mlx",
    "mlx-community/Qwen3-8B": "/Volumes/toshiba/models/qwen3-8b-mlx",
    "mlx-community/Llama-3.2-3B-Instruct-4bit": "/Volumes/toshiba/models/llama-3.2-3b-mlx",

    # Local paths
    "qwen3.5-2b-opus-distilled": "/Volumes/toshiba/models/qwen3.5-2b-opus-distilled",
    "qwen3.5-0.8b-opus-distilled": "/Volumes/toshiba/models/qwen3.5-0.8b-opus-distilled",
}

def update_file(file_path: Path):
    """更新单个文件的模型路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        modified = False

        # 替换所有匹配的路径
        for old_path, new_path in MODEL_PATH_MAPPING.items():
            if old_path in content:
                # 处理带引号的路径
                content = content.replace(f'"{old_path}"', f'"{new_path}"')
                content = content.replace(f"'{old_path}'", f"'{new_path}'")
                modified = True

        if modified and content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated: {file_path}")
            return True
        else:
            return False

    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def main():
    """主函数：批量更新所有脚本"""
    benchmarks_dir = Path("benchmarks")

    if not benchmarks_dir.exists():
        print("❌ benchmarks/ directory not found")
        return

    print("=" * 60)
    print("Updating model paths to Toshiba disk")
    print("=" * 60)
    print()

    updated_files = []
    unchanged_files = []

    # 遍历所有 Python 脚本
    for py_file in benchmarks_dir.glob("*.py"):
        if update_file(py_file):
            updated_files.append(py_file.name)
        else:
            unchanged_files.append(py_file.name)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✅ Updated: {len(updated_files)} files")
    if updated_files:
        for f in updated_files:
            print(f"   - {f}")

    print()
    print(f"⏭️  Unchanged: {len(unchanged_files)} files")

    print()
    print("✅ All model paths updated to /Volumes/toshiba/models/")

if __name__ == "__main__":
    main()
