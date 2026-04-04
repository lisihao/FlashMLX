#!/usr/bin/env python3
"""
快速测试 MAC-Attention Patch 是否工作
"""

import sys
sys.path.insert(0, "src")

print("=" * 80)
print("MAC-Attention Patch 测试")
print("=" * 80)
print()

# 测试 1: 导入 flashmlx
print("[1/3] 测试 import flashmlx...")
try:
    import flashmlx
    print("  ✅ 成功")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

# 测试 2: Patch mlx-lm
print("[2/3] 测试 patch_mlx_lm()...")
try:
    flashmlx.patch_mlx_lm()
    print("  ✅ 成功")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 3: Unpatch
print("[3/3] 测试 unpatch_mlx_lm()...")
try:
    flashmlx.unpatch_mlx_lm()
    print("  ✅ 成功")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    sys.exit(1)

print()
print("=" * 80)
print("✅ 所有测试通过！")
print("=" * 80)
print()
print("接下来可以:")
print("  1. 运行演示: python examples/mac_qwen_demo.py")
print("  2. 性能测试: python examples/mac_benchmark.py")
print()
