#!/usr/bin/env python3
"""
验证 Toshiba 盘上的模型可以正常加载
"""

import sys
from pathlib import Path

# 添加 mlx-lm 到路径
sys.path.insert(0, str(Path(__file__).parent / "mlx-lm-source"))

from mlx_lm import load

# 测试模型列表
TEST_MODELS = [
    ("/Volumes/toshiba/models/qwen3.5-35b-mlx", "Qwen3.5 35B MLX"),
    ("/Volumes/toshiba/models/qwen3.5-2b-opus-distilled", "Qwen3.5 2B Opus"),
    ("/Volumes/toshiba/models/qwen3-8b-mlx", "Qwen3 8B MLX"),
]

def verify_model(model_path: str, model_name: str):
    """验证单个模型"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*60}")

    # 检查路径存在
    path = Path(model_path)
    if not path.exists():
        print(f"❌ Path not found: {model_path}")
        return False

    print(f"✅ Path exists")

    # 尝试加载模型
    try:
        print(f"Loading model...")
        model, tokenizer = load(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Layers: {len(model.layers)}")
        return True

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def main():
    print("=" * 60)
    print("Verifying Toshiba Disk Models")
    print("=" * 60)

    # 检查 Toshiba 盘是否挂载
    toshiba_path = Path("/Volumes/toshiba")
    if not toshiba_path.exists():
        print("\n❌ Toshiba disk not mounted at /Volumes/toshiba")
        print("   Please mount the disk and try again")
        sys.exit(1)

    print("\n✅ Toshiba disk mounted")

    # 测试每个模型
    results = {}
    for model_path, model_name in TEST_MODELS:
        results[model_name] = verify_model(model_path, model_name)

    # 总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for model_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {model_name}")

    print(f"\n{'='*60}")
    print(f"Result: {success_count}/{total_count} models verified successfully")
    print(f"{'='*60}")

    if success_count == total_count:
        print("\n✅ All models ready to use!")
    else:
        print(f"\n⚠️  {total_count - success_count} model(s) failed verification")

if __name__ == "__main__":
    main()
