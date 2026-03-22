#!/usr/bin/env python3
"""
端到端基础测试：验证 CompactedKVCache + Attention Patcher 集成

这个测试使用手动构造的 CompactedKVCache，验证整个流程是否工作。
不包含真实的压缩算法（将在后续实现）。
"""
import mlx.core as mx
from mlx_lm import load
from flashmlx.cache.compacted_kv_cache import create_compacted_cache_list
from flashmlx.cache.attention_patcher import patch_attention_for_compacted_cache


def test_basic_integration():
    """基础集成测试：加载模型 → 应用patch → 使用CompactedKVCache推理"""
    print("=" * 80)
    print("端到端基础测试")
    print("=" * 80)

    # 1. 加载模型
    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    print(f"\n🔄 加载模型: {model_path}")
    model, tokenizer = load(model_path)
    print(f"✓ 模型加载成功")
    print(f"  Layers: {len(model.model.layers)}")

    # 2. 应用 Attention Patch
    print(f"\n🔧 应用 Attention Patch...")
    patch_attention_for_compacted_cache(model, verbose=True)

    # 3. 创建手动构造的 CompactedKVCache
    # 模拟一个已经压缩过的 cache
    print(f"\n🗜️ 创建模拟的 CompactedKVCache...")

    B = 1  # batch size
    num_layers = len(model.model.layers)

    # 获取模型配置
    args = model.model.args
    n_kv_heads = args.num_key_value_heads
    head_dim = args.head_dim

    # 模拟压缩：原始 1024 tokens → 压缩到 512 tokens
    original_len = 1024
    compressed_len = 512
    compression_ratio = original_len / compressed_len

    compacted_cache_data = []
    for layer_idx in range(num_layers):
        # C1: 压缩后的 keys (B, n_kv_heads, compressed_len, head_dim)
        c1 = mx.random.normal(shape=(B, n_kv_heads, compressed_len, head_dim)) * 0.02

        # Beta: 偏置项 (B, n_kv_heads, compressed_len)
        # 使用小的随机值模拟校准偏置
        beta = mx.random.normal(shape=(B, n_kv_heads, compressed_len)) * 0.01

        # C2: 压缩后的 values (B, n_kv_heads, compressed_len, head_dim)
        c2 = mx.random.normal(shape=(B, n_kv_heads, compressed_len, head_dim)) * 0.02

        compacted_cache_data.append((c1, beta, c2))

    cache = create_compacted_cache_list(compacted_cache_data, original_seq_len=original_len)

    print(f"✓ CompactedKVCache 创建成功")
    print(f"  原始长度: {original_len}")
    print(f"  压缩长度: {compressed_len}")
    print(f"  压缩比: {compression_ratio:.2f}x")
    print(f"  Cache list length: {len(cache)}")
    print(f"  Layer 0 Keys shape: {cache[0].keys.shape}")
    print(f"  Layer 0 Offset: {cache[0].offset}")

    # 4. 使用 CompactedKVCache 运行推理
    print(f"\n🧪 测试推理...")

    # 准备输入（新的query tokens）
    prompt = "你好，请介绍一下自己。"
    input_ids = mx.array(tokenizer.encode(prompt))
    input_ids = mx.expand_dims(input_ids, axis=0)  # (1, L)

    print(f"  提示词: {prompt}")
    print(f"  Input shape: {input_ids.shape}")

    try:
        # 运行前向传播
        outputs = model(input_ids, cache=cache)

        print(f"\n✓ 推理成功！")
        print(f"  Output shape: {outputs.shape}")

        # 验证 cache 已更新
        print(f"  Cache[0] offset after: {cache[0].offset}")
        print(f"  Cache[0] keys shape: {cache[0].keys.shape}")

        # 生成一些tokens验证完整性
        print(f"\n🔄 生成 tokens...")
        logits = outputs[0, -1, :]
        next_token = mx.argmax(logits).item()
        decoded = tokenizer.decode([next_token])
        print(f"  下一个token: {decoded}")

        return True

    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_with_normal_cache():
    """对比测试：CompactedKVCache vs 普通 KVCache"""
    print("\n" + "=" * 80)
    print("对比测试：Compacted vs Normal")
    print("=" * 80)

    from mlx_lm.models.cache import KVCache

    model_path = "/Volumes/toshiba/models/qwen3-8b-mlx"
    print(f"\n🔄 加载模型...")
    model, tokenizer = load(model_path)

    # 准备输入
    prompt = "人工智能的未来发展趋势是什么？"
    input_ids = mx.array(tokenizer.encode(prompt))
    input_ids = mx.expand_dims(input_ids, axis=0)

    print(f"\n提示词: {prompt}")
    print(f"Token 数量: {input_ids.shape[1]}")

    # Test 1: 普通 KVCache
    print(f"\n1️⃣ 测试普通 KVCache...")
    normal_cache = [KVCache() for _ in range(len(model.model.layers))]

    try:
        outputs_normal = model(input_ids, cache=normal_cache)
        logits_normal = outputs_normal[0, -1, :]
        next_token_normal = mx.argmax(logits_normal).item()

        print(f"✓ 普通 KVCache 推理成功")
        print(f"  Cache[0] offset: {normal_cache[0].offset}")
        print(f"  下一个token: {tokenizer.decode([next_token_normal])}")
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False

    # Test 2: CompactedKVCache (手动构造)
    print(f"\n2️⃣ 测试 CompactedKVCache...")

    # 应用patch
    patch_attention_for_compacted_cache(model, verbose=False)

    # 创建压缩cache（使用普通cache的数据）
    B = 1
    args = model.model.args
    n_kv_heads = args.num_key_value_heads
    head_dim = args.head_dim
    num_layers = len(model.model.layers)

    # 从普通cache中"压缩"（这里简化为直接使用）
    compacted_cache_data = []
    for layer_idx in range(num_layers):
        # 使用普通cache的keys/values作为"压缩"结果
        c1 = normal_cache[layer_idx].keys
        c2 = normal_cache[layer_idx].values

        # Beta设为0（无偏置）
        seq_len = c1.shape[2]
        beta = mx.zeros((B, n_kv_heads, seq_len))

        compacted_cache_data.append((c1, beta, c2))

    compacted_cache = create_compacted_cache_list(compacted_cache_data)

    # 重新运行推理（从头开始）
    try:
        outputs_compacted = model(input_ids, cache=compacted_cache)
        logits_compacted = outputs_compacted[0, -1, :]
        next_token_compacted = mx.argmax(logits_compacted).item()

        print(f"✓ CompactedKVCache 推理成功")
        print(f"  Cache[0] offset: {compacted_cache[0].offset}")
        print(f"  下一个token: {tokenizer.decode([next_token_compacted])}")

        # 对比结果
        print(f"\n📊 结果对比:")
        print(f"  Normal token:    {next_token_normal}")
        print(f"  Compacted token: {next_token_compacted}")
        print(f"  是否一致: {'✅' if next_token_normal == next_token_compacted else '❌'}")

        return True

    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("端到端基础集成测试")
    print("=" * 80)

    # Test 1: 基础集成
    success1 = test_basic_integration()

    # Test 2: 对比测试
    success2 = test_comparison_with_normal_cache()

    print("\n" + "=" * 80)
    if success1 and success2:
        print("✅ 所有测试通过！")
        print("=" * 80)
        return 0
    else:
        print("❌ 部分测试失败")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())
