#!/usr/bin/env python3
"""
测试真实文本的 hit rate
"""

import sys
sys.path.insert(0, "src")
sys.path.insert(0, "mlx-lm-source")

from mlx_lm import load, generate
import flashmlx
import mlx.core as mx

# Patch and load
model, tokenizer = load("/Volumes/toshiba/models/qwen3-8b-mlx")
flashmlx.patch_mlx_lm()

# 真实的长文本（多样化内容）
prompt = """
人工智能的发展历程可以追溯到20世纪50年代。当时，科学家们开始思考如何让机器具有智能。
经过几十年的发展，人工智能经历了多次浪潮。第一次浪潮是符号主义，第二次是连接主义，
第三次则是深度学习。深度学习的突破源于三个关键因素：大数据、强大的计算能力和改进的算法。
神经网络的概念最早可以追溯到1943年。McCulloch和Pitts提出了第一个数学模型。
然而，真正的突破发生在2012年，AlexNet在ImageNet比赛中取得了惊人的成绩。
从那时起，深度学习在计算机视觉、自然语言处理、语音识别等领域取得了巨大进展。
Transformer架构的提出更是革命性的。它完全基于注意力机制，抛弃了传统的循环结构。
BERT、GPT等模型的出现，让自然语言理解和生成达到了前所未有的水平。
大语言模型的训练需要海量的数据和计算资源。GPT-3使用了45TB的文本数据进行训练。
这些模型展现出了惊人的少样本学习能力，甚至零样本学习能力。
然而，大模型也面临着诸多挑战。首先是计算成本问题，训练一个大模型需要数百万美元。
其次是推理效率问题，部署大模型需要强大的硬件支持。
此外，模型的安全性和可靠性也是重要的考虑因素。如何防止模型产生有害内容，
如何确保模型的预测是可解释的，这些都是亟待解决的问题。
未来的发展方向包括模型压缩、知识蒸馏、稀疏化等技术。
同时，多模态学习、持续学习、因果推理等也是重要的研究方向。
""" * 100  # 重复 100 次，增加上下文长度

print("生成文本（真实多样化内容）...")
output = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=20,
    verbose=False,
)

# 查看 MAC 统计
mac = flashmlx.patch._global_mac_wrapper
if mac:
    print(f"\nMAC Statistics:")
    print(f"  Ring cache filled: {mac.ring_cache.filled} / {mac.cache_capacity}")

    # 多次测试 match
    hit_rates = []
    for _ in range(10):
        q = mx.random.normal((32, 128)).astype(mx.bfloat16)
        q_norm = mac._normalize_query(q)
        hit, left_start = mac._match(q_norm)
        hit_rates.append(float(hit.mean()))

    avg_hit_rate = sum(hit_rates) / len(hit_rates)
    print(f"  Average hit rate: {avg_hit_rate:.1%}")
    print(f"  (Sampled 10 random queries)")
