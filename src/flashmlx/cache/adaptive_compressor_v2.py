#!/usr/bin/env python3
"""
自适应压缩算法路由器 V2 - 基于模型系列

**重大更新 (2026-03-23)**:
AM 压缩是**模型特定的**，不是架构通用的！

测试结果:
- ✅ Llama 3.2 3B (纯 Transformer): AM 质量 1.0, 速度 +46%
- ❌ Qwen3-8B (纯 Transformer): AM 质量破坏 (13% 相似度), 速度 -6%
- ❌ Qwen3.5 (混合架构): AM 崩溃

结论: 不能仅基于架构选择算法，必须考虑模型系列！
"""

from typing import Optional, Dict, Any, List
import re
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_lm.models.compacted_cache import CompactedKVCache
from flashmlx.cache.h2o import H2OCache
from flashmlx.cache.streaming_llm import StreamingLLMCache


class ModelDetector:
    """模型系列和架构检测器"""

    @staticmethod
    def detect_model_series(model, model_path: Optional[str] = None) -> str:
        """
        检测模型系列

        Args:
            model: MLX-LM 模型对象
            model_path: 模型路径（可选，用于辅助检测）

        Returns:
            str: 模型系列 ('llama' | 'qwen3' | 'qwen3.5' | 'unknown')
        """
        # 方法 1: 从模型路径检测
        if model_path:
            path_lower = model_path.lower()
            if 'llama' in path_lower:
                return 'llama'
            elif 'qwen3.5' in path_lower or 'qwen-3.5' in path_lower:
                return 'qwen3.5'
            elif 'qwen3' in path_lower or 'qwen-3' in path_lower:
                return 'qwen3'

        # 方法 2: 从模型配置检测
        if hasattr(model, 'args'):
            model_type = str(model.args.model_type).lower() if hasattr(model.args, 'model_type') else ''
            if 'llama' in model_type:
                return 'llama'
            elif 'qwen' in model_type:
                # 需要进一步区分 Qwen3 和 Qwen3.5
                # Qwen3.5 使用混合架构（有 SSM 层）
                arch_info = ModelDetector.detect_architecture(model)
                if arch_info['type'] == 'hybrid':
                    return 'qwen3.5'
                else:
                    return 'qwen3'

        # 方法 3: 从架构特征推断
        arch_info = ModelDetector.detect_architecture(model)
        if arch_info['type'] == 'hybrid':
            # 混合架构很可能是 Qwen3.5
            return 'qwen3.5'

        return 'unknown'

    @staticmethod
    def detect_architecture(model) -> Dict[str, Any]:
        """
        检测模型架构类型

        Returns:
            dict: {
                'type': 'pure_transformer' | 'hybrid' | 'unknown',
                'attention_layers': int,
                'ssm_layers': int,
                'total_layers': int
            }
        """
        if not hasattr(model, 'layers'):
            return {
                'type': 'unknown',
                'attention_layers': 0,
                'ssm_layers': 0,
                'total_layers': 0
            }

        attention_layers = []
        ssm_layers = []

        for i, layer in enumerate(model.layers):
            if 'self_attn' in layer.state:
                attention_layers.append(i)
            elif 'linear_attn' in layer.state:
                ssm_layers.append(i)

        total = len(model.layers)
        attn_count = len(attention_layers)
        ssm_count = len(ssm_layers)

        # 判断架构类型
        if ssm_count > 0 and attn_count > 0:
            arch_type = 'hybrid'
        elif attn_count > 0 and ssm_count == 0:
            arch_type = 'pure_transformer'
        elif ssm_count > 0 and attn_count == 0:
            arch_type = 'pure_ssm'
        else:
            arch_type = 'unknown'

        return {
            'type': arch_type,
            'attention_layers': attn_count,
            'ssm_layers': ssm_count,
            'total_layers': total
        }


class AdaptiveCompressorV2:
    """
    自适应压缩算法路由器 V2 - 基于模型系列

    路由策略（基于实测数据）:
    - Llama 系列 → AM (质量 1.0, 速度 +46%) ✅
    - Qwen3 系列 → H2O (AM 破坏质量) ❌
    - Qwen3.5 系列 → H2O (混合架构，AM 崩溃) ❌
    - 未知模型 → H2O (保守选择)
    """

    # 模型系列 → 算法优先级
    MODEL_ALGORITHM_PRIORITY = {
        'llama': ['AM', 'H2O', 'StreamingLLM'],  # AM 验证成功
        'qwen3': ['H2O', 'StreamingLLM'],  # AM 破坏质量
        'qwen3.5': ['H2O', 'StreamingLLM'],  # 混合架构，AM 崩溃
        'unknown': ['H2O', 'StreamingLLM']  # 保守选择
    }

    # 算法性能特征（实测数据）
    ALGORITHM_CHARACTERISTICS = {
        'AM': {
            'tested_models': ['Llama 3.2 3B'],
            'quality_range': '1.0 (Llama) | 0.13 (Qwen3-8B)',
            'speed_boost_range': '+46% (Llama) | -6% (Qwen3-8B)',
            'recommended_for': ['llama'],
            'avoid_for': ['qwen3', 'qwen3.5'],
            'notes': '模型特定！只在 Llama 上验证成功，Qwen3 上破坏质量'
        },
        'H2O': {
            'tested_models': ['Qwen3.5-0.8B (hybrid)'],
            'quality_range': '0.69',
            'speed_boost_range': 'N/A',
            'recommended_for': ['qwen3', 'qwen3.5', 'unknown'],
            'avoid_for': [],
            'notes': '通用兼容，适合大多数模型'
        },
        'StreamingLLM': {
            'tested_models': ['Qwen3.5-0.8B (hybrid)'],
            'quality_range': '0.66',
            'speed_boost_range': 'N/A',
            'recommended_for': ['long_sequences'],
            'avoid_for': [],
            'notes': '专为极长序列设计'
        }
    }

    def __init__(
        self,
        model,
        model_path: Optional[str] = None,
        max_size: int = 4096,
        compression_ratio: float = 2.0,
        long_sequence_threshold: int = 8192,
        enable_fallback: bool = True,
        verbose: bool = False
    ):
        """
        Args:
            model: MLX-LM 模型对象
            model_path: 模型路径（用于辅助检测模型系列）
            max_size: 压缩触发阈值
            compression_ratio: 压缩比例
            long_sequence_threshold: 长序列阈值（超过此值优先使用 StreamingLLM）
            enable_fallback: 是否启用失败回退
            verbose: 是否打印详细信息
        """
        self.model = model
        self.model_path = model_path
        self.max_size = max_size
        self.compression_ratio = compression_ratio
        self.long_sequence_threshold = long_sequence_threshold
        self.enable_fallback = enable_fallback
        self.verbose = verbose

        # 检测模型系列和架构
        self.model_series = ModelDetector.detect_model_series(model, model_path)
        self.architecture = ModelDetector.detect_architecture(model)

        # 选择算法
        self.selected_algorithm = self._select_algorithm()

        # 统计信息
        self.stats = {
            'total_compressions': 0,
            'algorithm_usage': {},
            'fallback_count': 0,
            'failures': []
        }

        if self.verbose:
            self._print_initialization()

    def _select_algorithm(self, sequence_length: Optional[int] = None) -> str:
        """
        选择压缩算法（基于模型系列）

        Args:
            sequence_length: 当前序列长度（可选）

        Returns:
            str: 算法名称 ('AM' | 'H2O' | 'StreamingLLM')
        """
        # 特殊情况：极长序列优先使用 StreamingLLM
        if sequence_length and sequence_length > self.long_sequence_threshold:
            return 'StreamingLLM'

        # 按模型系列选择优先级最高的算法
        priority_list = self.MODEL_ALGORITHM_PRIORITY.get(
            self.model_series,
            self.MODEL_ALGORITHM_PRIORITY['unknown']
        )

        return priority_list[0] if priority_list else 'H2O'

    def _print_initialization(self):
        """打印初始化信息"""
        print(f"{'='*70}")
        print(f"自适应压缩路由器 V2 初始化")
        print(f"{'='*70}")
        print(f"模型系列: {self.model_series}")
        print(f"架构类型: {self.architecture['type']}")
        print(f"总层数: {self.architecture['total_layers']}")
        print(f"  - Attention 层: {self.architecture['attention_layers']}")
        print(f"  - SSM 层: {self.architecture['ssm_layers']}")
        print(f"\n选择算法: {self.selected_algorithm}")

        # 打印算法特性
        if self.selected_algorithm in self.ALGORITHM_CHARACTERISTICS:
            char = self.ALGORITHM_CHARACTERISTICS[self.selected_algorithm]
            print(f"算法特性:")
            print(f"  - 测试模型: {', '.join(char['tested_models'])}")
            print(f"  - 质量范围: {char['quality_range']}")
            print(f"  - 速度提升: {char['speed_boost_range']}")
            print(f"  - 备注: {char['notes']}")
        print(f"{'='*70}\n")

    def get_recommendation(self) -> Dict[str, Any]:
        """
        获取算法推荐信息

        Returns:
            dict: {
                'algorithm': str,
                'model_series': str,
                'architecture_type': str,
                'reason': str,
                'characteristics': dict
            }
        """
        char = self.ALGORITHM_CHARACTERISTICS.get(self.selected_algorithm, {})

        # 生成推荐理由
        if self.model_series == 'llama':
            reason = "Llama 系列在 AM 上验证成功：质量 1.0, 速度 +46%"
        elif self.model_series == 'qwen3':
            reason = "Qwen3 系列在 AM 上质量破坏（13% 相似度），使用 H2O 保守选择"
        elif self.model_series == 'qwen3.5':
            reason = "Qwen3.5 混合架构，AM 崩溃，使用 H2O"
        else:
            reason = "未知模型，使用 H2O 保守选择"

        return {
            'algorithm': self.selected_algorithm,
            'model_series': self.model_series,
            'architecture_type': self.architecture['type'],
            'reason': reason,
            'characteristics': char
        }

    def create_cache(self) -> KVCache:
        """
        创建压缩 cache

        Returns:
            KVCache: 根据选择的算法创建的 cache
        """
        if self.selected_algorithm == 'AM':
            return CompactedKVCache(
                max_size=self.max_size,
                compression_ratio=self.compression_ratio,
                use_quality_path=False  # Fast Path
            )
        elif self.selected_algorithm == 'H2O':
            return H2OCache(
                max_size=self.max_size,
                compression_ratio=self.compression_ratio
            )
        elif self.selected_algorithm == 'StreamingLLM':
            return StreamingLLMCache(
                max_size=self.max_size,
                compression_ratio=self.compression_ratio
            )
        else:
            # Fallback
            return KVCache()
