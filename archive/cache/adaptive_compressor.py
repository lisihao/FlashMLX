#!/usr/bin/env python3
"""
自适应压缩算法路由器

基于模型架构和场景特征自动选择最佳压缩算法：
- 纯 Transformer → AM (质量 1.0, 速度 +46%)
- 混合架构 → H2O (质量 0.69, 可用)
- 极长序列 → StreamingLLM (专为长序列设计)
"""

from typing import Optional, Dict, Any, Tuple
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_lm.models.compacted_cache import CompactedKVCache
from flashmlx.cache.h2o import H2OCache
from flashmlx.cache.streaming_llm import StreamingLLMCache


class ModelArchitectureDetector:
    """模型架构检测器"""

    @staticmethod
    def detect(model) -> Dict[str, Any]:
        """
        检测模型架构

        Returns:
            dict: {
                'type': 'pure_transformer' | 'hybrid',
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
            'total_layers': total,
            'attention_layer_indices': attention_layers,
            'ssm_layer_indices': ssm_layers
        }


class AdaptiveCompressor:
    """
    自适应压缩算法路由器

    根据模型架构和场景特征自动选择最佳压缩算法
    """

    # 算法优先级（基于质量和性能）
    ALGORITHM_PRIORITY = {
        'pure_transformer': ['AM', 'H2O', 'StreamingLLM'],
        'hybrid': ['H2O', 'StreamingLLM'],  # AM 在混合架构上会崩溃
        'pure_ssm': ['H2O', 'StreamingLLM'],  # SSM 未测试，保守选择
        'unknown': ['H2O', 'StreamingLLM']
    }

    # 算法性能特征（基于实测数据）
    ALGORITHM_CHARACTERISTICS = {
        'AM': {
            'quality': 1.0,  # Llama 3.2 3B 测试
            'speed_boost': 1.46,  # +46%
            'supported_architectures': ['pure_transformer'],
            'max_sequence_length': float('inf')
        },
        'H2O': {
            'quality': 0.69,  # Qwen3.5-0.8B 测试（混合架构下）
            'speed_boost': 1.0,  # 未测速度
            'supported_architectures': ['pure_transformer', 'hybrid'],
            'max_sequence_length': float('inf')
        },
        'StreamingLLM': {
            'quality': 0.66,  # Qwen3.5-0.8B 测试（混合架构下）
            'speed_boost': 1.0,  # 未测速度
            'supported_architectures': ['pure_transformer', 'hybrid'],
            'max_sequence_length': float('inf')
        }
    }

    def __init__(
        self,
        model,
        max_size: int = 4096,
        compression_ratio: float = 2.0,
        long_sequence_threshold: int = 8192,
        enable_fallback: bool = True,
        verbose: bool = False
    ):
        """
        Args:
            model: MLX-LM 模型对象
            max_size: 压缩触发阈值
            compression_ratio: 压缩比例
            long_sequence_threshold: 长序列阈值（超过此值优先使用 StreamingLLM）
            enable_fallback: 是否启用失败回退
            verbose: 是否打印详细信息
        """
        self.model = model
        self.max_size = max_size
        self.compression_ratio = compression_ratio
        self.long_sequence_threshold = long_sequence_threshold
        self.enable_fallback = enable_fallback
        self.verbose = verbose

        # 检测模型架构
        self.architecture = ModelArchitectureDetector.detect(model)

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
        选择压缩算法

        Args:
            sequence_length: 当前序列长度（可选）

        Returns:
            str: 算法名称 ('AM' | 'H2O' | 'StreamingLLM')
        """
        arch_type = self.architecture['type']

        # 特殊情况：极长序列优先使用 StreamingLLM
        if sequence_length and sequence_length > self.long_sequence_threshold:
            if 'StreamingLLM' in self.ALGORITHM_PRIORITY.get(arch_type, []):
                return 'StreamingLLM'

        # 按架构类型选择优先级最高的算法
        priority_list = self.ALGORITHM_PRIORITY.get(arch_type, ['H2O'])
        return priority_list[0] if priority_list else 'H2O'

    def create_cache(self, layer_idx: int) -> KVCache:
        """
        为指定层创建缓存

        Args:
            layer_idx: 层索引

        Returns:
            KVCache: 缓存对象
        """
        # 检查该层是否需要压缩
        if layer_idx not in self.architecture.get('attention_layer_indices', []):
            # SSM 层使用标准 cache
            return KVCache()

        # Attention 层使用压缩 cache
        algorithm = self.selected_algorithm

        try:
            if algorithm == 'AM':
                cache = CompactedKVCache(
                    max_size=self.max_size,
                    compression_ratio=self.compression_ratio
                )
            elif algorithm == 'H2O':
                cache = H2OCache(
                    max_capacity=int(self.max_size / self.compression_ratio),
                    recent_ratio=0.25
                )
            elif algorithm == 'StreamingLLM':
                cache = StreamingLLMCache(
                    max_capacity=int(self.max_size / self.compression_ratio),
                    num_sinks=4
                )
            else:
                # 未知算法，使用标准 cache
                cache = KVCache()

            # 统计
            self.stats['algorithm_usage'][algorithm] = \
                self.stats['algorithm_usage'].get(algorithm, 0) + 1

            return cache

        except Exception as e:
            # 失败回退
            if self.enable_fallback:
                if self.verbose:
                    print(f"⚠️  {algorithm} failed: {e}, falling back to standard KVCache")

                self.stats['fallback_count'] += 1
                self.stats['failures'].append({
                    'algorithm': algorithm,
                    'layer': layer_idx,
                    'error': str(e)
                })

                return KVCache()
            else:
                raise

    def get_recommendation(self) -> Dict[str, Any]:
        """
        获取算法推荐信息

        Returns:
            dict: 推荐信息
        """
        algorithm = self.selected_algorithm
        characteristics = self.ALGORITHM_CHARACTERISTICS.get(algorithm, {})

        return {
            'algorithm': algorithm,
            'architecture': self.architecture['type'],
            'expected_quality': characteristics.get('quality', 0.0),
            'expected_speed_boost': characteristics.get('speed_boost', 1.0),
            'reason': self._get_selection_reason()
        }

    def _get_selection_reason(self) -> str:
        """获取选择原因"""
        arch_type = self.architecture['type']
        algorithm = self.selected_algorithm

        if arch_type == 'pure_transformer':
            if algorithm == 'AM':
                return "纯 Transformer 架构，AM 质量最高 (1.0) 且速度最快 (+46%)"
            elif algorithm == 'H2O':
                return "纯 Transformer 架构，H2O 为备选方案"
        elif arch_type == 'hybrid':
            if algorithm == 'H2O':
                return "混合架构，AM 不可用，H2O 为最佳选择 (质量 0.69)"
            elif algorithm == 'StreamingLLM':
                return "混合架构，长序列场景，StreamingLLM 为最佳选择"
        elif arch_type == 'pure_ssm':
            return "纯 SSM 架构，保守选择 H2O"

        return "未知架构，使用默认算法"

    def _print_initialization(self):
        """打印初始化信息"""
        print("\n" + "="*70)
        print("Adaptive Compressor Initialized")
        print("="*70)
        print(f"Architecture: {self.architecture['type']}")
        print(f"  Attention layers: {self.architecture['attention_layers']}")
        print(f"  SSM layers: {self.architecture['ssm_layers']}")
        print(f"  Total layers: {self.architecture['total_layers']}")
        print()
        print(f"Selected algorithm: {self.selected_algorithm}")
        print(f"  Expected quality: {self.ALGORITHM_CHARACTERISTICS[self.selected_algorithm]['quality']:.2f}")
        print(f"  Expected speed boost: {self.ALGORITHM_CHARACTERISTICS[self.selected_algorithm]['speed_boost']:.2f}x")
        print()
        print(f"Configuration:")
        print(f"  Max size: {self.max_size}")
        print(f"  Compression ratio: {self.compression_ratio}")
        print(f"  Long sequence threshold: {self.long_sequence_threshold}")
        print(f"  Fallback enabled: {self.enable_fallback}")
        print("="*70 + "\n")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'architecture': self.architecture,
            'selected_algorithm': self.selected_algorithm,
            'total_compressions': self.stats['total_compressions'],
            'algorithm_usage': self.stats['algorithm_usage'],
            'fallback_count': self.stats['fallback_count'],
            'failure_count': len(self.stats['failures']),
            'failures': self.stats['failures']
        }

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()

        print("\n" + "="*70)
        print("Adaptive Compressor Statistics")
        print("="*70)
        print(f"Architecture: {stats['architecture']['type']}")
        print(f"Selected algorithm: {stats['selected_algorithm']}")
        print()
        print("Usage:")
        for algo, count in stats['algorithm_usage'].items():
            print(f"  {algo}: {count} times")
        print()
        print(f"Fallback count: {stats['fallback_count']}")
        print(f"Failure count: {stats['failure_count']}")
        print("="*70 + "\n")


def create_adaptive_cache(
    model,
    max_size: int = 4096,
    compression_ratio: float = 2.0,
    long_sequence_threshold: int = 8192,
    enable_fallback: bool = True,
    verbose: bool = False
) -> Tuple[list, AdaptiveCompressor]:
    """
    创建自适应压缩缓存

    Args:
        model: MLX-LM 模型对象
        max_size: 压缩触发阈值
        compression_ratio: 压缩比例
        long_sequence_threshold: 长序列阈值
        enable_fallback: 是否启用失败回退
        verbose: 是否打印详细信息

    Returns:
        tuple: (cache_list, compressor)
    """
    compressor = AdaptiveCompressor(
        model=model,
        max_size=max_size,
        compression_ratio=compression_ratio,
        long_sequence_threshold=long_sequence_threshold,
        enable_fallback=enable_fallback,
        verbose=verbose
    )

    # 创建缓存列表
    from mlx_lm.models.cache import ArraysCache
    cache = ArraysCache(size=len(model.layers))

    for i in range(len(model.layers)):
        cache[i] = compressor.create_cache(i)

    return cache, compressor
