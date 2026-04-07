#!/usr/bin/env python3
"""
Adaptive Configuration Generator for FlashMLX

Automatically generates optimal KV cache configurations based on:
- Model architecture fingerprint
- Historical optimization results
- Target context length and optimization goal

Uses transfer learning from similar models to provide cold-start recommendations.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import hashlib

from mlx_lm import load
from flashmlx_meta_harness import BenchmarkConfig


@dataclass
class ModelFingerprint:
    """
    Model architecture fingerprint for similarity matching.
    """

    model_id: str
    architecture_type: str  # 'pure_transformer', 'hybrid_ssm_attention', 'mla'
    num_layers: int
    num_attention_layers: int  # For hybrid architectures
    hidden_size: int
    head_dim: int
    num_kv_heads: int
    num_q_heads: int
    kv_q_ratio: float  # num_kv_heads / num_q_heads
    param_count_b: float  # In billions
    is_mla: bool  # Multi-head Latent Attention (DeepSeek style)
    is_gqa: bool  # Grouped Query Attention

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelFingerprint':
        """Create from dictionary."""
        return cls(**data)

    def similarity_score(self, other: 'ModelFingerprint') -> float:
        """
        Compute similarity score with another fingerprint.

        Returns
        -------
        score : float
            Similarity score in [0, 1], higher means more similar
        """
        score = 0.0
        weights_sum = 0.0

        # Architecture type (most important)
        if self.architecture_type == other.architecture_type:
            score += 0.3
        weights_sum += 0.3

        # MLA match (critical for KV cache behavior)
        if self.is_mla == other.is_mla:
            score += 0.2
        weights_sum += 0.2

        # Layer count similarity
        layer_ratio = min(self.num_layers, other.num_layers) / max(self.num_layers, other.num_layers)
        score += 0.15 * layer_ratio
        weights_sum += 0.15

        # KV/Q ratio similarity (affects KV cache size)
        ratio_diff = abs(self.kv_q_ratio - other.kv_q_ratio)
        ratio_similarity = max(0, 1 - ratio_diff / 0.5)  # 0.5 as normalization factor
        score += 0.15 * ratio_similarity
        weights_sum += 0.15

        # Parameter count similarity
        param_ratio = min(self.param_count_b, other.param_count_b) / max(self.param_count_b, other.param_count_b)
        score += 0.1 * param_ratio
        weights_sum += 0.1

        # Head dim similarity
        if self.head_dim == other.head_dim:
            score += 0.1
        weights_sum += 0.1

        return score / weights_sum if weights_sum > 0 else 0.0


@dataclass
class ConfigRecommendation:
    """
    Configuration recommendation with confidence and rationale.
    """

    config: BenchmarkConfig
    confidence: float  # [0, 1]
    rationale: List[str]  # Human-readable explanation
    source: str  # 'history', 'architecture_rule', 'cold_start'
    expected_pareto_score: Optional[float] = None


class AdaptiveConfigGenerator:
    """
    Generates optimal KV cache configurations based on model fingerprint
    and historical optimization results.
    """

    def __init__(self, model_cards_dir: str = "model_cards", fingerprints_file: str = "model_fingerprints.json"):
        """
        Parameters
        ----------
        model_cards_dir : str
            Directory containing model card JSON files
        fingerprints_file : str
            JSON file storing model fingerprints
        """
        self.model_cards_dir = Path(model_cards_dir)
        self.fingerprints_file = Path(fingerprints_file)

        # Load fingerprints database
        self.fingerprints: Dict[str, ModelFingerprint] = {}
        if self.fingerprints_file.exists():
            with open(self.fingerprints_file) as f:
                data = json.load(f)
                self.fingerprints = {k: ModelFingerprint.from_dict(v) for k, v in data.items()}

        # Load model cards (index by both model_id and model_path)
        self.model_cards: Dict[str, Dict] = {}
        self.model_cards_by_path: Dict[str, Dict] = {}
        if self.model_cards_dir.exists():
            for card_file in self.model_cards_dir.glob("*.json"):
                with open(card_file) as f:
                    card = json.load(f)
                    model_id = card.get('model_id')
                    model_path = card.get('model_path')
                    if model_id:
                        self.model_cards[model_id] = card
                    if model_path:
                        # Index by folder name from path
                        path_key = Path(model_path).name
                        self.model_cards_by_path[path_key] = card

    def analyze_model(self, model_path: str) -> ModelFingerprint:
        """
        Analyze model architecture and create fingerprint.

        Parameters
        ----------
        model_path : str
            Path to MLX model

        Returns
        -------
        fingerprint : ModelFingerprint
            Model architecture fingerprint
        """
        # Generate model_id from path
        model_id = Path(model_path).name

        # Check if fingerprint already exists
        if model_id in self.fingerprints:
            return self.fingerprints[model_id]

        # Load model config first (faster)
        print(f"Analyzing model architecture: {model_path}")
        config_path = Path(model_path) / "config.json"
        model_config = {}
        if config_path.exists():
            with open(config_path) as f:
                model_config = json.load(f)

        # Load model to analyze architecture
        model, tokenizer = load(model_path)

        # Detect architecture type and extract parameters
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Standard Transformer
            layers = model.model.layers
            architecture_type = 'pure_transformer'
            num_attention_layers = len(layers)
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
            # MoE / Hybrid architecture
            layers = model.language_model.model.layers
            # Count attention layers
            num_attention_layers = sum(1 for layer in layers if hasattr(layer, 'self_attn'))
            architecture_type = 'hybrid_ssm_attention' if num_attention_layers < len(layers) else 'pure_transformer'
        else:
            raise ValueError(f"Cannot analyze model structure: {model_path}")

        num_layers = len(layers)

        # Extract first attention layer for analysis
        attention_layer = None
        for layer in layers:
            if hasattr(layer, 'self_attn'):
                attention_layer = layer.self_attn
                break

        if attention_layer is None:
            raise ValueError("No attention layer found in model")

        # Extract dimensions (try config first, then model attributes)
        num_kv_heads = model_config.get('num_key_value_heads') or \
                       model_config.get('num_kv_heads') or \
                       getattr(attention_layer, 'n_kv_heads', None) or \
                       getattr(attention_layer, 'num_kv_heads', None) or \
                       getattr(attention_layer, 'num_key_value_heads', None)

        num_q_heads = model_config.get('num_attention_heads') or \
                      model_config.get('num_heads') or \
                      getattr(attention_layer, 'n_heads', None) or \
                      getattr(attention_layer, 'num_heads', None) or \
                      getattr(attention_layer, 'num_attention_heads', None)

        # Try config first, then model attributes
        head_dim = model_config.get('head_dim') or getattr(attention_layer, 'head_dim', None)
        hidden_size = model_config.get('hidden_size') or \
                      model_config.get('dim') or \
                      getattr(model.model if hasattr(model, 'model') else model, 'hidden_size', None) or \
                      getattr(model.model if hasattr(model, 'model') else model, 'dim', None)

        # If we can't get head_dim directly, compute it
        if head_dim is None and hidden_size is not None and num_q_heads is not None:
            head_dim = hidden_size // num_q_heads

        # Detect MLA (Multi-head Latent Attention)
        # MLA typically has rope_dims or uses compression
        is_mla = hasattr(attention_layer, 'rope_dims') or \
                 hasattr(attention_layer, 'qk_nope_head_dim')

        # GQA: num_kv_heads < num_q_heads
        is_gqa = num_kv_heads is not None and num_q_heads is not None and num_kv_heads < num_q_heads

        # Estimate parameter count (very rough)
        # Total params ≈ 12 * num_layers * hidden_size^2 (Transformer rule of thumb)
        param_count_b = (12 * num_layers * (hidden_size ** 2)) / 1e9 if hidden_size else 0.0

        # Create fingerprint
        fingerprint = ModelFingerprint(
            model_id=model_id,
            architecture_type=architecture_type,
            num_layers=num_layers,
            num_attention_layers=num_attention_layers,
            hidden_size=hidden_size or 0,
            head_dim=head_dim or 0,
            num_kv_heads=num_kv_heads or 0,
            num_q_heads=num_q_heads or 0,
            kv_q_ratio=num_kv_heads / num_q_heads if (num_kv_heads and num_q_heads) else 1.0,
            param_count_b=param_count_b,
            is_mla=is_mla,
            is_gqa=is_gqa,
        )

        # Save fingerprint
        self.fingerprints[model_id] = fingerprint
        self._save_fingerprints()

        del model, tokenizer

        return fingerprint

    def query_similar_models(self, fingerprint: ModelFingerprint, top_k: int = 3) -> List[Tuple[str, float, Dict]]:
        """
        Query model cards database for similar models.

        Parameters
        ----------
        fingerprint : ModelFingerprint
            Query fingerprint
        top_k : int
            Number of similar models to return

        Returns
        -------
        similar_models : List[Tuple[str, float, Dict]]
            List of (model_id, similarity_score, model_card)
        """
        similarities = []

        for model_id, fp in self.fingerprints.items():
            if model_id == fingerprint.model_id:
                continue  # Skip self

            similarity = fingerprint.similarity_score(fp)

            # Try to find model card by model_id or path
            card = self.model_cards.get(model_id) or self.model_cards_by_path.get(model_id)
            if card:
                similarities.append((model_id, similarity, card))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def adapt_for_context(self, context_len: int, target: str, base_config: Dict) -> Tuple[BenchmarkConfig, List[str]]:
        """
        Adapt configuration for specific context length and target.

        Parameters
        ----------
        context_len : int
            Target context length
        target : str
            Optimization target ('balanced', 'speed', 'memory', 'quality')
        base_config : Dict
            Base configuration from similar model

        Returns
        -------
        config : BenchmarkConfig
            Adapted configuration
        rationale : List[str]
            Explanation of adaptations
        """
        rationale = []
        config_dict = base_config.copy()

        # Normalize field names (model cards use different names)
        if 'warm_bits' in config_dict and 'kv_warm_bits' not in config_dict:
            config_dict['kv_warm_bits'] = config_dict['warm_bits']

        # Infer kv_cache from strategy if not present
        if 'kv_cache' not in config_dict and 'strategy' in config_dict:
            strategy = config_dict['strategy']
            if strategy in ['scored_pq', 'scored_kv_direct']:
                config_dict['kv_cache'] = strategy
            elif strategy in ['triple_pq', 'triple_tq', 'polarquant', 'turboangle']:
                config_dict['kv_cache'] = 'triple_pq' if 'triple' in strategy else 'triple_pq'
            else:
                config_dict['kv_cache'] = 'standard'

        # Context-length adaptive density mode
        if context_len < 8192:
            # Short context: prioritize quality
            if config_dict.get('strategy') in ['scored_pq', 'scored_kv_direct']:
                config_dict['density_mode'] = 'recall_first'
                rationale.append(f"Short context ({context_len} < 8K): using recall_first density mode for quality")
        elif context_len <= 32768:
            # Medium context: balanced
            if config_dict.get('strategy') in ['scored_pq', 'scored_kv_direct']:
                config_dict['density_mode'] = 'balanced'
                rationale.append(f"Medium context (8K-32K): using balanced density mode")
        else:
            # Long context: prioritize memory
            if config_dict.get('strategy') in ['scored_pq', 'scored_kv_direct']:
                config_dict['density_mode'] = 'ultra_long'
                rationale.append(f"Long context ({context_len} > 32K): using ultra_long density mode for memory")

        # Target-specific adaptations
        if target == 'memory':
            # Prefer scored strategies for memory optimization
            if config_dict.get('kv_cache') == 'standard':
                config_dict['kv_cache'] = 'scored_pq'
                config_dict['strategy'] = 'scored_pq'
                config_dict['flat_quant'] = 'q8_0'
                rationale.append("Memory target: switched to scored_pq with Q8 quantization")

        elif target == 'speed':
            # Prefer triple strategies (no eviction overhead)
            if config_dict.get('strategy') in ['scored_pq', 'scored_kv_direct']:
                config_dict['kv_cache'] = 'triple_pq'
                config_dict['strategy'] = 'polarquant'
                config_dict['kv_warm_bits'] = 4
                rationale.append("Speed target: switched to triple_pq for lower overhead")

        elif target == 'quality':
            # High precision quantization or no compression
            if config_dict.get('strategy') == 'polarquant' and config_dict.get('kv_warm_bits', 4) < 4:
                config_dict['kv_warm_bits'] = 4
                rationale.append("Quality target: increased quantization bits to 4")
            elif config_dict.get('strategy') == 'turboangle':
                # Increase codebook size
                if config_dict.get('n_k', 0) < 256:
                    config_dict['n_k'] = 256
                    config_dict['n_v'] = 128
                    rationale.append("Quality target: increased TurboAngle codebook size")

        # Create BenchmarkConfig
        config = BenchmarkConfig(
            kv_cache=config_dict.get('kv_cache', 'standard'),
            kv_warm_bits=config_dict.get('kv_warm_bits'),
            strategy=config_dict.get('strategy'),
            n_k=config_dict.get('n_k'),
            n_v=config_dict.get('n_v'),
            density_mode=config_dict.get('density_mode'),
            context_length=context_len,
        )

        return config, rationale

    def generate_config(
        self,
        model_path: str,
        context_len: int = 4096,
        target: str = 'balanced'
    ) -> ConfigRecommendation:
        """
        Generate recommended configuration for a model.

        Parameters
        ----------
        model_path : str
            Path to MLX model
        context_len : int
            Target context length (default: 4096)
        target : str
            Optimization target: 'balanced', 'speed', 'memory', 'quality'

        Returns
        -------
        recommendation : ConfigRecommendation
            Configuration recommendation with confidence and rationale
        """
        # Step 1: Analyze model
        fingerprint = self.analyze_model(model_path)
        rationale = [f"Model: {fingerprint.model_id}",
                     f"Architecture: {fingerprint.architecture_type}",
                     f"Layers: {fingerprint.num_layers}",
                     f"KV/Q ratio: {fingerprint.kv_q_ratio:.2f}"]

        # Step 2: Query similar models
        similar_models = self.query_similar_models(fingerprint, top_k=3)

        if similar_models:
            # Use best matching model's optimal config
            best_match_id, similarity, best_card = similar_models[0]
            rationale.append(f"Most similar model: {best_match_id} (similarity: {similarity:.2f})")

            # Extract optimal config from model card
            optimal = best_card.get('optimal', {})
            if not optimal and 'meta_harness_results' in best_card:
                # Use meta_harness_results if available
                meta_results = best_card['meta_harness_results']
                # Get config from closest context length
                closest_ctx = min(meta_results.keys(), key=lambda k: abs(int(k.replace('k', '000')) - context_len))
                optimal = meta_results[closest_ctx].get('optimal_config', {})

            # Adapt for context and target
            config, adapt_rationale = self.adapt_for_context(context_len, target, optimal)
            rationale.extend(adapt_rationale)

            confidence = similarity * 0.9  # High confidence from history
            source = 'history'

        else:
            # Cold start: use architecture-based rules
            rationale.append("No similar models found - using architecture-based rules")

            if fingerprint.is_mla:
                # MLA: KV cache is NOT the bottleneck
                config_dict = {
                    'kv_cache': 'triple_pq',
                    'strategy': 'turboangle',
                    'n_k': 256,
                    'n_v': 128,
                }
                rationale.append("MLA architecture: using TurboAngle (KV cache not bottleneck)")

            elif fingerprint.architecture_type == 'hybrid_ssm_attention':
                # Hybrid: KV cache is small, use high-precision
                config_dict = {
                    'kv_cache': 'triple_pq',
                    'strategy': 'turboangle',
                    'n_k': 256,
                    'n_v': 128,
                }
                rationale.append(f"Hybrid architecture ({fingerprint.num_attention_layers}/{fingerprint.num_layers} attention layers): using TurboAngle")

            else:
                # Pure Transformer: KV cache optimization is effective
                if target == 'memory' or context_len > 16384:
                    config_dict = {
                        'kv_cache': 'scored_pq',
                        'strategy': 'scored_pq',
                        'flat_quant': 'q8_0',
                        'density_mode': 'balanced',
                    }
                    rationale.append("Pure Transformer + long context: using scored_pq for extreme compression")
                else:
                    config_dict = {
                        'kv_cache': 'triple_pq',
                        'strategy': 'polarquant',
                        'kv_warm_bits': 4,
                    }
                    rationale.append("Pure Transformer: using triple_pq + PolarQuant")

            # Adapt for context and target
            config, adapt_rationale = self.adapt_for_context(context_len, target, config_dict)
            rationale.extend(adapt_rationale)

            confidence = 0.6  # Lower confidence for cold start
            source = 'architecture_rule'

        return ConfigRecommendation(
            config=config,
            confidence=confidence,
            rationale=rationale,
            source=source,
        )

    def _save_fingerprints(self):
        """Save fingerprints database to disk."""
        data = {k: v.to_dict() for k, v in self.fingerprints.items()}
        with open(self.fingerprints_file, 'w') as f:
            json.dump(data, f, indent=2)


def main():
    """CLI for adaptive config generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive Configuration Generator")
    parser.add_argument('model_path', type=str, help='Path to MLX model')
    parser.add_argument('--context-len', type=int, default=4096,
                        help='Target context length (default: 4096)')
    parser.add_argument('--target', type=str, default='balanced',
                        choices=['speed', 'memory', 'quality', 'balanced'],
                        help='Optimization target (default: balanced)')

    args = parser.parse_args()

    # Generate recommendation
    generator = AdaptiveConfigGenerator()
    recommendation = generator.generate_config(
        args.model_path,
        context_len=args.context_len,
        target=args.target
    )

    # Print recommendation
    print(f"\n{'='*80}")
    print("CONFIGURATION RECOMMENDATION")
    print(f"{'='*80}\n")

    print(f"Model: {args.model_path}")
    print(f"Context: {args.context_len} tokens")
    print(f"Target: {args.target}")
    print(f"Confidence: {recommendation.confidence:.1%}")
    print(f"Source: {recommendation.source}\n")

    print(f"{'='*80}")
    print("RECOMMENDED CONFIG")
    print(f"{'='*80}\n")
    print(recommendation.config)

    print(f"\n{'='*80}")
    print("RATIONALE")
    print(f"{'='*80}\n")
    for reason in recommendation.rationale:
        print(f"  • {reason}")
    print()


if __name__ == '__main__':
    main()
