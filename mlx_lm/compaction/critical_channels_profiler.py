"""
Critical Channels Profiler for SSM State Compression

Identifies which channels in SSM state are critical for maintaining:
- Chinese token generation
- <think> tag control
- Format control
- Overall output quality

Author: FlashMLX Research
Date: 2026-03-21
Task: #53 - SSM State Compression (Three-Stage Conservative Approach)
"""

import mlx.core as mx
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# Test prompt with critical features
TEST_PROMPT = """请用中文回答以下问题：

<think>
首先，我需要理解机器学习的核心概念...
</think>

问题：什么是机器学习？

回答：
"""


class CriticalChannelsProfiler:
    """
    Profile SSM state to identify critical channels

    Critical channels are those that significantly affect:
    - Chinese token probabilities
    - <think> tag control
    - Format token control
    - Overall output quality
    """

    def __init__(
        self,
        model,
        tokenizer,
        test_prompt: str = TEST_PROMPT,
        perturbation_strength: float = 0.1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.test_prompt = test_prompt
        self.perturbation_strength = perturbation_strength

        # Special tokens to monitor
        self.chinese_token_ids = self._get_chinese_token_ids()
        self.think_tag_ids = self._get_special_token_ids(['<think>', '</think>'])
        self.format_token_ids = self._get_format_token_ids()

    def _get_chinese_token_ids(self) -> List[int]:
        """Get token IDs for common Chinese characters"""
        chinese_samples = ['的', '是', '在', '我', '有', '和', '人', '这', '中', '了']
        token_ids = []
        for char in chinese_samples:
            try:
                ids = self.tokenizer.encode(char)
                token_ids.extend(ids)
            except:
                pass
        return list(set(token_ids))

    def _get_special_token_ids(self, tokens: List[str]) -> List[int]:
        """Get token IDs for special tokens"""
        token_ids = []
        for token in tokens:
            try:
                ids = self.tokenizer.encode(token)
                token_ids.extend(ids)
            except:
                pass
        return list(set(token_ids))

    def _get_format_token_ids(self) -> List[int]:
        """Get token IDs for format control tokens"""
        format_samples = ['\n', '：', '、', '1.', '2.', '- ', '* ']
        token_ids = []
        for token in format_samples:
            try:
                ids = self.tokenizer.encode(token)
                token_ids.extend(ids)
            except:
                pass
        return list(set(token_ids))

    def capture_ssm_state_at_layer(
        self,
        layer_idx: int,
        prompt: Optional[str] = None
    ) -> mx.array:
        """
        Capture SSM state at specific layer during generation

        Returns:
            state: (B, Hv, Dv, Dk) - SSM state at layer_idx
        """
        if prompt is None:
            prompt = self.test_prompt

        # Import cache
        from mlx_lm.models import cache as cache_module

        # Create cache using model's make_cache method
        cache = cache_module.make_prompt_cache(self.model)

        # Encode prompt
        tokens = mx.array(self.tokenizer.encode(prompt))

        # Step 1: Prefill (process entire prompt)
        # This doesn't fill cache, just processes the prompt
        logits = self.model(tokens[None], cache=cache)
        mx.eval(logits)

        # Step 2: Decode one token (this fills the cache!)
        # Sample next token
        next_token = mx.argmax(logits[0, -1, :], keepdims=True)

        # Decode step - this will populate cache
        logits = self.model(next_token[None], cache=cache)
        mx.eval(logits, cache)

        # Now cache should be populated
        layer_cache = cache[layer_idx]

        if layer_cache is None:
            raise ValueError(f"Layer {layer_idx} has no cache (not an SSM layer?)")

        # For Qwen3.5, each layer cache is an ArraysCache object
        # with a 'state' attribute containing [conv_state, ssm_state]
        if hasattr(layer_cache, 'state'):
            cache_state = layer_cache.state
            if isinstance(cache_state, list) and len(cache_state) == 2:
                ssm_state = cache_state[1]  # Get SSM state (index 1)
                if ssm_state is None:
                    raise ValueError(f"Layer {layer_idx} SSM state is None")
                return ssm_state
            else:
                raise ValueError(f"Layer {layer_idx} cache.state format unexpected: {type(cache_state)}")
        else:
            # Might be attention layer or KVCache
            raise ValueError(f"Layer {layer_idx} cache has no 'state' attribute (not an SSM layer?)")

        return ssm_state

    def perturb_channel(
        self,
        state: mx.array,
        channel_idx: int,
        strength: Optional[float] = None
    ) -> mx.array:
        """
        Perturb single channel in SSM state

        Args:
            state: (B, Hv, Dv, Dk) - SSM state
            channel_idx: Index of channel to perturb (0-127 for Dv dimension)
            strength: Perturbation strength (default: self.perturbation_strength)

        Returns:
            perturbed_state: SSM state with perturbed channel
        """
        if strength is None:
            strength = self.perturbation_strength

        # Clone state (MLX uses direct array operations)
        perturbed = mx.array(state)

        # Add noise to this channel across all heads and all Dk positions
        B, Hv, Dv, Dk = state.shape
        noise = mx.random.normal(shape=(B, Hv, 1, Dk)).astype(state.dtype)
        perturbed[:, :, channel_idx:channel_idx+1, :] += strength * noise

        return perturbed

    def generate_with_perturbed_state(
        self,
        state: mx.array,
        layer_idx: int,
        max_tokens: int = 20
    ) -> Tuple[mx.array, List[int]]:
        """
        Generate tokens using perturbed SSM state

        Returns:
            logits: (max_tokens, vocab_size) - Logits for generated tokens
            token_ids: List[int] - Generated token IDs
        """
        from mlx_lm.models import cache as cache_module

        # Create cache using model's make_cache method
        cache = cache_module.make_prompt_cache(self.model)

        # First, we need to populate the cache with a prefill+decode
        # This is the same as capture_ssm_state_at_layer
        tokens = mx.array(self.tokenizer.encode(self.test_prompt))

        # Prefill
        logits = self.model(tokens[None], cache=cache)
        mx.eval(logits)

        # Decode one token to populate cache
        next_token = mx.argmax(logits[0, -1, :], keepdims=True)
        logits = self.model(next_token[None], cache=cache)
        mx.eval(logits, cache)

        # Now inject perturbed state
        layer_cache = cache[layer_idx]
        if hasattr(layer_cache, 'state'):
            # Preserve conv_state, replace ssm_state
            current_state = layer_cache.state
            layer_cache.state = [current_state[0], state]  # [conv_state, new_ssm_state]
        else:
            raise ValueError(f"Cannot inject state into layer {layer_idx}")

        # Generate tokens and capture logits
        all_logits = []
        token_ids = []

        # Continue from where we left off
        current_tokens = next_token.reshape(1, 1)

        for _ in range(max_tokens):
            # Forward pass
            logits = self.model(current_tokens, cache=cache)
            mx.eval(logits)

            # Store logits
            all_logits.append(logits[0, -1, :])  # (vocab_size,)

            # Sample next token (greedy)
            next_token = mx.argmax(logits[0, -1, :], axis=-1)
            token_ids.append(int(next_token.item()))

            # Update current tokens
            current_tokens = next_token.reshape(1, 1)

        # Stack all logits
        all_logits = mx.stack(all_logits)  # (max_tokens, vocab_size)

        return all_logits, token_ids

    def measure_chinese_prob_change(
        self,
        original_logits: mx.array,
        perturbed_logits: mx.array
    ) -> float:
        """
        Measure change in Chinese token probabilities

        Returns:
            change: Average absolute change in Chinese token probabilities
        """
        # Get probabilities
        original_probs = mx.softmax(original_logits, axis=-1)
        perturbed_probs = mx.softmax(perturbed_logits, axis=-1)

        # Extract Chinese token probabilities
        chinese_original = original_probs[:, self.chinese_token_ids].sum(axis=-1)
        chinese_perturbed = perturbed_probs[:, self.chinese_token_ids].sum(axis=-1)

        # Compute average absolute change
        change = mx.abs(chinese_original - chinese_perturbed).mean()

        return float(change.item())

    def measure_think_tag_change(
        self,
        original_logits: mx.array,
        perturbed_logits: mx.array
    ) -> float:
        """
        Measure change in <think> tag probabilities
        """
        original_probs = mx.softmax(original_logits, axis=-1)
        perturbed_probs = mx.softmax(perturbed_logits, axis=-1)

        think_original = original_probs[:, self.think_tag_ids].sum(axis=-1)
        think_perturbed = perturbed_probs[:, self.think_tag_ids].sum(axis=-1)

        change = mx.abs(think_original - think_perturbed).mean()

        return float(change.item())

    def measure_format_change(
        self,
        original_logits: mx.array,
        perturbed_logits: mx.array
    ) -> float:
        """
        Measure change in format token probabilities
        """
        original_probs = mx.softmax(original_logits, axis=-1)
        perturbed_probs = mx.softmax(perturbed_logits, axis=-1)

        format_original = original_probs[:, self.format_token_ids].sum(axis=-1)
        format_perturbed = perturbed_probs[:, self.format_token_ids].sum(axis=-1)

        change = mx.abs(format_original - format_perturbed).mean()

        return float(change.item())

    def compute_kl_divergence(
        self,
        original_logits: mx.array,
        perturbed_logits: mx.array,
        top_k: int = 100
    ) -> float:
        """
        Compute KL divergence between top-k logits distributions
        """
        # Get top-k probabilities
        original_probs = mx.softmax(original_logits, axis=-1)
        perturbed_probs = mx.softmax(perturbed_logits, axis=-1)

        # Get top-k indices from original
        top_k_indices = mx.argpartition(-original_probs, kth=top_k, axis=-1)[:, :top_k]

        # Extract top-k probabilities
        original_top_k = mx.take_along_axis(original_probs, top_k_indices, axis=-1)
        perturbed_top_k = mx.take_along_axis(perturbed_probs, top_k_indices, axis=-1)

        # Compute KL divergence
        kl = (original_top_k * mx.log(original_top_k / (perturbed_top_k + 1e-10))).sum(axis=-1).mean()

        return float(kl.item())

    def measure_impact(
        self,
        original_logits: mx.array,
        perturbed_logits: mx.array
    ) -> Dict[str, float]:
        """
        Measure overall impact of channel perturbation

        Returns:
            {
                'chinese_prob_change': float,
                'think_tag_change': float,
                'format_change': float,
                'logits_kl': float,
                'overall_score': float
            }
        """
        chinese_change = self.measure_chinese_prob_change(original_logits, perturbed_logits)
        think_change = self.measure_think_tag_change(original_logits, perturbed_logits)
        format_change = self.measure_format_change(original_logits, perturbed_logits)
        kl = self.compute_kl_divergence(original_logits, perturbed_logits)

        # Weighted score (higher = more critical)
        overall_score = (
            0.3 * chinese_change +
            0.3 * think_change +
            0.2 * format_change +
            0.2 * kl
        )

        return {
            'chinese_prob_change': chinese_change,
            'think_tag_change': think_change,
            'format_change': format_change,
            'logits_kl': kl,
            'overall_score': overall_score
        }

    def profile_layer(
        self,
        layer_idx: int,
        num_channels: int = 128,
        critical_ratio: float = 0.05
    ) -> Tuple[List[int], List[float]]:
        """
        Profile single layer to identify critical channels

        Args:
            layer_idx: Layer index
            num_channels: Number of channels (Dv dimension)
            critical_ratio: Ratio of channels to mark as critical

        Returns:
            critical_channels: List of critical channel indices
            importance_scores: Importance score for each channel
        """
        print(f"\n{'='*60}")
        print(f"Profiling Layer {layer_idx}")
        print(f"{'='*60}")

        # 1. Capture baseline state
        print("Capturing baseline state...")
        baseline_state = self.capture_ssm_state_at_layer(layer_idx)

        # 2. Generate baseline logits
        print("Generating baseline logits...")
        baseline_logits, _ = self.generate_with_perturbed_state(
            baseline_state, layer_idx
        )

        importance_scores = []

        # 3. Profile each channel
        print(f"Profiling {num_channels} channels...")
        for channel_idx in range(num_channels):
            print(f"  Channel {channel_idx+1}/{num_channels}", end='\r')

            # Perturb this channel
            perturbed_state = self.perturb_channel(baseline_state, channel_idx)

            # Generate with perturbed state
            perturbed_logits, _ = self.generate_with_perturbed_state(
                perturbed_state, layer_idx
            )

            # Measure impact
            impact = self.measure_impact(baseline_logits, perturbed_logits)
            importance_scores.append(impact['overall_score'])

        print()  # New line after progress

        # 4. Sort and select critical channels
        sorted_indices = np.argsort(importance_scores)[::-1]
        num_critical = max(1, int(num_channels * critical_ratio))
        critical_channels = sorted_indices[:num_critical].tolist()

        print(f"\n✅ Identified {num_critical} critical channels:")
        print(f"   Indices: {critical_channels[:10]}...")
        print(f"   Top 5 scores: {[importance_scores[i] for i in critical_channels[:5]]}")

        return critical_channels, importance_scores

    def save_calibration(
        self,
        layer_idx: int,
        critical_channels: List[int],
        importance_scores: List[float],
        output_dir: str = ".solar/calibration"
    ):
        """
        Save calibration file for layer
        """
        os.makedirs(output_dir, exist_ok=True)

        calibration = {
            "layer": layer_idx,
            "rank": 32,  # Default rank for low-rank compression
            "critical_channels": critical_channels,
            "safe": True,
            "profiling_metadata": {
                "test_prompt": self.test_prompt,
                "num_channels": len(importance_scores),
                "critical_ratio": len(critical_channels) / len(importance_scores),
                "importance_scores": importance_scores,
                "perturbation_strength": self.perturbation_strength
            }
        }

        output_path = os.path.join(output_dir, f"layer_{layer_idx}_calibration.json")
        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=2)

        print(f"✅ Saved: {output_path}")

    def profile_all_layers(
        self,
        ssm_layer_indices: List[int],
        output_dir: str = ".solar/calibration"
    ):
        """
        Profile all SSM layers
        """
        print(f"\n{'='*60}")
        print(f"Profiling {len(ssm_layer_indices)} SSM Layers")
        print(f"{'='*60}")

        for i, layer_idx in enumerate(ssm_layer_indices):
            print(f"\n[{i+1}/{len(ssm_layer_indices)}] Layer {layer_idx}")

            # Profile layer
            critical_channels, importance_scores = self.profile_layer(layer_idx)

            # Save calibration
            self.save_calibration(
                layer_idx, critical_channels, importance_scores, output_dir
            )

        print(f"\n{'='*60}")
        print(f"✅ All layers profiled!")
        print(f"{'='*60}")
        print(f"\nCalibration files saved to: {output_dir}")
