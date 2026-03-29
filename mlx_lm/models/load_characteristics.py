"""
Load Characteristics Analyzer - Adaptive Recent Window Based on Workload

Analyzes input token sequences to determine redundancy and recommend
optimal recent_window_size for DoubleLayerKVCache.

Key Insight:
    Different workloads have different context redundancy:
    - Long document summarization: High redundancy (70-80%) → Small window (128)
    - Coding tasks: Medium redundancy (40-50%) → Medium window (256-384)
    - Agent execution: Low redundancy (20-30%) → Large window (384-512)
    - QA dialogue: Low redundancy (10-20%) → Large window (512+)

Design:
    1. N-gram overlap detection (simple but effective)
    2. Code structure detection (identify coding tasks)
    3. Redundancy scoring (0.0-1.0)
    4. Window size recommendation (128/256/384/512)
"""

from typing import List, Tuple, Optional
import numpy as np
from collections import Counter


class LoadCharacteristicsAnalyzer:
    """
    Analyze workload characteristics and recommend adaptive recent window.

    Parameters
    ----------
    sample_size : int
        Number of tokens to analyze (default: 500)
        Trade-off: larger = more accurate, but slower
    ngram_size : int
        N-gram size for overlap detection (default: 4)

    Example
    -------
    >>> analyzer = LoadCharacteristicsAnalyzer()
    >>> tokens = tokenizer.encode("Long document with repeated phrases...")
    >>> redundancy = analyzer.analyze_redundancy(tokens)
    >>> window_size = analyzer.recommend_window_size(redundancy)
    >>> print(f"Redundancy: {redundancy:.2%}, Window: {window_size}")
    """

    def __init__(self, sample_size: int = 500, ngram_size: int = 4):
        self.sample_size = sample_size
        self.ngram_size = ngram_size

        # Code structure indicators (common token IDs)
        # These are heuristics - adjust based on tokenizer
        self.code_indicators = {
            'braces': ['{', '}', '[', ']', '(', ')'],
            'keywords': ['def', 'class', 'function', 'if', 'else', 'for', 'while'],
            'operators': ['=', '==', '!=', '<=', '>=', '+', '-', '*', '/']
        }

    def analyze_redundancy(self, tokens: List[int]) -> float:
        """
        Analyze token sequence redundancy.

        Parameters
        ----------
        tokens : List[int]
            Token IDs to analyze

        Returns
        -------
        redundancy : float
            Redundancy score (0.0-1.0)
            - 0.0: No redundancy (unique tokens)
            - 1.0: Maximum redundancy (all repeated)
        """
        # Sample tokens for efficiency
        if len(tokens) > self.sample_size:
            # Sample from beginning (most representative of workload)
            sampled_tokens = tokens[:self.sample_size]
        else:
            sampled_tokens = tokens

        if len(sampled_tokens) < self.ngram_size * 2:
            # Too short to analyze
            return 0.0

        # 1. N-gram overlap detection (local repetition)
        ngram_overlap = self._ngram_overlap(sampled_tokens)

        # 2. Token repetition rate (global repetition)
        token_repetition = self._token_repetition_rate(sampled_tokens)

        # 3. Information entropy (diversity measure)
        entropy_score = self._information_entropy(sampled_tokens)

        # 4. Sliding window similarity (semantic clustering)
        window_similarity = self._sliding_window_similarity(sampled_tokens)

        # 5. Code structure detection
        is_code, code_score = self._detect_code_structure(sampled_tokens)

        # Combined redundancy score
        if is_code:
            # Code: structure matters, but local dependencies are critical
            # Weight: local patterns (ngram) > global repetition > entropy
            redundancy = (
                0.35 * ngram_overlap +
                0.20 * token_repetition +
                0.15 * (1.0 - entropy_score) +  # Low entropy = high redundancy
                0.10 * window_similarity +
                0.05 * code_score
            )
        else:
            # Natural language: semantic clustering matters most
            # Weight: window similarity > ngram > token repetition > entropy
            redundancy = (
                0.40 * window_similarity +
                0.30 * ngram_overlap +
                0.20 * token_repetition +
                0.10 * (1.0 - entropy_score)
            )

        return min(1.0, max(0.0, redundancy))

    def _ngram_overlap(self, tokens: List[int]) -> float:
        """
        Calculate N-gram overlap rate.

        Returns
        -------
        overlap : float
            Ratio of repeated N-grams (0.0-1.0)
        """
        ngrams = []
        for i in range(len(tokens) - self.ngram_size + 1):
            ngram = tuple(tokens[i:i + self.ngram_size])
            ngrams.append(ngram)

        if len(ngrams) == 0:
            return 0.0

        # Count unique vs total
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)

        # Overlap rate = (total - unique) / total
        overlap = 1.0 - (unique_ngrams / total_ngrams)

        return overlap

    def _token_repetition_rate(self, tokens: List[int]) -> float:
        """
        Calculate token-level repetition rate.

        Returns
        -------
        repetition : float
            Ratio of repeated tokens (0.0-1.0)
        """
        token_counts = Counter(tokens)
        unique_tokens = len(token_counts)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return 0.0

        # Repetition rate = (total - unique) / total
        repetition = 1.0 - (unique_tokens / total_tokens)

        return repetition

    def _information_entropy(self, tokens: List[int]) -> float:
        """
        Calculate normalized information entropy.

        Returns
        -------
        entropy : float
            Normalized entropy (0.0-1.0)
            - 0.0: All tokens identical (no diversity)
            - 1.0: Uniform distribution (maximum diversity)
        """
        if len(tokens) == 0:
            return 0.0

        # Count token frequencies
        token_counts = Counter(tokens)
        total_tokens = len(tokens)

        # Calculate Shannon entropy
        entropy = 0.0
        for count in token_counts.values():
            probability = count / total_tokens
            if probability > 0:
                entropy -= probability * np.log2(probability)

        # Normalize by maximum possible entropy (log2 of unique tokens)
        max_entropy = np.log2(len(token_counts)) if len(token_counts) > 1 else 1.0

        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return min(1.0, normalized_entropy)

    def _sliding_window_similarity(self, tokens: List[int], window_size: int = 50) -> float:
        """
        Calculate sliding window similarity (detect repeated segments).

        Parameters
        ----------
        tokens : List[int]
            Token sequence
        window_size : int
            Window size for comparison (default: 50)

        Returns
        -------
        similarity : float
            Average similarity between windows (0.0-1.0)
            - 0.0: No repeated segments
            - 1.0: All segments identical
        """
        if len(tokens) < window_size * 2:
            return 0.0

        # Create sliding windows
        windows = []
        for i in range(0, len(tokens) - window_size + 1, window_size // 2):
            window = tuple(tokens[i:i + window_size])
            windows.append(window)

        if len(windows) < 2:
            return 0.0

        # Calculate pairwise Jaccard similarity
        similarities = []
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                set_i = set(windows[i])
                set_j = set(windows[j])
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                jaccard = intersection / union if union > 0 else 0.0
                similarities.append(jaccard)

        # Return average similarity
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        return avg_similarity

    def _detect_code_structure(self, tokens: List[int]) -> Tuple[bool, float]:
        """
        Detect if token sequence is code.

        Returns
        -------
        is_code : bool
            Whether sequence is likely code
        code_score : float
            Code structure score (0.0-1.0)
        """
        # Heuristic 1: High unique token ratio (variable names, function names)
        unique_ratio = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0.0

        # Heuristic 2: Low N-gram overlap (code is less repetitive than natural language)
        ngram_overlap = self._ngram_overlap(tokens)

        # Heuristic 3: Token distribution pattern
        # Code typically has a few high-frequency tokens (keywords, operators)
        # and many low-frequency tokens (identifiers)
        token_counts = Counter(tokens)
        frequencies = list(token_counts.values())
        if len(frequencies) > 0:
            # Calculate coefficient of variation
            mean_freq = np.mean(frequencies)
            std_freq = np.std(frequencies)
            cv = std_freq / mean_freq if mean_freq > 0 else 0.0
            high_variance = cv > 2.0  # Code has high variance
        else:
            high_variance = False

        # Combined heuristic
        is_code = (unique_ratio > 0.65) and (ngram_overlap < 0.3) and high_variance
        code_score = unique_ratio if is_code else 0.0

        return is_code, code_score

    def recommend_window_size(
        self,
        redundancy: float,
        workload_hint: Optional[str] = None
    ) -> int:
        """
        Recommend recent window size based on redundancy.

        Parameters
        ----------
        redundancy : float
            Redundancy score (0.0-1.0)
        workload_hint : str, optional
            Explicit workload type hint (used as fallback or validation)

        Returns
        -------
        window_size : int
            Recommended recent window size (128/256/384/512)

        Strategy
        --------
        Primary: Use detected redundancy (data-driven)
        Fallback: Use workload hint if redundancy is inconclusive
        """
        # Primary: Redundancy-based recommendation (data-driven)
        # Note: Real-world redundancy typically ranges 10-40% for text workloads
        # Adjusted thresholds based on empirical observations
        if redundancy > 0.35:
            # High redundancy (>35%): aggressive compression
            recommended_by_data = 128
        elif redundancy > 0.25:
            # Medium-high redundancy (25-35%)
            recommended_by_data = 256
        elif redundancy > 0.18:
            # Medium-low redundancy (18-25%)
            recommended_by_data = 384
        else:
            # Low redundancy (<18%): conservative compression
            recommended_by_data = 512

        # Fallback: If hint provided, use it as validation
        if workload_hint:
            workload_map = {
                "summarization": 128,  # High redundancy expected
                "coding": 256,         # Medium redundancy, structured
                "agent": 512,          # Low redundancy, critical state
                "qa": 512,             # Low redundancy, precise context
                "chat": 384            # Medium-low redundancy
            }
            recommended_by_hint = workload_map.get(workload_hint, recommended_by_data)

            # Combine: if they differ significantly, use more conservative (larger) window
            if abs(recommended_by_data - recommended_by_hint) > 128:
                # Significant disagreement - be conservative
                return max(recommended_by_data, recommended_by_hint)
            else:
                # Agreement or minor difference - use data-driven
                return recommended_by_data
        else:
            # No hint - pure data-driven
            return recommended_by_data

    def analyze_and_recommend(
        self,
        tokens: List[int],
        workload_hint: Optional[str] = None
    ) -> Tuple[float, int]:
        """
        One-shot analysis and recommendation.

        Parameters
        ----------
        tokens : List[int]
            Token IDs to analyze
        workload_hint : str, optional
            Explicit workload type hint

        Returns
        -------
        redundancy : float
            Redundancy score
        window_size : int
            Recommended window size
        """
        redundancy = self.analyze_redundancy(tokens)
        window_size = self.recommend_window_size(redundancy, workload_hint)

        return redundancy, window_size
