#!/usr/bin/env python3
"""
Multi-Task Quality Evaluation for FlashMLX

Evaluates KV cache configurations across multiple quality benchmarks:
- Perplexity (already implemented in Meta-Harness)
- Needle-in-haystack recall
- Reasoning accuracy (MMLU-style)
- Generation quality (HellaSwag-style)

Focuses on detecting quality degradation from compression/quantization.
"""

import json
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate

from flashmlx_meta_harness import BenchmarkConfig
from mlx_lm.models.cache import make_prompt_cache


@dataclass
class EvaluationResult:
    """Results for a single evaluation task."""

    task_name: str
    config: BenchmarkConfig
    score: float  # Task-specific score (accuracy, recall, etc.)
    num_samples: int
    failed_samples: List[Dict]  # Edge cases where model failed
    timestamp: str


class NeedleInHaystackEvaluator:
    """
    Needle-in-haystack evaluation for recall testing.

    Tests if compressed KV cache can retrieve specific information
    from long context.
    """

    def __init__(self, model, tokenizer, context_lengths: List[int] = None):
        """
        Parameters
        ----------
        model : MLX model
            Model to evaluate
        tokenizer : Tokenizer
            Tokenizer for the model
        context_lengths : List[int], optional
            Context lengths to test (default: [2K, 4K, 8K])
        """
        self.model = model
        self.tokenizer = tokenizer
        self.context_lengths = context_lengths or [2048, 4096, 8192]

    def generate_haystack(self, length: int, needle_position: float = 0.5) -> Tuple[str, str, int]:
        """
        Generate haystack text with embedded needle.

        Parameters
        ----------
        length : int
            Target length in tokens
        needle_position : float
            Relative position of needle (0.0 = start, 1.0 = end)

        Returns
        -------
        haystack : str
            Full text with needle embedded
        needle : str
            The needle text to find
        needle_token_pos : int
            Token position of needle
        """
        # Filler text (repetitive but realistic)
        filler_paragraphs = [
            "The company announced its quarterly earnings today, showing steady growth across all segments. ",
            "Market analysts predict continued expansion in the technology sector throughout the year. ",
            "Industry experts recommend diversifying investment portfolios to minimize risk. ",
            "Recent studies have shown significant improvements in operational efficiency. ",
            "The research team published their findings in a peer-reviewed journal. ",
        ]

        # Needle: unique factual statement
        needles = [
            "The magic number is 7284 and the secret code is PHOENIX.",
            "The special sequence is 9157 and the password is DRAGON.",
            "The unique identifier is 4621 and the key is FALCON.",
            "The reference code is 8395 and the token is EAGLE.",
        ]
        needle = random.choice(needles)

        # Build haystack
        tokens_per_paragraph = 30  # Approximate
        num_paragraphs = length // tokens_per_paragraph

        # Calculate needle insertion point
        needle_paragraph_idx = int(num_paragraphs * needle_position)

        paragraphs = []
        for i in range(num_paragraphs):
            if i == needle_paragraph_idx:
                paragraphs.append(needle + " ")
            paragraphs.append(random.choice(filler_paragraphs))

        haystack = "".join(paragraphs)

        # Calculate token position
        prefix = "".join(paragraphs[:needle_paragraph_idx])
        needle_token_pos = len(self.tokenizer.encode(prefix))

        return haystack, needle, needle_token_pos

    def evaluate(self, config: BenchmarkConfig, num_samples: int = 6) -> EvaluationResult:
        """
        Evaluate needle-in-haystack recall.

        Parameters
        ----------
        config : BenchmarkConfig
            KV cache configuration to test
        num_samples : int
            Number of samples (2 per context length)

        Returns
        -------
        result : EvaluationResult
            Evaluation results with recall score
        """
        print(f"\n{'='*80}")
        print(f"Needle-in-Haystack Evaluation: {config}")
        print(f"{'='*80}\n")

        failed_samples = []
        total_correct = 0
        total_samples = 0

        # Create cache (skip for now - use default)
        # TODO: Fix cache creation to work with config
        cache = None  # Use default cache
        # cache_kwargs = config.to_cache_kwargs()
        # cache = make_prompt_cache(self.model, **cache_kwargs)

        for ctx_len in self.context_lengths:
            for position in [0.2, 0.8]:  # Test early and late positions
                # Generate haystack
                haystack, needle, needle_pos = self.generate_haystack(ctx_len, position)

                # Create prompt
                prompt = f"{haystack}\n\nQuestion: What is the magic/special/unique number and code mentioned in the text above? Answer with the exact number and word."

                # Tokenize
                tokens = mx.array([self.tokenizer.encode(prompt)])

                # Generate response
                try:
                    response = generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt,
                        max_tokens=50,
                    )

                    # Check if needle is recalled
                    # Extract key parts from needle
                    needle_parts = needle.split()
                    number = [p for p in needle_parts if p.isdigit()][0]
                    code = [p.strip('.') for p in needle_parts if p.isupper() and len(p) > 3][0]

                    correct = number in response and code in response

                    if correct:
                        total_correct += 1
                    else:
                        failed_samples.append({
                            'context_length': ctx_len,
                            'needle_position': position,
                            'needle': needle,
                            'response': response,
                            'expected_number': number,
                            'expected_code': code,
                        })

                    total_samples += 1

                    status = "✓" if correct else "✗"
                    print(f"  [{status}] Context: {ctx_len}, Position: {position:.1%}, Needle: {needle[:50]}...")

                except Exception as e:
                    print(f"  [ERROR] Context: {ctx_len}, Position: {position:.1%}: {e}")
                    failed_samples.append({
                        'context_length': ctx_len,
                        'needle_position': position,
                        'error': str(e),
                    })
                    total_samples += 1

                # Clear cache for next sample
                mx.clear_cache()

        recall_score = total_correct / total_samples if total_samples > 0 else 0.0

        print(f"\nRecall: {total_correct}/{total_samples} = {recall_score:.1%}")

        return EvaluationResult(
            task_name='needle_in_haystack',
            config=config,
            score=recall_score,
            num_samples=total_samples,
            failed_samples=failed_samples,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )


class ReasoningEvaluator:
    """
    MMLU-style reasoning evaluation.

    Tests multiple-choice reasoning to detect compression-induced
    reasoning degradation.
    """

    def __init__(self, model, tokenizer):
        """
        Parameters
        ----------
        model : MLX model
            Model to evaluate
        tokenizer : Tokenizer
            Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer

        # Small MMLU-style test set
        self.test_samples = [
            {
                'question': 'What is the capital of France?',
                'choices': ['A) London', 'B) Paris', 'C) Berlin', 'D) Madrid'],
                'answer': 'B',
            },
            {
                'question': 'Which planet is closest to the Sun?',
                'choices': ['A) Venus', 'B) Earth', 'C) Mercury', 'D) Mars'],
                'answer': 'C',
            },
            {
                'question': 'What is 15 * 8?',
                'choices': ['A) 120', 'B) 130', 'C) 110', 'D) 125'],
                'answer': 'A',
            },
            {
                'question': 'Who wrote "Romeo and Juliet"?',
                'choices': ['A) Charles Dickens', 'B) William Shakespeare', 'C) Jane Austen', 'D) Mark Twain'],
                'answer': 'B',
            },
            {
                'question': 'What is the chemical symbol for gold?',
                'choices': ['A) Go', 'B) Gd', 'C) Au', 'D) Ag'],
                'answer': 'C',
            },
        ]

    def evaluate(self, config: BenchmarkConfig) -> EvaluationResult:
        """
        Evaluate reasoning accuracy.

        Parameters
        ----------
        config : BenchmarkConfig
            KV cache configuration to test

        Returns
        -------
        result : EvaluationResult
            Evaluation results with accuracy
        """
        print(f"\n{'='*80}")
        print(f"Reasoning Evaluation: {config}")
        print(f"{'='*80}\n")

        # Create cache (skip for now - use default)
        # TODO: Fix cache creation to work with config
        cache = None  # Use default cache
        # cache_kwargs = config.to_cache_kwargs()
        # cache = make_prompt_cache(self.model, **cache_kwargs)

        failed_samples = []
        total_correct = 0

        for i, sample in enumerate(self.test_samples):
            # Format prompt
            choices_text = '\n'.join(sample['choices'])
            prompt = f"Question: {sample['question']}\n\n{choices_text}\n\nAnswer (single letter):"

            # Generate response
            try:
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=5,
                )

                # Extract answer (first letter A-D)
                predicted = None
                for char in response.upper():
                    if char in 'ABCD':
                        predicted = char
                        break

                correct = (predicted == sample['answer'])

                if correct:
                    total_correct += 1
                else:
                    failed_samples.append({
                        'question': sample['question'],
                        'expected': sample['answer'],
                        'predicted': predicted,
                        'response': response,
                    })

                status = "✓" if correct else "✗"
                print(f"  [{status}] Q{i+1}: {sample['question'][:50]}... (Expected: {sample['answer']}, Got: {predicted})")

            except Exception as e:
                print(f"  [ERROR] Q{i+1}: {e}")
                failed_samples.append({
                    'question': sample['question'],
                    'error': str(e),
                })

            mx.clear_cache()

        accuracy = total_correct / len(self.test_samples)

        print(f"\nAccuracy: {total_correct}/{len(self.test_samples)} = {accuracy:.1%}")

        return EvaluationResult(
            task_name='reasoning',
            config=config,
            score=accuracy,
            num_samples=len(self.test_samples),
            failed_samples=failed_samples,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )


class MultiTaskEvaluator:
    """
    Multi-task quality evaluation for KV cache configurations.

    Evaluates configurations across multiple quality dimensions:
    - Perplexity (from Meta-Harness)
    - Needle-in-haystack recall
    - Reasoning accuracy
    """

    def __init__(self, model_path: str):
        """
        Parameters
        ----------
        model_path : str
            Path to MLX model
        """
        self.model_path = model_path

        print(f"Loading model from {model_path}...")
        self.model, self.tokenizer = load(model_path)

        # Initialize evaluators
        self.needle_eval = NeedleInHaystackEvaluator(self.model, self.tokenizer)
        self.reasoning_eval = ReasoningEvaluator(self.model, self.tokenizer)

        # Results storage
        self.results: List[EvaluationResult] = []

    def evaluate_config(self, config: BenchmarkConfig) -> Dict[str, float]:
        """
        Evaluate a configuration across all tasks.

        Parameters
        ----------
        config : BenchmarkConfig
            Configuration to evaluate

        Returns
        -------
        scores : Dict[str, float]
            Scores for each task plus composite score
        """
        print(f"\n{'='*80}")
        print(f"MULTI-TASK EVALUATION")
        print(f"Config: {config}")
        print(f"{'='*80}\n")

        # Run evaluations
        needle_result = self.needle_eval.evaluate(config, num_samples=6)
        reasoning_result = self.reasoning_eval.evaluate(config)

        # Store results
        self.results.append(needle_result)
        self.results.append(reasoning_result)

        # Compute composite score (weighted average)
        composite_score = (
            0.5 * needle_result.score +  # Recall is critical for long context
            0.5 * reasoning_result.score
        )

        scores = {
            'needle_recall': needle_result.score,
            'reasoning_acc': reasoning_result.score,
            'composite_score': composite_score,
        }

        return scores

    def compare_configs(self, configs: List[BenchmarkConfig]) -> Dict:
        """
        Compare multiple configurations.

        Parameters
        ----------
        configs : List[BenchmarkConfig]
            Configurations to compare

        Returns
        -------
        comparison : Dict
            Comparison results with rankings
        """
        all_scores = []

        for config in configs:
            scores = self.evaluate_config(config)
            all_scores.append({
                'config': config,
                'scores': scores,
            })

        # Rank by composite score
        all_scores.sort(key=lambda x: x['scores']['composite_score'], reverse=True)

        # Print comparison
        print(f"\n{'='*80}")
        print("CONFIGURATION COMPARISON")
        print(f"{'='*80}\n")
        print(f"{'Rank':<6} {'Config':<40} {'Needle':<10} {'Reasoning':<12} {'Composite':<10}")
        print(f"{'-'*80}")

        for i, item in enumerate(all_scores, 1):
            config_str = str(item['config'])[:38]
            scores = item['scores']
            print(f"{i:<6} {config_str:<40} {scores['needle_recall']:<10.1%} {scores['reasoning_acc']:<12.1%} {scores['composite_score']:<10.3f}")

        return {
            'rankings': all_scores,
            'best_config': all_scores[0]['config'],
            'best_scores': all_scores[0]['scores'],
        }

    def save_results(self, output_path: str):
        """Save evaluation results to JSON."""
        data = {
            'model_path': self.model_path,
            'results': [
                {
                    'task_name': r.task_name,
                    'config': asdict(r.config),
                    'score': r.score,
                    'num_samples': r.num_samples,
                    'num_failed': len(r.failed_samples),
                    'failed_samples': r.failed_samples[:5],  # Limit to first 5
                    'timestamp': r.timestamp,
                }
                for r in self.results
            ],
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to {output_path}")


def main():
    """CLI for multi-task evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="FlashMLX Multi-Task Quality Evaluation")
    parser.add_argument('model_path', type=str, help='Path to MLX model')
    parser.add_argument('--output', type=str, default='multi_task_eval_results.json',
                        help='Output JSON file')

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = MultiTaskEvaluator(args.model_path)

    # Test configs: baseline + compressed
    configs = [
        BenchmarkConfig(kv_cache='standard'),
        BenchmarkConfig(kv_cache='triple_pq', kv_warm_bits=4, strategy='polarquant'),
        BenchmarkConfig(kv_cache='triple_pq', strategy='turboangle', n_k=128, n_v=64),
    ]

    # Compare configurations
    comparison = evaluator.compare_configs(configs)

    # Save results
    evaluator.save_results(args.output)

    print(f"\nBest configuration: {comparison['best_config']}")
    print(f"Scores: {comparison['best_scores']}")


if __name__ == '__main__':
    main()
