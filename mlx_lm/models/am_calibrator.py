"""
Auto-calibration for AM (Attention Matching) compression.

Generates and caches AM calibration data when first needed.
Calibration cached at ~/.cache/flashmlx/calibrations/{model_key}_{ratio}x.pkl

Usage (automatic — called by make_optimized_cache):
    generate(model, tokenizer, prompt, kv_cache="scored_pq")
    # Auto-calibrates on first use, caches for future sessions

Usage (explicit):
    from mlx_lm.models.am_calibrator import auto_calibrate
    path = auto_calibrate(model, tokenizer, compression_ratio=2.0)
"""

import os
import time
import pickle
import numpy as np
import mlx.core as mx
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime


CALIBRATION_CACHE_DIR = Path.home() / ".cache" / "flashmlx" / "calibrations"

# Diverse calibration corpus (8 entries, ~200-400 tokens each)
# Covers: narrative, technical, Chinese, code, math, dialogue, lists, structured
_CALIBRATION_CORPUS = [
    # English narrative (fact-dense)
    "Dr. Sarah Chen founded the Quantum Dynamics Research Lab at Stanford University "
    "in 2019 with $5 million from the National Science Foundation. Her team aimed to "
    "develop room-temperature quantum computers. She recruited Dr. Robert Kim from MIT, "
    "Dr. Elena Rodriguez from Caltech, and Dr. Yuki Tanaka from Tokyo. In 2020, they "
    "built their first prototype. Early tests were disappointing - quantum coherence "
    "lasted only milliseconds. The breakthrough came on July 15, 2022 at 3:47 AM. "
    "The quantum processor achieved stable coherence at 294 Kelvin for 47 seconds. "
    "They ran 127 experiments with 89% success rate. Five independent teams replicated "
    "the results with 84% success rate. Dr. Chen received the Nobel Prize in 2024.",

    # Technical documentation
    "The HTTP/2 protocol uses binary framing instead of text-based HTTP/1.1. Key "
    "features include header compression via HPACK, multiplexed streams over a single "
    "TCP connection, server push, and stream prioritization. The frame format consists "
    "of a 9-byte header: length (24 bits), type (8 bits), flags (8 bits), reserved (1 "
    "bit), and stream identifier (31 bits). Common frame types are DATA (0x0), HEADERS "
    "(0x1), PRIORITY (0x2), RST_STREAM (0x3), SETTINGS (0x4), PUSH_PROMISE (0x5), "
    "PING (0x6), GOAWAY (0x7), WINDOW_UPDATE (0x8), and CONTINUATION (0x9). Flow "
    "control operates at both the stream and connection level using WINDOW_UPDATE frames.",

    # Chinese text
    "据新华社报道，2025年中国国内生产总值（GDP）增长率达到5.2%，超出市场预期。其中，"
    "第一产业增加值为89,755亿元，同比增长4.1%；第二产业增加值为482,066亿元，增长4.7%；"
    "第三产业增加值为688,219亿元，增长5.8%。全年社会消费品零售总额达到471,495亿元，"
    "比上年增长7.2%。全国固定资产投资额为503,036亿元，增长3.0%。进出口总值为41.76万亿元，"
    "增长0.2%。城镇新增就业1,244万人，全国城镇调查失业率平均值为5.2%。",

    # Code
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n"
    "    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n"
    "    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n\n"
    "class BinarySearchTree:\n    def __init__(self, value):\n        self.value = value\n"
    "        self.left = None\n        self.right = None\n\n"
    "    def insert(self, value):\n        if value < self.value:\n"
    "            if self.left is None:\n                self.left = BinarySearchTree(value)\n"
    "            else:\n                self.left.insert(value)\n"
    "        else:\n            if self.right is None:\n"
    "                self.right = BinarySearchTree(value)\n"
    "            else:\n                self.right.insert(value)\n",

    # Math/reasoning
    "Given a right triangle with sides a=3 and b=4, the hypotenuse c = sqrt(a^2 + b^2) "
    "= sqrt(9 + 16) = sqrt(25) = 5. The area is (1/2) * a * b = (1/2) * 3 * 4 = 6 "
    "square units. The perimeter is a + b + c = 3 + 4 + 5 = 12 units. For a circle "
    "inscribed in this triangle, the inradius r = (a + b - c) / 2 = (3 + 4 - 5) / 2 = 1. "
    "The circumradius R = c / 2 = 5 / 2 = 2.5. Euler's formula: d^2 = R(R - 2r) gives "
    "d^2 = 2.5(2.5 - 2) = 2.5 * 0.5 = 1.25, so d = sqrt(1.25) ≈ 1.118.",

    # Dialogue
    "User: Can you explain how transformers work in machine learning?\n"
    "Assistant: Transformers are a neural network architecture introduced in the paper "
    "'Attention Is All You Need' (Vaswani et al., 2017). The key innovation is the "
    "self-attention mechanism, which allows the model to weigh the importance of different "
    "parts of the input when producing each output element.\n"
    "User: How does self-attention actually compute those weights?\n"
    "Assistant: Self-attention uses three learned projections: Query (Q), Key (K), and "
    "Value (V). The attention score is computed as softmax(QK^T / sqrt(d_k)) * V, where "
    "d_k is the dimension of the key vectors. This allows each position to attend to all "
    "other positions in the sequence.",

    # List/enumeration (tests positional attention)
    "The top 20 programming languages by popularity in 2025 are: 1. Python, 2. JavaScript, "
    "3. TypeScript, 4. Java, 5. C++, 6. C#, 7. Go, 8. Rust, 9. PHP, 10. Swift, "
    "11. Kotlin, 12. Ruby, 13. Scala, 14. R, 15. Dart, 16. Lua, 17. Perl, 18. Haskell, "
    "19. Elixir, 20. Julia. Python leads with 28.1% share, JavaScript at 17.4%, "
    "TypeScript at 12.3%. The fastest growing languages are Rust (+45%), Zig (+38%), "
    "and Mojo (+32%). The most used frameworks are React (40.6%), Next.js (22.1%), "
    "Django (15.3%), Spring Boot (14.8%), and FastAPI (12.5%).",

    # JSON/structured data
    '{"company": "TechCorp", "founded": 2018, "revenue_millions": 450, "employees": 2300, '
    '"products": [{"name": "CloudSync", "category": "SaaS", "users": 150000, "price": 29.99}, '
    '{"name": "DataPipe", "category": "ETL", "users": 45000, "price": 99.99}, '
    '{"name": "SecureVault", "category": "Security", "users": 80000, "price": 49.99}], '
    '"offices": [{"city": "San Francisco", "country": "US", "headcount": 800}, '
    '{"city": "London", "country": "UK", "headcount": 500}, '
    '{"city": "Tokyo", "country": "JP", "headcount": 400}, '
    '{"city": "Berlin", "country": "DE", "headcount": 300}], '
    '"metrics": {"mrr": 12500000, "churn_rate": 0.023, "nps": 72, "cac": 185}}',
]

# Calibration questions (for QA-style prefill)
_CALIBRATION_QUESTIONS = [
    "When was the Quantum Dynamics Research Lab founded?",
    "Who did Dr. Chen recruit for her team?",
    "What was the breakthrough date and result?",
    "How does HTTP/2 differ from HTTP/1.1?",
    "What are the main HTTP/2 frame types?",
    "What was China's GDP growth rate?",
    "How does quicksort partition the array?",
    "What is the hypotenuse of a 3-4-5 triangle?",
    "How does self-attention compute weights?",
    "Which programming language is most popular?",
    "What is TechCorp's monthly recurring revenue?",
    "How many employees does TechCorp have?",
]

# Number of repeat-prefill passes per corpus entry
_NUM_REPEATS = 5


def _softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Numerically stable softmax (pure numpy, no scipy)."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def get_model_key(model) -> str:
    """Derive unique model identity from architecture parameters."""
    args = model.args
    model_type = getattr(args, 'model_type', 'unknown')
    hidden_size = getattr(args, 'hidden_size', 0)
    num_layers = getattr(args, 'num_hidden_layers', 0)
    num_kv_heads = getattr(args, 'num_key_value_heads', 0)
    return f"{model_type}_h{hidden_size}_l{num_layers}_kv{num_kv_heads}"


def get_calibration_path(model, compression_ratio: float) -> Path:
    """Get standard calibration cache path for a model."""
    key = get_model_key(model)
    return CALIBRATION_CACHE_DIR / f"{key}_{compression_ratio}x.pkl"


def auto_calibrate(
    model,
    tokenizer=None,
    compression_ratio: float = 2.0,
    force: bool = False,
) -> Optional[str]:
    """
    Auto-calibrate AM compression. Returns path to calibration file.

    Checks cache first. If not found and tokenizer is available, runs
    fast calibration (~40-60s) and saves for future sessions.

    Args:
        model: The language model (must have .args and .model.layers).
        tokenizer: Tokenizer for encoding calibration prompts. Required for
            first-time calibration; None skips calibration.
        compression_ratio: AM compression ratio (default 2.0).
        force: If True, re-calibrate even if cached file exists.

    Returns:
        Path to calibration .pkl file, or None if calibration unavailable.
    """
    # Get inner model (handle wrapper)
    inner = model.model if hasattr(model, 'model') else model

    cache_path = get_calibration_path(inner, compression_ratio)

    # Check cache
    if not force and cache_path.exists():
        print(f"[AutoCalibrate] Using cached: {cache_path}")
        return str(cache_path)

    # Cannot calibrate without tokenizer
    if tokenizer is None:
        print("[AutoCalibrate] No tokenizer available, cannot auto-calibrate.")
        return None

    # Run calibration
    print(f"[AutoCalibrate] First-time calibration for {get_model_key(inner)} "
          f"(ratio={compression_ratio}x)...")
    t0 = time.perf_counter()

    calibrator = AMCalibrator(model, tokenizer)
    calibration = calibrator.run(compression_ratio)

    # Save
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _save_calibration(calibration, inner, compression_ratio, cache_path)

    elapsed = time.perf_counter() - t0
    print(f"[AutoCalibrate] Done in {elapsed:.1f}s. Cached at: {cache_path}")

    return str(cache_path)


def _save_calibration(
    calibration: Dict[int, Dict],
    model,
    compression_ratio: float,
    path: Path,
):
    """Save calibration dict to .pkl file (compatible with existing format)."""
    # Convert MLX arrays to numpy for pickle
    calibration_np = {}
    for layer_idx, params in calibration.items():
        calibration_np[layer_idx] = {
            'Ck': np.array(params['Ck']) if isinstance(params['Ck'], mx.array) else params['Ck'],
            'beta': np.array(params['beta']) if isinstance(params['beta'], mx.array) else params['beta'],
            'selected_indices': (np.array(params['selected_indices'])
                                 if isinstance(params['selected_indices'], mx.array)
                                 else params['selected_indices']),
            'compression_ratio': params['compression_ratio'],
            'budget': params['budget'],
        }

    data = {
        'model_name': get_model_key(model),
        'compression_ratio': compression_ratio,
        'num_layers': len(calibration),
        'calibration': calibration_np,
        'created_at': datetime.now().isoformat(),
        'version': '2.0',
        'calibration_method': 'auto',
    }

    with open(path, 'wb') as f:
        pickle.dump(data, f)


class AMCalibrator:
    """Streamlined AM calibration — fast, no scipy dependency."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.inner = model.model if hasattr(model, 'model') else model
        self.num_layers = len(self.inner.layers)

    def run(self, compression_ratio: float = 2.0) -> Dict[int, Dict]:
        """Run full calibration pipeline. Returns calibration dict."""
        from mlx_lm.models.cache import KVCache

        # Step 1: Generate queries via repeat-prefill + QA prefill
        print(f"  [1/3] Generating queries...", flush=True)
        queries_per_layer = self._generate_queries()
        total_q = queries_per_layer[0].shape[2]
        print(f"         {total_q} queries collected", flush=True)

        # Step 2: Generate reference KV cache (~512 tokens)
        print(f"  [2/3] Generating reference keys...", flush=True)
        ref_keys_per_layer = self._generate_reference_keys()
        ref_len = ref_keys_per_layer[0].shape[2]
        print(f"         {ref_len} reference tokens", flush=True)

        # Step 3: Fit AM parameters per layer
        print(f"  [3/3] Fitting {self.num_layers} layers...", flush=True)
        calibration = {}
        for layer_idx in range(self.num_layers):
            calibration[layer_idx] = self._fit_layer(
                layer_idx,
                queries_per_layer[layer_idx],
                ref_keys_per_layer[layer_idx],
                compression_ratio,
            )
            if (layer_idx + 1) % 10 == 0 or layer_idx == self.num_layers - 1:
                print(f"         Layer {layer_idx + 1}/{self.num_layers} done", flush=True)

        return calibration

    def _generate_queries(self) -> List[mx.array]:
        """Generate Q tensors from diverse prompts (prefill-only, no decode)."""
        from mlx_lm.models.cache import KVCache

        all_queries = [[] for _ in range(self.num_layers)]

        # Part 1: Repeat-prefill (8 corpus × 5 repeats)
        for ci, corpus in enumerate(_CALIBRATION_CORPUS):
            tokens = self.tokenizer.encode(corpus)
            if len(tokens) < 10:
                continue

            for rep in range(_NUM_REPEATS):
                cache = [KVCache() for _ in range(self.num_layers)]
                y = mx.array([tokens])
                _ = self.model(y[:, :-1], cache=cache)
                mx.eval([c.state for c in cache])

                for li in range(self.num_layers):
                    all_queries[li].append(cache[li].keys)

                del cache

        # Part 2: QA prefill (12 questions)
        base_corpus = _CALIBRATION_CORPUS[0]  # Use first corpus as context
        for qi, question in enumerate(_CALIBRATION_QUESTIONS):
            prompt = f"{base_corpus}\n\nQuestion: {question}\nAnswer:"
            tokens = self.tokenizer.encode(prompt)

            cache = [KVCache() for _ in range(self.num_layers)]
            y = mx.array([tokens])
            _ = self.model(y[:, :-1], cache=cache)
            mx.eval([c.state for c in cache])

            for li in range(self.num_layers):
                all_queries[li].append(cache[li].keys)

            del cache

        # Merge along seq_len dimension
        merged = []
        for li in range(self.num_layers):
            merged.append(mx.concatenate(all_queries[li], axis=2))

        return merged

    def _generate_reference_keys(self) -> List[mx.array]:
        """Generate reference KV cache (~512 tokens) for importance scoring."""
        from mlx_lm.models.cache import KVCache

        # Concatenate first 2 corpus entries for ~512 tokens
        text = _CALIBRATION_CORPUS[0] + "\n\n" + _CALIBRATION_CORPUS[1]
        tokens = self.tokenizer.encode(text)

        # Pad or truncate to ~512
        if len(tokens) > 520:
            tokens = tokens[:512]

        cache = [KVCache() for _ in range(self.num_layers)]
        y = mx.array([tokens])
        _ = self.model(y[:, :-1], cache=cache)
        mx.eval([c.state for c in cache])

        ref_keys = [cache[li].keys for li in range(self.num_layers)]
        return ref_keys

    def _fit_layer(
        self,
        layer_idx: int,
        queries: mx.array,
        keys: mx.array,
        compression_ratio: float,
    ) -> Dict[str, Any]:
        """
        Fit AM parameters for a single layer.

        Only computes selected_indices (top-k by attention score).
        Beta set to ones (scored_pq doesn't use beta at runtime).
        Ck extracted for backward compatibility with pipeline AM strategies.
        """
        # queries: (1, n_kv_heads, num_queries, head_dim)
        # keys:    (1, n_kv_heads, seq_len, head_dim)
        # We average across heads (use head 0 for simplicity, same as offline)
        q_np = np.array(queries[0, 0, :, :].astype(mx.float32))  # (num_queries, head_dim)
        k_np = np.array(keys[0, 0, :, :].astype(mx.float32))     # (seq_len, head_dim)

        seq_len = k_np.shape[0]
        budget = int(seq_len / compression_ratio)

        # Attention scores → softmax → average → top-k
        raw_scores = q_np @ k_np.T  # (num_queries, seq_len)
        scores = _softmax(raw_scores, axis=1)
        avg_scores = np.mean(scores, axis=0)  # (seq_len,)

        # Select top-budget keys by importance
        selected_indices = np.argsort(avg_scores)[-budget:]
        selected_indices = np.sort(selected_indices)

        # Extract Ck for backward compat
        Ck = k_np[selected_indices]  # (budget, head_dim)

        # Beta = ones (scored_pq doesn't use beta; pipeline AM can still work)
        beta = np.ones(budget, dtype=np.float32)

        return {
            'Ck': Ck,
            'beta': beta,
            'selected_indices': selected_indices.astype(np.int32),
            'compression_ratio': compression_ratio,
            'budget': budget,
        }
