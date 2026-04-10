# FlashMLX Vision-Language Models (VLM) Guide

Complete guide for using Vision-Language Models with FlashMLX optimizations.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Cache Strategies](#cache-strategies)
- [Performance Guide](#performance-guide)
- [API Reference](#api-reference)
- [Examples](#examples)

## Quick Start

```python
from flashmlx.vlm import load_vlm_components
from flashmlx.generation import VLMGenerator, create_vlm_cache

# 1. Load VLM components
model, tokenizer, processor, config = load_vlm_components(
    "mlx-community/Qwen2-VL-2B-Instruct-bf16"
)

# 2. Create optimized cache
cache = create_vlm_cache(model, kv_cache="standard")

# 3. Create generator
generator = VLMGenerator(model, tokenizer, config.image_token_id)

# 4. Generate text
response = generator.generate("What is MLX?", cache=cache)
print(response)
```

## Installation

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- 8GB+ RAM (16GB recommended for 2B models)

### Install FlashMLX

```bash
git clone https://github.com/your-repo/FlashMLX.git
cd FlashMLX
pip install -e .
```

### Download VLM Model

Models are automatically downloaded from HuggingFace on first use:

```python
# Automatically downloads ~4GB model
load_vlm_components("mlx-community/Qwen2-VL-2B-Instruct-bf16")
```

## Basic Usage

### Text-Only Generation

```python
from flashmlx.vlm import load_vlm_components
from flashmlx.generation import VLMGenerator, create_vlm_cache

# Load
model, tokenizer, processor, config = load_vlm_components(
    "mlx-community/Qwen2-VL-2B-Instruct-bf16"
)

# Setup
cache = create_vlm_cache(model, kv_cache="standard")
generator = VLMGenerator(model, tokenizer, config.image_token_id)

# Generate
response = generator.generate(
    prompt="Explain deep learning in one sentence.",
    cache=cache,
    use_chat_template=True,
)
```

### Vision+Text Generation

```python
from PIL import Image

# Prepare image
image = Image.open("photo.jpg")

# Preprocess
pixel_values = processor.preprocess(image)
# ... (see examples/demo_vlm_simple.py for full preprocessing)

# Format prompt with image tokens
image_tokens = "<|image_pad|>" * 256
prompt = f"{image_tokens}\nWhat is in this image?"

# Generate
response = generator.generate(
    prompt=prompt,
    pixel_values=pixel_values,
    grid_thw=grid_thw,
    cache=cache,
    use_chat_template=True,
)
```

## Advanced Features

### Multi-turn Conversations

Cache reuse speeds up multi-turn conversations:

```python
# Create cache once
cache = create_vlm_cache(model, kv_cache="standard")
generator = VLMGenerator(model, tokenizer, config.image_token_id)

# Multiple turns with shared cache
questions = [
    "What is machine learning?",
    "Can you explain neural networks?",
    "How does backpropagation work?",
]

for question in questions:
    response = generator.generate(question, cache=cache)
    print(f"Q: {question}")
    print(f"A: {response}\n")
```

### Batch Processing

Process multiple images or questions efficiently:

```python
# Same image, different questions
questions = [
    "What is in this image?",
    "What colors do you see?",
    "Is this indoors or outdoors?",
]

for question in questions:
    prompt = f"{image_tokens}\n{question}"
    response = generator.generate(
        prompt=prompt,
        pixel_values=pixel_values,
        grid_thw=grid_thw,
        cache=cache,
    )
    print(f"{question}: {response}")
```

## Cache Strategies

FlashMLX provides optimized KV cache compression strategies:

| Strategy | Speed | Memory | Quality | Use Case |
|----------|-------|--------|---------|----------|
| **standard** | Baseline | 100% | Perfect | Production (recommended) |
| **triple_pq** | +5-43% | 72% | Good (short) | Experimental |
| **scored_pq** | +5-43% | 81% | Needs calibration | Future optimization |

### Standard Cache (Recommended)

```python
cache = create_vlm_cache(model, kv_cache="standard")
```

- ✅ Perfect quality preservation
- ✅ Stable across all context lengths
- ✅ Production-ready
- ❌ No memory savings

### Compressed Cache (Experimental)

```python
cache = create_vlm_cache(model, kv_cache="triple_pq")
```

- ✅ 5-43% faster (context-dependent)
- ✅ 72% memory savings for KV cache
- ⚠️ Quality degrades on long contexts (4K+ tokens)
- ⚠️ Requires calibration for production use

### Performance Comparison

**Short Context (<2K tokens)**:
- Standard: 52 tok/s, perfect quality
- Compressed: 55 tok/s (+5%), perfect quality

**Vision+Text (256 image tokens)**:
- Standard: 11.2 tok/s, perfect quality
- Compressed: 16.1 tok/s (+43.6%), quality OK for first turn

**Long Context (4K+ tokens)**:
- Standard: 1891 tok/s, perfect quality
- Compressed: 1896 tok/s (+0.2%), ⚠️ quality degradation

## Performance Guide

### Hardware Requirements

| Model | RAM | Recommended HW |
|-------|-----|----------------|
| Qwen2-VL-2B-bf16 | 4-6 GB | M1/M2/M3 (8GB+) |
| Qwen2-VL-7B-bf16 | 14-16 GB | M2/M3 Pro (16GB+) |

### Optimization Tips

1. **Use Standard Cache for Production**
   ```python
   cache = create_vlm_cache(model, kv_cache="standard")
   ```

2. **Reuse Cache Across Turns**
   ```python
   # Create cache once
   cache = create_vlm_cache(model, kv_cache="standard")

   # Reuse for multiple generations
   for question in questions:
       response = generator.generate(question, cache=cache)
   ```

3. **Monitor Memory Usage**
   ```python
   import mlx.core as mx

   mx.eval(model.parameters())
   mem_before = mx.metal.get_active_memory() / 1024**2  # MB

   # ... generate ...

   mem_after = mx.metal.get_active_memory() / 1024**2  # MB
   print(f"Memory used: {mem_after - mem_before:.1f} MB")
   ```

4. **Optimize Generation Parameters**
   ```python
   generator = VLMGenerator(
       model=model,
       tokenizer=tokenizer,
       image_token_id=config.image_token_id,
       max_tokens=50,  # Lower for faster generation
   )

   response = generator.generate(
       prompt=prompt,
       temperature=0.0,  # Greedy sampling (fastest)
       cache=cache,
   )
   ```

### Benchmarking

Run benchmarks to measure performance on your hardware:

```bash
# Simple benchmark
python examples/bench_vlm_vision_cache.py

# Long context benchmark
python examples/bench_vlm_long_context.py
```

## API Reference

### load_vlm_components()

```python
def load_vlm_components(
    model_path: str,
    use_4bit: bool = False
) -> tuple[Model, Tokenizer, Processor, Config]
```

Load VLM components from HuggingFace.

**Parameters:**
- `model_path`: Model identifier (e.g., "mlx-community/Qwen2-VL-2B-Instruct-bf16")
- `use_4bit`: Use 4-bit quantized model (experimental)

**Returns:** (model, tokenizer, processor, config)

### create_vlm_cache()

```python
def create_vlm_cache(
    model: nn.Module,
    kv_cache: str = "standard",
    **kwargs
) -> List[Cache]
```

Create optimized KV cache for VLM.

**Parameters:**
- `model`: VLM model instance
- `kv_cache`: Cache strategy ("standard", "triple_pq", "scored_pq")
- `**kwargs`: Additional cache parameters

**Returns:** List of cache objects (one per layer)

### VLMGenerator

```python
class VLMGenerator:
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        image_token_id: int = 151655,
        max_tokens: int = 512,
    )

    def generate(
        self,
        prompt: str,
        pixel_values: Optional[mx.array] = None,
        grid_thw: Optional[mx.array] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        cache = None,
        use_chat_template: bool = True,
    ) -> str
```

VLM text generator with chat template support.

**generate() Parameters:**
- `prompt`: Text prompt (can include image tokens)
- `pixel_values`: Image tensor [batch, C, T, H, W] (optional)
- `grid_thw`: Grid dimensions [batch, 3] (optional)
- `max_tokens`: Max tokens to generate
- `temperature`: Sampling temperature (0.0 = greedy)
- `cache`: FlashMLX optimized cache
- `use_chat_template`: Format with chat template

**Returns:** Generated text string

## Examples

### Example 1: Simple Text Generation

```python
from flashmlx.vlm import load_vlm_components
from flashmlx.generation import VLMGenerator, create_vlm_cache

# Load
model, tokenizer, processor, config = load_vlm_components(
    "mlx-community/Qwen2-VL-2B-Instruct-bf16"
)

# Setup
cache = create_vlm_cache(model, kv_cache="standard")
generator = VLMGenerator(model, tokenizer, config.image_token_id)

# Generate
response = generator.generate("What is MLX?", cache=cache)
print(response)
```

See: `examples/demo_vlm_simple.py`

### Example 2: Vision Understanding

```python
from PIL import Image
from test_real_weights import prepare_test_image

# Load image
image = Image.open("cat.jpg")
pixel_values, grid_thw = prepare_test_image(processor)

# Format prompt
image_tokens = "<|image_pad|>" * 256
prompt = f"{image_tokens}\nWhat is in this image?"

# Generate
response = generator.generate(
    prompt=prompt,
    pixel_values=pixel_values,
    grid_thw=grid_thw,
    cache=cache,
)
```

See: `examples/demo_vlm_simple.py`

### Example 3: Multi-turn Conversation

```python
cache = create_vlm_cache(model, kv_cache="standard")
generator = VLMGenerator(model, tokenizer, config.image_token_id)

questions = [
    "What is machine learning?",
    "Can you explain neural networks?",
    "How does backpropagation work?",
]

for question in questions:
    response = generator.generate(question, cache=cache)
    print(f"Q: {question}")
    print(f"A: {response}\n")
```

See: `examples/demo_vlm_advanced.py`

### Example 4: Batch Processing

```python
# Process multiple images
images = ["img1.jpg", "img2.jpg", "img3.jpg"]

for img_path in images:
    pixel_values, grid_thw = prepare_image(img_path)

    prompt = f"{image_tokens}\nDescribe this image."
    response = generator.generate(
        prompt=prompt,
        pixel_values=pixel_values,
        grid_thw=grid_thw,
        cache=cache,
    )
    print(f"{img_path}: {response}")
```

See: `examples/demo_vlm_advanced.py`

## Troubleshooting

### Issue: Out of Memory

**Solution:** Use smaller model or reduce max_tokens:

```python
# Use 2B model instead of 7B
model_path = "mlx-community/Qwen2-VL-2B-Instruct-bf16"

# Reduce max tokens
generator = VLMGenerator(
    model, tokenizer, config.image_token_id,
    max_tokens=50  # Instead of 512
)
```

### Issue: Slow Generation

**Solution:** Use greedy sampling and standard cache:

```python
cache = create_vlm_cache(model, kv_cache="standard")

response = generator.generate(
    prompt=prompt,
    temperature=0.0,  # Greedy = faster
    cache=cache,
)
```

### Issue: Poor Vision Understanding

**Solution:** Check image preprocessing and prompt format:

```python
# Ensure proper preprocessing
pixel_values, grid_thw = prepare_test_image(processor)

# Use correct image token format
image_tokens = "<|image_pad|>" * 256  # Qwen2-VL expects 256 tokens
prompt = f"{image_tokens}\n{your_question}"

# Use chat template
response = generator.generate(
    prompt=prompt,
    pixel_values=pixel_values,
    grid_thw=grid_thw,
    use_chat_template=True,  # Important!
)
```

### Issue: Quality Degradation with Compressed Cache

**Solution:** Use standard cache for production:

```python
# Compressed cache is experimental
# For production, use standard:
cache = create_vlm_cache(model, kv_cache="standard")
```

## Roadmap

### Current (v0.2.0)

- ✅ Qwen2-VL support
- ✅ Text + Vision generation
- ✅ Chat template support
- ✅ Standard cache (production-ready)
- ✅ Experimental compressed cache

### Planned (v0.3.0)

- ⏳ Cache calibration for quality
- ⏳ More VLM architectures (LLaVA, InternVL)
- ⏳ Streaming generation
- ⏳ Batch inference optimization

### Future

- Multi-image support
- Video understanding
- Fine-tuning support

## Contributing

See `CONTRIBUTING.md` for development guidelines.

## License

Apache 2.0 - See `LICENSE` file.

## Citation

If you use FlashMLX VLM in your research, please cite:

```bibtex
@software{flashmlx_vlm,
  title = {FlashMLX: Optimized Vision-Language Models for Apple Silicon},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-repo/FlashMLX}
}
```

## Support

- Issues: https://github.com/your-repo/FlashMLX/issues
- Discussions: https://github.com/your-repo/FlashMLX/discussions
- Discord: [Coming soon]
