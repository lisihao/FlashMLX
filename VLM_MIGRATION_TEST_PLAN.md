# FlashMLX VLM 迁移测试计划

**版本**: v1.0
**创建日期**: 2026-04-09
**负责人**: 测试团队
**审核**: Claude Opus 4.6

---

## 📋 测试策略概览

### 测试金字塔

```
                    ┌──────────────┐
                    │  E2E Tests   │  ← 10% (关键路径验证)
                    │  (12 cases)  │
                    └──────────────┘
                  ┌──────────────────┐
                  │ Integration Tests│  ← 30% (模块协同)
                  │   (45 cases)     │
                  └──────────────────┘
              ┌────────────────────────┐
              │   Unit Tests           │  ← 60% (组件隔离)
              │   (120 cases)          │
              └────────────────────────┘

总计: 177 test cases
预估执行时间:
  - Unit: 5 min (并行)
  - Integration: 20 min
  - E2E: 45 min
  - 总计: ~70 min per run
```

### 分阶段测试重点

| Phase | 测试重点 | 覆盖率目标 | 关键指标 |
|-------|---------|-----------|---------|
| **Phase 1** | 文本模型回归 | 100% 现有功能 | TG speed ±5% |
| **Phase 2** | VLM 功能验证 | 80% VLM 基础 | 推理完成率 100% |
| **Phase 3** | VLM 性能优化 | 90% VLM 全量 | PP/TG <50% 退化 |
| **Phase 4** | 生产就绪 | 95% 全量 | 质量基准达标 |

---

## 第一部分：Phase 1 测试计划（文本模型回归）

### 1.1 测试目标

**验证**: Git merge 后，所有文本模型功能和性能无退化

**测试范围**:
- ✅ 3 个主要文本模型（Mistral, Qwen3, Llama）
- ✅ 5 种 KV cache 策略（standard, triple_pq, scored_pq, scored_kv, triple_tq）
- ✅ 4 种上下文长度（4K, 8K, 16K, 32K）
- ✅ Route 3/4/5 核心功能

**成功标准**:
- ✅ 所有现有单元测试通过（100%）
- ✅ TG speed 误差 ±5%
- ✅ Memory 误差 ±10%
- ✅ 输出质量无退化（PPL ±0.5%）

---

### 1.2 单元测试（60 cases）

#### A. Cache Factory 测试（15 cases）

```python
# tests/test_cache_factory_regression.py

class TestCacheFactoryRegression:
    """验证 cache_factory.py API 向后兼容"""

    def test_make_prompt_cache_signature_backward_compat(self):
        """验证 make_prompt_cache() 签名兼容"""
        model = load_model("mlx-community/Mistral-7B-v0.3-4bit")

        # 旧签名（13 个参数）应该仍然工作
        cache = make_prompt_cache(
            model=model,
            kv_cache="triple_pq",
            kv_warm_bits=4,
            kv_flat_quant="q8_0",
            # ... 其他现有参数
        )

        assert cache is not None
        assert isinstance(cache, TripleLayerKVCache)

    def test_detect_architecture_text_models(self):
        """验证文本模型检测无变化"""
        models = [
            "Mistral-7B",
            "Qwen3-8B",
            "Llama-3.2-3B"
        ]

        for model_name in models:
            model = load_model(model_name)
            is_hybrid, attn_idx, native, vision_count = _detect_architecture(model)

            # 文本模型应该 vision_count == 0
            assert vision_count == 0, f"{model_name} detected as VLM"

    def test_triple_pq_bug_fix_preserved(self):
        """验证 triple_pq lazy_prefill_threshold 修复仍在"""
        config = CacheConfig(kv_cache="triple_pq")
        cache_kwargs = config.to_cache_kwargs()

        assert cache_kwargs["lazy_prefill_threshold"] == 32768

    # ... 其他 12 个测试
```

#### B. Triple Layer Cache 测试（20 cases）

```python
# tests/test_triple_layer_cache_regression.py

class TestTripleLayerCacheRegression:
    """验证三层缓存逻辑无退化"""

    @pytest.mark.parametrize("context_len", [4096, 8192, 16384, 32768])
    def test_l0_l1_l2_boundaries(self, context_len):
        """验证 L0/L1/L2 边界逻辑"""
        cache = TripleLayerKVCache(
            recent_size=512,
            warm_size=1536,
            num_heads=32,
            head_dim=128,
        )

        # 模拟 prefill
        keys, values = generate_dummy_kv(context_len)
        cache.update_and_fetch(keys, values)

        # 验证状态
        assert cache._current_state in ["recent", "warm", "cold"]
        if context_len <= 512:
            assert cache._current_state == "recent"
        elif context_len <= 2048:
            assert cache._current_state == "warm"
        else:
            assert cache._current_state == "cold"

    def test_flat_buffer_upgrade(self):
        """验证 flat buffer 升级逻辑"""
        cache = TripleLayerKVCache(
            flat_quant="q8_0",
            lazy_prefill_threshold=32768,
        )

        # Prefill 阶段
        keys_pp, values_pp = generate_dummy_kv(4096)
        cache.update_and_fetch(keys_pp, values_pp)

        # TG 阶段第一个 token
        keys_tg, values_tg = generate_dummy_kv(1)
        cache.update_and_fetch(keys_tg, values_tg)

        # 验证升级到 flat buffer
        assert cache._flat_k is not None
        assert cache._flat_v is not None
        assert cache._flat_k.dtype == mx.uint8  # Q8_0

    # ... 其他 18 个测试
```

#### C. KV-Direct Cache 测试（15 cases）

```python
# tests/test_kv_direct_cache_regression.py

class TestKVDirectCacheRegression:
    """验证 Route 5 重建逻辑"""

    def test_h0_store_capture(self):
        """验证 h^(0) 捕获"""
        model = load_model("Qwen3-8B")
        _install_h0_capture(model)

        assert hasattr(model, '_h0_store_text')
        assert model._h0_store_text.count == 0

        # 执行 prefill
        tokens = mx.array([[1, 2, 3, 4, 5]])
        h0 = model.embed_tokens(tokens)

        # 验证捕获
        assert model._h0_store_text.count == 5

    def test_reconstruct_prefix_kv(self):
        """验证前缀重建精度"""
        model = load_model("Qwen3-8B")
        _install_h0_capture(model)

        # Prefill
        tokens = mx.random.randint(0, 1000, (1, 100))
        h0 = model.embed_tokens(tokens)

        # 重建前 50 个 tokens 的 K/V
        reconstructed_kv = reconstruct_prefix_kv(
            model, model._h0_store_text, 0, 10, tokens=50
        )

        # 验证形状
        assert reconstructed_kv[0].shape == (1, 8, 50, 128)  # K
        assert reconstructed_kv[1].shape == (1, 8, 50, 128)  # V

    # ... 其他 13 个测试
```

#### D. Generate Pipeline 测试（10 cases）

```python
# tests/test_generate_regression.py

class TestGenerateRegression:
    """验证 generate 流程无退化"""

    @pytest.mark.parametrize("model_name", [
        "Mistral-7B-v0.3-4bit",
        "Qwen3-8B-4bit",
        "Llama-3.2-3B"
    ])
    def test_generate_basic(self, model_name):
        """验证基础生成"""
        model, tokenizer = load_model(model_name)

        prompt = "Once upon a time"
        output = generate(
            model, tokenizer, prompt,
            max_tokens=50,
            temp=0.0  # 确定性输出
        )

        assert len(output) > len(prompt)
        assert output.startswith(prompt)

    def test_chunked_prefill(self):
        """验证 Route 4 chunked prefill"""
        model, tokenizer = load_model("Qwen3-8B")

        prompt = "A" * 8000  # 长 prompt
        output = generate(
            model, tokenizer, prompt,
            max_tokens=10,
            chunk_size=512,  # 分块 prefill
        )

        # 验证内存峰值未超载
        # (需要 profiler 支持)

    # ... 其他 8 个测试
```

---

### 1.3 集成测试（20 cases）

#### A. 端到端性能测试（8 cases）

```python
# tests/test_e2e_performance_regression.py

class TestE2EPerformanceRegression:
    """端到端性能回归测试"""

    @pytest.fixture(scope="class")
    def baseline_metrics(self):
        """加载 Phase 1 前的基准数据"""
        return json.load(open("benchmarks/phase0_baseline.json"))

    @pytest.mark.parametrize("config", [
        {"model": "Mistral-7B", "context": 4096, "strategy": "triple_pq"},
        {"model": "Qwen3-8B", "context": 8192, "strategy": "scored_pq"},
        {"model": "Qwen3-8B", "context": 32768, "strategy": "scored_pq"},
        {"model": "Llama-3.2-3B", "context": 16384, "strategy": "triple_pq_am"},
    ])
    def test_tg_speed_regression(self, config, baseline_metrics):
        """验证 TG speed 无退化"""
        model, tokenizer = load_model(config["model"])
        cache = make_prompt_cache(model, kv_cache=config["strategy"])

        # Benchmark
        result = benchmark_tg_speed(
            model, tokenizer, cache,
            context_len=config["context"],
            num_tokens=100
        )

        # 对比 baseline
        key = f"{config['model']}_{config['context']}_{config['strategy']}"
        baseline_speed = baseline_metrics[key]["tg_speed"]

        # 允许 ±5% 误差
        assert abs(result["tg_speed"] - baseline_speed) / baseline_speed < 0.05, \
            f"TG speed regression: {result['tg_speed']:.1f} vs baseline {baseline_speed:.1f}"

    @pytest.mark.parametrize("config", [
        {"model": "Qwen3-8B", "context": 32768, "strategy": "scored_pq"},
    ])
    def test_memory_regression(self, config, baseline_metrics):
        """验证内存峰值无退化"""
        model, tokenizer = load_model(config["model"])
        cache = make_prompt_cache(model, kv_cache=config["strategy"])

        # Profiler
        with MemoryProfiler() as profiler:
            result = benchmark_prefill_and_tg(
                model, tokenizer, cache,
                context_len=config["context"]
            )

        # 对比 baseline
        key = f"{config['model']}_{config['context']}_{config['strategy']}"
        baseline_mem = baseline_metrics[key]["tg_memory_mb"]

        # 允许 ±10% 误差
        assert abs(profiler.peak_mb - baseline_mem) / baseline_mem < 0.10, \
            f"Memory regression: {profiler.peak_mb:.0f}MB vs baseline {baseline_mem:.0f}MB"
```

#### B. 质量回归测试（12 cases）

```python
# tests/test_quality_regression.py

class TestQualityRegression:
    """输出质量回归测试"""

    @pytest.fixture(scope="class")
    def golden_outputs(self):
        """加载金标准输出"""
        return json.load(open("tests/fixtures/golden_outputs.json"))

    @pytest.mark.parametrize("model_name", [
        "Mistral-7B-v0.3-4bit",
        "Qwen3-8B-4bit",
        "Llama-3.2-3B"
    ])
    def test_deterministic_output(self, model_name, golden_outputs):
        """验证确定性输出（temp=0）"""
        model, tokenizer = load_model(model_name)

        test_prompts = [
            "Once upon a time",
            "The capital of France is",
            "1 + 1 = "
        ]

        for prompt in test_prompts:
            output = generate(
                model, tokenizer, prompt,
                max_tokens=20,
                temp=0.0
            )

            golden_key = f"{model_name}_{hash(prompt)}"
            if golden_key in golden_outputs:
                assert output == golden_outputs[golden_key], \
                    f"Output changed for {model_name} on '{prompt}'"

    @pytest.mark.parametrize("config", [
        {"model": "Qwen3-8B", "strategy": "scored_pq", "context": 8192},
        {"model": "Qwen3-8B", "strategy": "triple_pq", "context": 4096},
    ])
    def test_perplexity_regression(self, config):
        """验证压缩策略对 PPL 的影响"""
        model, tokenizer = load_model(config["model"])

        # Standard cache (baseline)
        cache_baseline = make_prompt_cache(model, kv_cache="standard")
        ppl_baseline = compute_perplexity(
            model, tokenizer, cache_baseline,
            dataset="wikitext-2-validation",
            context_len=config["context"]
        )

        # Optimized cache
        cache_opt = make_prompt_cache(model, kv_cache=config["strategy"])
        ppl_opt = compute_perplexity(
            model, tokenizer, cache_opt,
            dataset="wikitext-2-validation",
            context_len=config["context"]
        )

        # PPL 退化 < 0.5%
        assert abs(ppl_opt - ppl_baseline) / ppl_baseline < 0.005, \
            f"PPL regression: {ppl_opt:.4f} vs baseline {ppl_baseline:.4f}"
```

---

### 1.4 基准数据建立

#### A. 性能基准（phase0_baseline.json）

```json
{
  "Mistral-7B_4096_triple_pq": {
    "pp_speed": 1250.3,
    "tg_speed": 24.7,
    "pp_memory_mb": 3200,
    "tg_memory_mb": 2800,
    "timestamp": "2026-04-09T10:00:00Z"
  },
  "Qwen3-8B_8192_scored_pq": {
    "pp_speed": 320.5,
    "tg_speed": 18.2,
    "pp_memory_mb": 4100,
    "tg_memory_mb": 3500,
    "timestamp": "2026-04-09T10:15:00Z"
  },
  "Qwen3-8B_32768_scored_pq": {
    "pp_speed": 409.5,
    "tg_speed": 21.6,
    "pp_memory_mb": 550,
    "tg_memory_mb": 529,
    "timestamp": "2026-04-09T10:30:00Z"
  },
  "Llama-3.2-3B_16384_triple_pq_am": {
    "pp_speed": 2100.0,
    "tg_speed": 52.3,
    "pp_memory_mb": 1800,
    "tg_memory_mb": 1500,
    "timestamp": "2026-04-09T10:45:00Z"
  }
}
```

#### B. 质量基准（golden_outputs.json）

```json
{
  "Mistral-7B-v0.3-4bit_hash123": "Once upon a time, there was a young girl named Sophia who lived in",
  "Qwen3-8B-4bit_hash456": "The capital of France is Paris, which is located in the north-central",
  "Llama-3.2-3B_hash789": "1 + 1 = 2\n\nThis is a basic arithmetic"
}
```

---

### 1.5 CI/CD 集成

#### A. GitHub Actions Workflow

```yaml
# .github/workflows/phase1_regression.yml

name: Phase 1 - Text Model Regression

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [phase1-migration]

jobs:
  unit-tests:
    runs-on: macos-14  # M3 Max
    timeout-minutes: 15

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-xdist pytest-timeout

      - name: Run unit tests (parallel)
        run: |
          pytest tests/test_cache_factory_regression.py \
                 tests/test_triple_layer_cache_regression.py \
                 tests/test_kv_direct_cache_regression.py \
                 tests/test_generate_regression.py \
                 -n 4 --timeout=300

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: unit-test-results
          path: test-results/

  integration-tests:
    runs-on: self-hosted  # M4 Max 64GB
    timeout-minutes: 30
    needs: unit-tests

    steps:
      - uses: actions/checkout@v4

      - name: Load baseline metrics
        run: |
          cp benchmarks/phase0_baseline.json /tmp/

      - name: Run performance regression
        run: |
          pytest tests/test_e2e_performance_regression.py \
                 --baseline=/tmp/phase0_baseline.json \
                 -v

      - name: Run quality regression
        run: |
          pytest tests/test_quality_regression.py \
                 --golden=tests/fixtures/golden_outputs.json \
                 -v

      - name: Generate regression report
        if: always()
        run: |
          python scripts/generate_regression_report.py \
                 --output=regression_report.html

      - name: Upload regression report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: regression-report
          path: regression_report.html
```

---

## 第二部分：Phase 2 测试计划（VLM 功能验证）

### 2.1 测试目标

**验证**: VLM 模型能正确加载和推理

**测试范围**:
- ✅ 5 个 VLM 模型（Qwen2-VL, Qwen3-VL, Qwen3-VL-MOE, Kimi-VL, LFM2-VL）
- ✅ Vision encoder 权重加载
- ✅ Vision token 序列处理
- ✅ 多源 H0Store
- ✅ 多模态 generate pipeline

**成功标准**:
- ✅ Vision encoder 权重成功加载
- ✅ 推理完成率 100%
- ✅ 输出合理性检查通过
- ✅ 文本模型性能不受影响

---

### 2.2 单元测试（40 cases）

#### A. Vision Encoder 加载测试（8 cases）

```python
# tests/test_vision_encoder_loading.py

class TestVisionEncoderLoading:
    """验证 Vision encoder 权重正确加载"""

    @pytest.mark.parametrize("model_name", [
        "Qwen/Qwen2-VL-7B",
        "Qwen/Qwen3-VL",
        "Qwen/Qwen3-VL-MOE",
    ])
    def test_vision_encoder_exists(self, model_name):
        """验证 vision encoder 被加载（而非删除）"""
        model, tokenizer = load_model(model_name)

        # 关键: vision encoder 应该存在
        assert hasattr(model, 'visual') or hasattr(model, 'vision_tower'), \
            f"{model_name} vision encoder not loaded"

        vision_encoder = getattr(model, 'visual',
                                 getattr(model, 'vision_tower', None))
        assert vision_encoder is not None
        assert len(list(vision_encoder.parameters())) > 0, \
            "Vision encoder has no parameters"

    def test_vision_encoder_inference(self):
        """验证 vision encoder 可推理"""
        model, _ = load_model("Qwen/Qwen2-VL-7B")

        # 构造 dummy image
        dummy_image = mx.random.normal((3, 336, 336))

        # Vision encoder 应该返回 vision embeddings
        vision_output = model.visual(dummy_image)

        assert vision_output is not None
        assert len(vision_output.shape) == 3  # (batch, seq, dim)
        assert vision_output.shape[1] > 100  # 至少 100+ tokens
```

#### B. Vision Token 检测测试（10 cases）

```python
# tests/test_vision_token_detection.py

class TestVisionTokenDetection:
    """验证 vision token 数量检测"""

    def test_estimate_vision_tokens_qwen2vl(self):
        """验证 Qwen2-VL vision token 估算"""
        model, _ = load_model("Qwen/Qwen2-VL-7B")

        vision_count = _estimate_vision_tokens(model)

        # Qwen2-VL: ~576-2048 tokens per image
        assert 100 < vision_count < 5000
        print(f"Qwen2-VL vision_token_count: {vision_count}")

    def test_detect_architecture_vlm(self):
        """验证 VLM 架构检测"""
        model, _ = load_model("Qwen/Qwen2-VL-7B")

        is_hybrid, attn_idx, native, vision_count = _detect_architecture(model)

        # VLM 应该被正确识别
        assert vision_count > 0, "VLM not detected"
        print(f"Detected VLM: vision_count={vision_count}")

    def test_cache_factory_vlm_path(self):
        """验证 cache_factory 选择 VLM 路径"""
        model, _ = load_model("Qwen/Qwen2-VL-7B")

        with patch('mlx_lm.models.cache_factory.ENABLE_VLM_SUPPORT', True):
            cache = make_prompt_cache(model, kv_cache="triple_pq")

            # 应该使用 VLM-aware 缓存
            assert hasattr(cache, '_vision_token_count')
            assert cache._vision_token_count > 0
```

#### C. 多源 H0Store 测试（12 cases）

```python
# tests/test_multi_source_h0store.py

class TestMultiSourceH0Store:
    """验证 text 和 vision 的 h^(0) 分离存储"""

    def test_h0_store_installation_vlm(self):
        """验证 VLM 安装两个 H0Store"""
        model, _ = load_model("Qwen/Qwen2-VL-7B")

        _install_h0_capture(model, vision_aware=True)

        # 应该有两个 store
        assert hasattr(model, '_h0_store_text')
        assert hasattr(model, '_h0_store_vision')
        assert model._h0_store_text._source_type == 'text'
        assert model._h0_store_vision._source_type == 'vision'

    def test_h0_capture_text_tokens(self):
        """验证文本 tokens 的 h^(0) 捕获"""
        model, tokenizer = load_model("Qwen/Qwen2-VL-7B")
        _install_h0_capture(model, vision_aware=True)

        # 文本 tokens
        text_tokens = tokenizer.encode("Hello world")
        h0_text = model.embed_tokens(mx.array([text_tokens]))

        # 验证 text store 捕获
        assert model._h0_store_text.count == len(text_tokens)
        assert model._h0_store_vision.count == 0  # vision 未触发

    def test_h0_capture_vision_tokens(self):
        """验证 vision tokens 的 h^(0) 捕获"""
        model, _ = load_model("Qwen/Qwen2-VL-7B")
        _install_h0_capture(model, vision_aware=True)

        # Vision input
        dummy_image = mx.random.normal((3, 336, 336))
        h0_vision = model.visual(dummy_image)

        # 验证 vision store 捕获
        assert model._h0_store_vision.count > 0
        print(f"Vision h0 count: {model._h0_store_vision.count}")

    def test_reconstruct_from_text_h0(self):
        """验证从 text h^(0) 重建 K/V"""
        model, tokenizer = load_model("Qwen/Qwen2-VL-7B")
        _install_h0_capture(model, vision_aware=True)

        # Prefill text
        text_tokens = tokenizer.encode("A" * 100)
        h0 = model.embed_tokens(mx.array([text_tokens]))

        # 重建
        reconstructed = reconstruct_prefix_kv(
            model, model._h0_store_text, 0, 10, tokens=50
        )

        assert reconstructed[0].shape[2] == 50  # 50 tokens

    def test_reconstruct_from_vision_h0(self):
        """验证从 vision h^(0) 重建 K/V"""
        model, _ = load_model("Qwen/Qwen2-VL-7B")
        _install_h0_capture(model, vision_aware=True)

        # Vision prefill
        dummy_image = mx.random.normal((3, 336, 336))
        h0_vision = model.visual(dummy_image)

        # 重建 vision K/V
        vision_count = model._h0_store_vision.count
        reconstructed = reconstruct_prefix_kv(
            model, model._h0_store_vision, 0, 10, tokens=vision_count
        )

        assert reconstructed[0].shape[2] == vision_count
```

#### D. VLM Generate Pipeline 测试（10 cases）

```python
# tests/test_vlm_generate_pipeline.py

class TestVLMGeneratePipeline:
    """验证 VLM 多模态 generate pipeline"""

    def test_generate_with_single_image(self):
        """验证单张图像推理"""
        model, tokenizer = load_model("Qwen/Qwen2-VL-7B")

        image = load_test_image("tests/fixtures/cat.jpg")
        prompt = "Describe the image"

        output = generate(
            model, tokenizer, prompt,
            images=[image],
            max_tokens=50
        )

        # 验证输出
        assert len(output) > len(prompt)
        assert output != prompt
        print(f"Output: {output}")

    def test_generate_with_multiple_images(self):
        """验证多张图像推理"""
        model, tokenizer = load_model("Qwen/Qwen3-VL")

        images = [
            load_test_image("tests/fixtures/cat.jpg"),
            load_test_image("tests/fixtures/dog.jpg"),
            load_test_image("tests/fixtures/bird.jpg"),
        ]
        prompt = "What's common in these images?"

        output = generate(
            model, tokenizer, prompt,
            images=images,
            max_tokens=100
        )

        assert len(output) > len(prompt)
        # 简单合理性检查: 输出应该提到 "animal" 或相关词
        # (需要更严格的质量测试在 Phase 3)

    def test_vision_text_token_concatenation(self):
        """验证 vision + text tokens 拼接正确"""
        model, tokenizer = load_model("Qwen/Qwen2-VL-7B")

        image = load_test_image("tests/fixtures/cat.jpg")
        prompt = "This is a cat"

        # 提取 vision tokens
        vision_output = model.visual(preprocess_image(image))
        vision_count = vision_output.shape[1]

        # 提取 text tokens
        text_tokens = tokenizer.encode(prompt)
        text_count = len(text_tokens)

        # Generate
        output, cache_state = generate_with_cache_inspection(
            model, tokenizer, prompt, images=[image]
        )

        # 验证 cache 中 token 总数 = vision + text
        total_cached = cache_state['total_tokens']
        assert total_cached == vision_count + text_count
```

---

### 2.3 集成测试（15 cases）

#### A. 端到端 VLM 推理测试（8 cases）

```python
# tests/test_e2e_vlm_inference.py

class TestE2EVLMInference:
    """端到端 VLM 推理测试"""

    @pytest.mark.parametrize("config", [
        {"model": "Qwen/Qwen2-VL-7B", "images": 1, "context": 2048},
        {"model": "Qwen/Qwen3-VL", "images": 3, "context": 4096},
        {"model": "Qwen/Qwen3-VL-MOE", "images": 1, "context": 8192},
    ])
    def test_vlm_inference_completion(self, config):
        """验证 VLM 推理能完成"""
        model, tokenizer = load_model(config["model"])

        images = [load_test_image(f"fixture_{i}.jpg")
                  for i in range(config["images"])]
        prompt = "Describe these images"

        # 关键: 不要求速度，只要求完成
        with timeout(120):  # 2 分钟超时
            output = generate(
                model, tokenizer, prompt,
                images=images,
                max_tokens=100
            )

        # 验证
        assert len(output) > len(prompt), "No output generated"
        assert output != prompt, "Output unchanged"

    def test_vlm_with_cache_strategies(self):
        """验证 VLM 与不同 cache 策略"""
        model, tokenizer = load_model("Qwen/Qwen2-VL-7B")
        image = load_test_image("tests/fixtures/cat.jpg")
        prompt = "What is this?"

        strategies = ["standard", "triple_pq"]  # Phase 2 只测基础策略
        outputs = {}

        for strategy in strategies:
            cache = make_prompt_cache(model, kv_cache=strategy)
            output = generate(
                model, tokenizer, prompt,
                images=[image],
                cache=cache,
                max_tokens=50
            )
            outputs[strategy] = output
            print(f"{strategy}: {output}")

        # 验证: 不同策略应该产生相似输出 (质量测试在 Phase 3)
        # Phase 2 只验证都能完成
        assert all(len(out) > 0 for out in outputs.values())
```

#### B. VLM 功能正确性测试（7 cases）

```python
# tests/test_vlm_functional_correctness.py

class TestVLMFunctionalCorrectness:
    """VLM 功能正确性测试"""

    def test_vision_token_boundary_adjustment(self):
        """验证 vision tokens 导致 L0 边界调整"""
        model, _ = load_model("Qwen/Qwen2-VL-7B")

        # 检测 vision token count
        vision_count = _estimate_vision_tokens(model)

        # 创建 cache
        cache = make_prompt_cache(model, kv_cache="triple_pq")

        # 验证 L0 边界被扩展
        assert cache.recent_size >= vision_count, \
            f"L0 size {cache.recent_size} < vision_count {vision_count}"
        print(f"L0 adjusted to {cache.recent_size} for {vision_count} vision tokens")

    def test_vision_h0_isolation(self):
        """验证 vision 和 text h^(0) 隔离"""
        model, tokenizer = load_model("Qwen/Qwen2-VL-7B")
        _install_h0_capture(model, vision_aware=True)

        # Vision input
        image = load_test_image("tests/fixtures/cat.jpg")
        vision_output = model.visual(preprocess_image(image))
        vision_count_before = model._h0_store_vision.count

        # Text input
        text_tokens = tokenizer.encode("Hello")
        h0_text = model.embed_tokens(mx.array([text_tokens]))

        # 验证: text 不影响 vision store
        assert model._h0_store_vision.count == vision_count_before
        # vision 不影响 text store
        assert model._h0_store_text.count == len(text_tokens)

    def test_vlm_memory_footprint(self):
        """验证 VLM 内存占用（初步）"""
        model, tokenizer = load_model("Qwen/Qwen2-VL-7B")

        with MemoryProfiler() as profiler:
            image = load_test_image("tests/fixtures/cat.jpg")
            prompt = "Describe"

            output = generate(
                model, tokenizer, prompt,
                images=[image],
                max_tokens=50
            )

        # Phase 2: 只验证不 OOM (不要求优化)
        assert profiler.peak_mb < 30000, f"Memory too high: {profiler.peak_mb}MB"
        print(f"VLM memory peak: {profiler.peak_mb}MB")
```

---

### 2.4 功能验证清单

```
VLM 功能验证清单 (Phase 2)
──────────────────────────────────────────

✓ P1: Vision encoder 权重加载
  ├─ Qwen2-VL visual encoder exists
  ├─ Qwen3-VL visual encoder exists
  ├─ Kimi-VL vision_tower exists
  └─ 权重参数 > 0

✓ P2: Vision token 检测
  ├─ _estimate_vision_tokens() 返回 > 0
  ├─ _detect_architecture() 识别 VLM
  └─ L0 边界自动调整

✓ P3: 多源 H0Store
  ├─ _h0_store_text 创建
  ├─ _h0_store_vision 创建
  ├─ text tokens → text store
  ├─ vision tokens → vision store
  └─ 两者隔离

✓ P4: VLM Generate Pipeline
  ├─ 单张图像推理完成
  ├─ 多张图像推理完成
  ├─ vision + text tokens 拼接正确
  └─ 输出合理性检查

✓ P5: 文本模型不受影响
  ├─ Mistral-7B TG speed ±5%
  ├─ Qwen3-8B TG speed ±5%
  └─ ENABLE_VLM_SUPPORT=False 回退正常
```

---

## 第三部分：Phase 3 测试计划（VLM 性能优化）

### 3.1 测试目标

**验证**: Vision tokens 压缩优化后，性能和质量达标

**测试范围**:
- ✅ Vision-aware quantization (bf16, q8_0, q4_0)
- ✅ Vision density router
- ✅ Vision AM scoring (跳过 vs 启用)
- ✅ 多上下文长度性能 (2K, 4K, 8K, 16K)
- ✅ 质量基准 (VQAV2, GQA)

**成功标准**:
- ✅ PP/TG 性能退化 <50%
- ✅ Memory 优化 >30%
- ✅ VQA 质量无损 (>95% baseline)

---

### 3.2 性能测试（20 cases）

#### A. VLM Benchmark Suite

```python
# tests/test_vlm_benchmark.py

class TestVLMBenchmark:
    """VLM 性能基准测试"""

    @pytest.fixture(scope="class")
    def vlm_baseline(self):
        """VLM Phase 2 baseline (无优化)"""
        return {
            "Qwen2-VL_2K_no-opt": {"pp_speed": 150, "tg_speed": 8.5, "memory": 6800},
            "Qwen3-VL_4K_no-opt": {"pp_speed": 120, "tg_speed": 7.2, "memory": 9200},
        }

    @pytest.mark.parametrize("config", [
        {"model": "Qwen2-VL", "context": 2048, "images": 1, "quant": "bf16"},
        {"model": "Qwen2-VL", "context": 2048, "images": 1, "quant": "q8_0"},
        {"model": "Qwen2-VL", "context": 4096, "images": 2, "quant": "q8_0"},
        {"model": "Qwen3-VL", "context": 4096, "images": 3, "quant": "q8_0"},
        {"model": "Qwen3-VL", "context": 8192, "images": 1, "quant": "q8_0"},
    ])
    def test_vlm_tg_speed(self, config, vlm_baseline):
        """VLM TG speed benchmark"""
        model, tokenizer = load_model(config["model"])

        cache = make_prompt_cache(
            model,
            kv_cache="triple_pq",
            vision_quantizer=config["quant"]
        )

        images = [load_test_image(f"fixture_{i}.jpg")
                  for i in range(config["images"])]

        result = benchmark_vlm_tg_speed(
            model, tokenizer, cache,
            images=images,
            context_len=config["context"],
            num_tokens=100
        )

        # 对比 baseline (无优化)
        key = f"{config['model']}_{config['context']}K_no-opt"
        if key in vlm_baseline:
            baseline_speed = vlm_baseline[key]["tg_speed"]

            # 允许 <50% 退化
            assert result["tg_speed"] > baseline_speed * 0.5, \
                f"TG speed: {result['tg_speed']:.1f} < 50% baseline {baseline_speed:.1f}"

            print(f"{config['model']} {config['quant']}: "
                  f"{result['tg_speed']:.1f} tok/s "
                  f"({result['tg_speed']/baseline_speed*100:.0f}% of baseline)")

    @pytest.mark.parametrize("config", [
        {"model": "Qwen2-VL", "context": 2048, "quant": "q8_0"},
        {"model": "Qwen3-VL", "context": 4096, "quant": "q8_0"},
    ])
    def test_vlm_memory_optimization(self, config, vlm_baseline):
        """VLM 内存优化验证"""
        model, tokenizer = load_model(config["model"])

        cache = make_prompt_cache(
            model,
            kv_cache="triple_pq",
            vision_quantizer=config["quant"]
        )

        with MemoryProfiler() as profiler:
            images = [load_test_image("fixture_0.jpg")]
            prompt = "Describe"

            output = generate(
                model, tokenizer, prompt,
                images=images,
                cache=cache,
                max_tokens=50
            )

        # 对比 baseline
        key = f"{config['model']}_{config['context']}K_no-opt"
        if key in vlm_baseline:
            baseline_mem = vlm_baseline[key]["memory"]

            # 期望内存优化 >30%
            assert profiler.peak_mb < baseline_mem * 0.7, \
                f"Memory not optimized: {profiler.peak_mb}MB vs baseline {baseline_mem}MB"

            reduction = (1 - profiler.peak_mb / baseline_mem) * 100
            print(f"{config['model']} {config['quant']}: "
                  f"{profiler.peak_mb:.0f}MB (-{reduction:.0f}%)")
```

---

### 3.3 质量测试（25 cases）

#### A. VQA Benchmark 测试

```python
# tests/test_vlm_quality.py

class TestVLMQuality:
    """VLM 质量基准测试"""

    @pytest.fixture(scope="class")
    def vqa_dataset(self):
        """加载 VQAV2 验证集（子集）"""
        return load_vqa_dataset("vqav2-validation", max_samples=500)

    @pytest.fixture(scope="class")
    def vqa_baseline(self):
        """VLM 无压缩 baseline"""
        return {
            "Qwen2-VL_vqav2": 0.723,
            "Qwen3-VL_vqav2": 0.756,
        }

    @pytest.mark.parametrize("config", [
        {"model": "Qwen2-VL", "quant": "bf16"},
        {"model": "Qwen2-VL", "quant": "q8_0"},
        {"model": "Qwen2-VL", "quant": "q4_0"},  # 激进
        {"model": "Qwen3-VL", "quant": "q8_0"},
    ])
    def test_vqa_accuracy(self, config, vqa_dataset, vqa_baseline):
        """VQAV2 准确率测试"""
        model, tokenizer = load_model(config["model"])

        cache = make_prompt_cache(
            model,
            kv_cache="triple_pq",
            vision_quantizer=config["quant"]
        )

        # 评估
        correct = 0
        total = len(vqa_dataset)

        for sample in vqa_dataset:
            output = generate(
                model, tokenizer,
                prompt=sample["question"],
                images=[sample["image"]],
                cache=cache,
                max_tokens=20
            )

            if evaluate_vqa_answer(output, sample["answers"]):
                correct += 1

        accuracy = correct / total

        # 对比 baseline
        key = f"{config['model']}_vqav2"
        if key in vqa_baseline:
            baseline_acc = vqa_baseline[key]

            # 质量无损: >95% baseline
            assert accuracy > baseline_acc * 0.95, \
                f"VQA accuracy drop: {accuracy:.3f} < 95% baseline {baseline_acc:.3f}"

            print(f"{config['model']} {config['quant']}: "
                  f"VQA accuracy {accuracy:.3f} "
                  f"({accuracy/baseline_acc*100:.1f}% of baseline)")

    @pytest.mark.parametrize("config", [
        {"model": "Qwen3-VL", "quant": "q8_0", "density_scale": 0.0},
        {"model": "Qwen3-VL", "quant": "q8_0", "density_scale": 1.5},
        {"model": "Qwen3-VL", "quant": "q8_0", "density_scale": 2.5},
    ])
    def test_density_router_impact(self, config, vqa_dataset):
        """验证 density router 对质量的影响"""
        model, tokenizer = load_model(config["model"])

        cache = make_prompt_cache(
            model,
            kv_cache="scored_kv_direct",
            vision_quantizer=config["quant"],
            density_scale_vision=config["density_scale"]
        )

        # 评估子集 (100 samples)
        accuracy = evaluate_vqa_subset(
            model, tokenizer, cache,
            vqa_dataset[:100]
        )

        print(f"Density scale {config['density_scale']}: VQA {accuracy:.3f}")

        # 激进压缩可能导致质量下降
        if config["density_scale"] > 2.0:
            # 允许轻微下降
            assert accuracy > 0.60
        else:
            # 保守压缩应无损
            assert accuracy > 0.70
```

#### B. 多任务质量评估

```python
# tests/test_vlm_multi_task_quality.py

class TestVLMMultiTaskQuality:
    """多任务质量评估"""

    @pytest.mark.parametrize("benchmark", [
        {"name": "GQA", "metric": "accuracy", "baseline": 0.651},
        {"name": "TextVQA", "metric": "accuracy", "baseline": 0.584},
        {"name": "POPE", "metric": "f1", "baseline": 0.872},
    ])
    def test_vlm_multi_task(self, benchmark):
        """多任务 benchmark"""
        model, tokenizer = load_model("Qwen3-VL")

        cache = make_prompt_cache(
            model,
            kv_cache="triple_pq",
            vision_quantizer="q8_0"
        )

        # 加载 benchmark
        dataset = load_benchmark(benchmark["name"], max_samples=200)

        # 评估
        score = evaluate_benchmark(
            model, tokenizer, cache,
            dataset, metric=benchmark["metric"]
        )

        # 对比 baseline
        assert score > benchmark["baseline"] * 0.95, \
            f"{benchmark['name']} {benchmark['metric']}: " \
            f"{score:.3f} < 95% baseline {benchmark['baseline']:.3f}"

        print(f"{benchmark['name']}: {score:.3f}")
```

---

### 3.4 压缩策略对比测试

```python
# tests/test_vlm_compression_strategies.py

class TestVLMCompressionStrategies:
    """对比不同压缩策略"""

    @pytest.fixture(scope="class")
    def test_images(self):
        return [load_test_image(f"fixture_{i}.jpg") for i in range(5)]

    @pytest.mark.parametrize("strategy", [
        {"name": "no-opt", "vision_quant": "bf16", "skip_am": False},
        {"name": "q8-no-am", "vision_quant": "q8_0", "skip_am": True},
        {"name": "q8-with-am", "vision_quant": "q8_0", "skip_am": False},
        {"name": "q4-no-am", "vision_quant": "q4_0", "skip_am": True},
    ])
    def test_compression_strategy_comparison(self, strategy, test_images):
        """对比不同压缩策略的性能和质量"""
        model, tokenizer = load_model("Qwen3-VL")

        cache = make_prompt_cache(
            model,
            kv_cache="triple_pq",
            vision_quantizer=strategy["vision_quant"],
            skip_vision_am_scoring=strategy["skip_am"]
        )

        # 性能测试
        perf_result = benchmark_vlm_tg_speed(
            model, tokenizer, cache,
            images=test_images[:2],
            context_len=4096
        )

        # 质量测试 (简化 VQA)
        vqa_subset = load_vqa_dataset("vqav2-validation", max_samples=50)
        quality_result = evaluate_vqa_subset(
            model, tokenizer, cache, vqa_subset
        )

        print(f"{strategy['name']}:")
        print(f"  TG: {perf_result['tg_speed']:.1f} tok/s")
        print(f"  Memory: {perf_result['memory_mb']:.0f} MB")
        print(f"  VQA: {quality_result:.3f}")

        # 存储结果用于后续分析
        return {
            "strategy": strategy["name"],
            "tg_speed": perf_result["tg_speed"],
            "memory": perf_result["memory_mb"],
            "vqa_acc": quality_result
        }
```

---

## 第四部分：Phase 4 测试计划（生产就绪）

### 4.1 测试目标

**验证**: VLM 生产就绪，所有高级特性工作正常

**测试范围**:
- ✅ Expert offloading (vision weights)
- ✅ Chunked prefill (vision tokens)
- ✅ Model Cards 准确性
- ✅ 长时间稳定性测试
- ✅ 边界条件测试

**成功标准**:
- ✅ 所有功能测试通过
- ✅ 生产质量基准达标
- ✅ 文档和示例可用

---

### 4.2 高级特性测试（15 cases）

#### A. Expert Offloading 测试

```python
# tests/test_expert_offload_vision.py

class TestExpertOffloadVision:
    """Vision weights expert offloading"""

    def test_vision_weights_offload_to_cpu(self):
        """验证 vision weights 可 offload 到 CPU"""
        model, _ = load_model("Qwen3.5-35B-VL-MOE")

        offload_config = OffloadConfig(
            enable_vision_offload=True,
            vision_cpu_cache_mb=2048
        )

        cache = make_prompt_cache(
            model,
            kv_cache="scored_pq",
            offload_config=offload_config
        )

        # 验证 offload 正常工作
        telemetry = offload_config.telemetry

        # 推理后应该有 CPU cache hits
        image = load_test_image("fixture_0.jpg")
        output = generate(model, tokenizer, "Describe", images=[image])

        assert telemetry.vision_cpu_hits > 0
        print(f"Vision CPU cache hits: {telemetry.vision_cpu_hits}")

    def test_vision_offload_performance(self):
        """验证 vision offload 性能影响"""
        model, tokenizer = load_model("Qwen3.5-35B-VL-MOE")

        # Without offload
        cache_no_offload = make_prompt_cache(model, kv_cache="scored_pq")
        result_no_offload = benchmark_vlm_tg_speed(
            model, tokenizer, cache_no_offload,
            images=[load_test_image("fixture_0.jpg")]
        )

        # With offload
        offload_config = OffloadConfig(enable_vision_offload=True)
        cache_offload = make_prompt_cache(
            model, kv_cache="scored_pq",
            offload_config=offload_config
        )
        result_offload = benchmark_vlm_tg_speed(
            model, tokenizer, cache_offload,
            images=[load_test_image("fixture_0.jpg")]
        )

        # Offload 性能损失应该 <20%
        assert result_offload["tg_speed"] > result_no_offload["tg_speed"] * 0.8
```

#### B. Chunked Prefill 测试

```python
# tests/test_vlm_chunked_prefill.py

class TestVLMChunkedPrefill:
    """Vision tokens chunked prefill"""

    def test_chunked_prefill_memory_peak(self):
        """验证 chunked prefill 降低内存峰值"""
        model, tokenizer = load_model("Qwen3-VL")

        # Large context: 3 images + 4K text
        images = [load_test_image(f"fixture_{i}.jpg") for i in range(3)]
        prompt = "A" * 4000

        # Without chunking
        with MemoryProfiler() as profiler_no_chunk:
            generate(
                model, tokenizer, prompt,
                images=images,
                chunk_size=None  # No chunking
            )

        # With chunking
        with MemoryProfiler() as profiler_chunk:
            generate(
                model, tokenizer, prompt,
                images=images,
                chunk_size=512  # Chunked
            )

        # Chunked 应该降低峰值
        assert profiler_chunk.peak_mb < profiler_no_chunk.peak_mb
        reduction = (1 - profiler_chunk.peak_mb / profiler_no_chunk.peak_mb) * 100
        print(f"Chunked prefill memory reduction: {reduction:.0f}%")
```

---

### 4.3 稳定性测试（10 cases）

```python
# tests/test_vlm_stability.py

class TestVLMStability:
    """长时间稳定性测试"""

    @pytest.mark.slow
    def test_long_session_stability(self):
        """验证长时间运行稳定性"""
        model, tokenizer = load_model("Qwen3-VL")
        cache = make_prompt_cache(model, kv_cache="triple_pq")

        # 连续推理 100 次
        for i in range(100):
            image = load_test_image(f"fixture_{i % 5}.jpg")
            prompt = f"Iteration {i}: Describe"

            output = generate(
                model, tokenizer, prompt,
                images=[image],
                cache=cache,
                max_tokens=50
            )

            assert len(output) > 0

            if i % 10 == 0:
                print(f"Iteration {i} completed")

    @pytest.mark.slow
    def test_memory_leak_check(self):
        """验证无内存泄漏"""
        model, tokenizer = load_model("Qwen3-VL")

        memory_snapshots = []

        for i in range(50):
            cache = make_prompt_cache(model, kv_cache="triple_pq")

            image = load_test_image("fixture_0.jpg")
            output = generate(
                model, tokenizer, "Describe",
                images=[image],
                cache=cache
            )

            # 记录内存
            current_mem = get_current_memory_mb()
            memory_snapshots.append(current_mem)

            # 清理
            del cache
            mx.metal.clear_cache()

        # 验证内存不增长
        initial_mem = np.mean(memory_snapshots[:10])
        final_mem = np.mean(memory_snapshots[-10:])

        assert final_mem < initial_mem * 1.1, \
            f"Memory leak detected: {initial_mem:.0f}MB -> {final_mem:.0f}MB"
```

---

### 4.4 边界条件测试（10 cases）

```python
# tests/test_vlm_edge_cases.py

class TestVLMEdgeCases:
    """边界条件测试"""

    def test_no_images_fallback(self):
        """验证无图像时回退到文本模式"""
        model, tokenizer = load_model("Qwen2-VL")

        # VLM 但无图像输入
        output = generate(
            model, tokenizer,
            prompt="Hello world",
            images=None,  # No images
            max_tokens=20
        )

        assert len(output) > 0

    def test_very_large_image(self):
        """验证超大图像处理"""
        model, tokenizer = load_model("Qwen3-VL")

        # 4K 分辨率图像
        large_image = mx.random.normal((3, 2160, 3840))

        output = generate(
            model, tokenizer,
            prompt="Describe",
            images=[large_image],
            max_tokens=50
        )

        assert len(output) > 0

    def test_many_images(self):
        """验证多图像（>10）处理"""
        model, tokenizer = load_model("Qwen3-VL")

        # 15 张图像
        images = [load_test_image(f"fixture_{i % 5}.jpg")
                  for i in range(15)]

        output = generate(
            model, tokenizer,
            prompt="Compare all these images",
            images=images,
            max_tokens=100
        )

        assert len(output) > 0

    def test_vision_token_overflow(self):
        """验证 vision tokens 超过 L0 边界"""
        model, tokenizer = load_model("Qwen3-VL")

        cache = make_prompt_cache(model, kv_cache="triple_pq")

        # 多图像导致 vision tokens > L0 recent_size
        images = [load_test_image(f"fixture_{i}.jpg") for i in range(10)]

        with MemoryProfiler() as profiler:
            output = generate(
                model, tokenizer,
                prompt="Describe all",
                images=images,
                cache=cache,
                max_tokens=50
            )

        # 应该正常完成（溢出到 L1/L2）
        assert len(output) > 0
        print(f"10 images memory: {profiler.peak_mb:.0f}MB")
```

---

## 第五部分：测试基础设施

### 5.1 测试工具和辅助函数

```python
# tests/utils/test_helpers.py

class MemoryProfiler:
    """内存性能分析器"""

    def __enter__(self):
        mx.metal.clear_cache()
        self.start_mb = get_current_memory_mb()
        return self

    def __exit__(self, *args):
        mx.metal.clear_cache()
        self.end_mb = get_current_memory_mb()
        self.peak_mb = max(self.start_mb, self.end_mb)

def benchmark_vlm_tg_speed(model, tokenizer, cache, images, context_len=4096, num_tokens=100):
    """VLM TG speed benchmark"""
    prompt = "A" * context_len

    # Warmup
    _ = generate(model, tokenizer, prompt, images=images, max_tokens=10, cache=cache)

    # Measure
    start = time.time()
    output = generate(model, tokenizer, prompt, images=images, max_tokens=num_tokens, cache=cache)
    elapsed = time.time() - start

    tg_speed = num_tokens / elapsed

    with MemoryProfiler() as profiler:
        _ = generate(model, tokenizer, prompt, images=images, max_tokens=10, cache=cache)

    return {
        "tg_speed": tg_speed,
        "memory_mb": profiler.peak_mb,
        "elapsed_s": elapsed
    }

def evaluate_vqa_answer(predicted, ground_truths):
    """VQA 答案评估"""
    predicted = predicted.lower().strip()
    for gt in ground_truths:
        if predicted == gt.lower().strip():
            return True
        if predicted in gt.lower() or gt.lower() in predicted:
            return True
    return False

def load_test_image(path):
    """加载测试图像"""
    from PIL import Image
    import numpy as np

    img = Image.open(path).convert("RGB")
    img = img.resize((336, 336))
    arr = np.array(img).transpose(2, 0, 1) / 255.0
    return mx.array(arr)
```

---

### 5.2 测试数据管理

```
tests/
├── fixtures/
│   ├── images/
│   │   ├── cat.jpg              # 基础测试图像
│   │   ├── dog.jpg
│   │   ├── bird.jpg
│   │   ├── fixture_0.jpg        # 通用 fixture
│   │   ├── fixture_1.jpg
│   │   └── ...
│   ├── golden_outputs.json      # 金标准输出
│   └── vqa_samples.json         # VQA 测试样本
│
├── datasets/
│   ├── vqav2_validation_500.json    # VQAV2 子集
│   ├── gqa_validation_200.json      # GQA 子集
│   ├── textvqa_validation_200.json  # TextVQA 子集
│   └── pope_validation_200.json     # POPE 子集
│
└── benchmarks/
    ├── phase0_baseline.json     # Phase 1 前基准
    ├── phase2_baseline.json     # Phase 2 基准
    └── phase3_baseline.json     # Phase 3 基准
```

---

### 5.3 CI/CD 完整流程

```yaml
# .github/workflows/vlm_migration_ci.yml

name: VLM Migration CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [phase*/*, develop]

env:
  PYTHON_VERSION: '3.11'
  CACHE_DIR: ~/.cache/flashmlx

jobs:
  # Phase 1: 文本模型回归
  phase1-regression:
    runs-on: self-hosted
    if: startsWith(github.ref, 'refs/heads/phase1')

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-xdist pytest-timeout pytest-html

      - name: Run Phase 1 tests
        run: |
          pytest tests/test_*_regression.py \
                 -n 4 \
                 --timeout=600 \
                 --html=reports/phase1_report.html \
                 --self-contained-html

      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: phase1-test-report
          path: reports/phase1_report.html

  # Phase 2: VLM 功能验证
  phase2-functional:
    runs-on: self-hosted
    if: startsWith(github.ref, 'refs/heads/phase2')

    steps:
      - uses: actions/checkout@v4

      - name: Download test fixtures
        run: |
          wget -q -P tests/fixtures/images/ \
            https://example.com/test_images.tar.gz
          tar -xzf tests/fixtures/images/test_images.tar.gz

      - name: Run Phase 2 tests
        run: |
          pytest tests/test_vision_*.py \
                 tests/test_vlm_*.py \
                 --timeout=1200 \
                 --html=reports/phase2_report.html

      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: phase2-test-report
          path: reports/phase2_report.html

  # Phase 3: VLM 性能和质量
  phase3-benchmark:
    runs-on: self-hosted
    if: startsWith(github.ref, 'refs/heads/phase3')

    steps:
      - uses: actions/checkout@v4

      - name: Download VQA datasets
        run: |
          python scripts/download_vqa_datasets.py \
            --output tests/datasets/

      - name: Run Phase 3 benchmarks
        timeout-minutes: 120
        run: |
          pytest tests/test_vlm_benchmark.py \
                 tests/test_vlm_quality.py \
                 tests/test_vlm_compression_strategies.py \
                 --timeout=7200 \
                 --html=reports/phase3_report.html

      - name: Generate benchmark visualization
        run: |
          python scripts/visualize_benchmarks.py \
            --input test-results/ \
            --output reports/phase3_viz.html

      - name: Upload reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: phase3-reports
          path: reports/

  # Phase 4: 生产就绪
  phase4-production:
    runs-on: self-hosted
    if: startsWith(github.ref, 'refs/heads/phase4')

    steps:
      - uses: actions/checkout@v4

      - name: Run all tests
        timeout-minutes: 180
        run: |
          pytest tests/ \
                 --timeout=10800 \
                 --html=reports/phase4_full_report.html \
                 -v

      - name: Run stability tests
        timeout-minutes: 60
        run: |
          pytest tests/test_vlm_stability.py \
                 -m slow \
                 --timeout=3600

      - name: Generate coverage report
        run: |
          pytest --cov=mlx_lm \
                 --cov-report=html:reports/coverage/

      - name: Upload all reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: phase4-production-reports
          path: reports/
```

---

## 第六部分：测试执行计划

### 6.1 测试时间表

```
Week 1-2 (Phase 1)
├─ Day 1-2: 建立 Phase 0 baseline
│   └─ 运行所有现有 benchmark，记录到 phase0_baseline.json
├─ Day 3-4: Git merge + 单元测试
│   └─ 60 unit tests × 5 min = 5 min (并行)
├─ Day 5-7: 集成测试 + 回归分析
│   └─ 20 integration tests × 20 min
└─ Day 8-10: 修复回归 issue

Week 3-6 (Phase 2)
├─ Week 3: 单元测试开发 + 执行
│   └─ 40 VLM unit tests × 8 min
├─ Week 4: 集成测试开发 + 执行
│   └─ 15 VLM integration tests × 30 min
├─ Week 5-6: 功能验证 + bug 修复
└─ 建立 Phase 2 baseline

Week 7-10 (Phase 3)
├─ Week 7: 性能 benchmark 开发
│   └─ 20 performance tests
├─ Week 8-9: 质量 benchmark 执行
│   └─ 25 quality tests (VQA, GQA, TextVQA)
│   └─ 每个 dataset 200 samples × 2s/sample = ~7 hours
├─ Week 10: 压缩策略对比 + 报告
└─ 建立 Phase 3 baseline

Week 11-12 (Phase 4)
├─ Week 11: 高级特性测试
│   └─ 15 advanced tests + 10 stability tests
├─ Week 12: 边界条件 + 最终验收
│   └─ 10 edge case tests
│   └─ 全量回归测试
└─ 生成最终测试报告
```

---

### 6.2 测试优先级

```
P0 (必须通过，阻塞发布)
├─ Phase 1: 文本模型 TG speed 回归
├─ Phase 2: VLM 推理完成率 100%
├─ Phase 3: VQA 质量 >95% baseline
└─ Phase 4: 稳定性测试无 crash

P1 (重要，可接受小幅退化)
├─ Phase 1: 文本模型 Memory 回归
├─ Phase 2: VLM L0 边界调整
├─ Phase 3: VLM TG speed <50% 退化
└─ Phase 4: Expert offload 正常工作

P2 (Nice-to-have)
├─ Phase 3: 多任务 benchmark (GQA, TextVQA)
├─ Phase 4: Chunked prefill 内存优化
└─ Phase 4: 边界条件全覆盖
```

---

## 第七部分：测试报告模板

### 7.1 每日测试报告

```markdown
# FlashMLX VLM Migration - Daily Test Report

**Date**: 2026-04-XX
**Phase**: Phase X
**Tester**: @username

## Test Execution Summary

| Category | Total | Passed | Failed | Skipped | Pass Rate |
|----------|-------|--------|--------|---------|-----------|
| Unit Tests | 60 | 58 | 2 | 0 | 96.7% |
| Integration | 20 | 18 | 2 | 0 | 90.0% |
| E2E | 8 | 7 | 1 | 0 | 87.5% |
| **Total** | **88** | **83** | **5** | **0** | **94.3%** |

## Failed Tests

### 1. test_triple_pq_bug_fix_preserved
- **Category**: Unit Test
- **Failure**: AssertionError: lazy_prefill_threshold != 32768
- **Root Cause**: Git merge conflict 未正确解决
- **Action**: 修复 cache_factory.py line 347
- **ETA**: 2 hours

### 2. test_tg_speed_regression[Qwen3-8B_32K]
- **Category**: Integration Test
- **Failure**: TG speed 18.5 tok/s < baseline 21.6 tok/s (14% regression)
- **Root Cause**: 未知，需 profiling
- **Action**: 使用 Instruments 分析
- **ETA**: 1 day

## Performance Metrics

| Model | Context | Baseline TG | Current TG | Δ |
|-------|---------|-------------|------------|---|
| Mistral-7B | 4K | 24.7 | 24.9 | +0.8% ✅ |
| Qwen3-8B | 8K | 18.2 | 18.0 | -1.1% ✅ |
| Qwen3-8B | 32K | 21.6 | 18.5 | -14.4% ❌ |
| Llama-3.2-3B | 16K | 52.3 | 53.1 | +1.5% ✅ |

## Next Steps
1. 修复 Git merge conflict (2h)
2. Profiling Qwen3-8B 32K regression (1d)
3. 继续 Phase 2 单元测试开发

---
**Report Generated**: 2026-04-XX 18:00 UTC
```

---

### 7.2 阶段总结报告

```markdown
# FlashMLX VLM Migration - Phase X Summary Report

**Phase**: Phase X
**Duration**: Week X - Week Y
**Status**: ✅ PASSED / ⚠️ PARTIAL / ❌ FAILED

## Objectives vs Achievements

| Objective | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Text model regression | ±5% TG speed | ±2.3% avg | ✅ |
| Memory footprint | ±10% | +7.1% avg | ✅ |
| Test coverage | 100% existing | 100% | ✅ |
| CI integration | Full automation | 95% automated | ✅ |

## Test Results Summary

**Total Tests**: 177
**Passed**: 170 (96.0%)
**Failed**: 5 (2.8%)
**Skipped**: 2 (1.1%)

### Failed Tests Breakdown
1. test_xxx - P1 - Fixed
2. test_yyy - P2 - Workaround applied
3. test_zzz - P0 - **BLOCKER** (需解决)

## Key Findings

### ✅ Successes
- Text model 性能无明显退化
- CI pipeline 自动化完成
- Baseline 数据建立完整

### ⚠️ Concerns
- Qwen3-8B 32K context 性能退化 14%
- 部分 edge cases 未覆盖

### ❌ Blockers
- test_zzz 失败阻塞 Phase 2 开始

## Performance Comparison

[插入性能对比图表]

## Recommendations
1. 优先修复 test_zzz (P0, 2 days)
2. 调查 Qwen3-8B 32K regression (P1, 3 days)
3. 补充 edge case coverage (P2, 1 week)

## Phase Transition Readiness

**Ready for Phase 2?** ✅ YES / ❌ NO
**Conditions**:
- ✅ All P0 tests passed
- ✅ Baseline established
- ⚠️ 1 P1 issue outstanding (non-blocking)

---
**Report Date**: 2026-04-XX
**Approved By**: @tech-lead
```

---

## 总结

### 测试计划概览

```
总测试用例数: 177
├─ Phase 1: 80 cases (文本回归)
│   ├─ Unit: 60
│   ├─ Integration: 20
│   └─ E2E: 0
│
├─ Phase 2: 55 cases (VLM 功能)
│   ├─ Unit: 40
│   ├─ Integration: 15
│   └─ E2E: 0
│
├─ Phase 3: 45 cases (VLM 优化)
│   ├─ Performance: 20
│   ├─ Quality: 25
│   └─ E2E: 0
│
└─ Phase 4: 35 cases (生产就绪)
    ├─ Advanced: 15
    ├─ Stability: 10
    ├─ Edge Cases: 10
    └─ E2E: 0

总执行时间: ~70 min per full run
├─ Unit: 5 min (并行)
├─ Integration: 20 min
├─ Performance: 25 min
└─ Quality: 20 min
```

### 关键成功指标

| Phase | 关键指标 | 目标 | 验收标准 |
|-------|---------|------|----------|
| **Phase 1** | TG speed regression | ±5% | 100% models pass |
| **Phase 2** | VLM inference completion | 100% | All 5 VLMs work |
| **Phase 3** | VQA quality | >95% baseline | VQAV2 + GQA |
| **Phase 4** | Production readiness | All P0/P1 pass | Stability + docs |

---

**创建日期**: 2026-04-09
**版本**: v1.0
**维护者**: 测试团队
