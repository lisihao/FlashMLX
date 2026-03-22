# 模型路径迁移记录

**日期**: 2026-03-21 16:30
**操作**: 将所有模型路径从 mlx-community 更新到 Toshiba 外置硬盘

---

## 路径映射

| 原路径 | 新路径 |
|--------|--------|
| `mlx-community/Qwen3.5-35B-A3B-6bit` | `/Volumes/toshiba/models/qwen3.5-35b-mlx` |
| `mlx-community/Qwen3-8B` | `/Volumes/toshiba/models/qwen3-8b-mlx` |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | `/Volumes/toshiba/models/llama-3.2-3b-mlx` |
| `qwen3.5-2b-opus-distilled` | `/Volumes/toshiba/models/qwen3.5-2b-opus-distilled` |
| `qwen3.5-0.8b-opus-distilled` | `/Volumes/toshiba/models/qwen3.5-0.8b-opus-distilled` |

---

## 已更新的文件 (16个)

### 性能测试脚本
1. `benchmarks/detailed_profiling.py` - 详细性能分析
2. `benchmarks/profile_model_forward.py` - 模型前向传播分析
3. `benchmarks/profile_with_metal_capture.py` - Metal GPU 性能分析
4. `benchmarks/profile_correct_decode.py` - 正确解码性能分析
5. `benchmarks/profile_mlx_lm_generate.py` - MLX-LM 生成性能分析
6. `benchmarks/profile_simple_gpu_time.py` - 简单 GPU 时间测量
7. `benchmarks/profile_with_cprofile.py` - Python cProfile 分析
8. `benchmarks/baseline_benchmark_simple.py` - 基准测试

### 压缩相关测试
9. `benchmarks/quick_ssm_test.py` - SSM 快速测试
10. `benchmarks/ssm_compression_test.py` - SSM 压缩测试
11. `benchmarks/ssm_methods_comparison.py` - SSM 压缩方法对比
12. `benchmarks/test_cached_lowrank.py` - 缓存 Low-Rank 测试
13. `benchmarks/test_rank_tradeoff.py` - Rank 权衡测试
14. `benchmarks/layerwise_ablation.py` - 层级消融实验

### 混合架构测试
15. `benchmarks/hetero_cache_test.py` - 异构缓存测试
16. `benchmarks/hetero_cache_quality_test.py` - 异构缓存质量测试

---

## 未更新的文件 (14个)

这些文件不包含模型路径或使用的是其他模型：

- `accurate_benchmark.py`
- `simple_benchmark.py`
- `qwen3_test.py`
- `output_quality_test.py`
- `correct_usage_test.py`
- `compacted_cache_benchmark.py`
- `llama_test.py`
- `simple_timing_test.py`
- `qwen3_5_hybrid_test.py`
- 等等

---

## Toshiba 盘模型目录结构

```
/Volumes/toshiba/models/
├── llama-3.2-3b-mlx/              # Llama 3.2 3B MLX
├── qwen3-8b-mlx/                  # Qwen3 8B MLX
├── qwen3.5-0.8b-opus-distilled/   # Qwen3.5 0.8B Opus 蒸馏
├── qwen3.5-2b-opus-distilled/     # Qwen3.5 2B Opus 蒸馏
├── qwen3.5-35b-a3b-gguf/          # Qwen3.5 35B A3B GGUF
└── qwen3.5-35b-mlx/               # Qwen3.5 35B MLX (主要使用)
```

---

## 验证步骤

### 1. 检查路径更新
```bash
grep -r "Volumes/toshiba/models" benchmarks/
```

### 2. 运行快速测试
```bash
python3 benchmarks/quick_ssm_test.py
```

### 3. 验证模型加载
```bash
python3 -c "from mlx_lm import load; model, tokenizer = load('/Volumes/toshiba/models/qwen3.5-35b-mlx'); print('✅ Model loaded successfully')"
```

---

## 注意事项

1. **外置硬盘依赖**: 所有测试脚本现在依赖 Toshiba 盘挂载在 `/Volumes/toshiba`
2. **挂载检查**: 运行脚本前确保 Toshiba 盘已挂载
3. **路径一致性**: 所有新脚本都应使用 `/Volumes/toshiba/models/` 作为基础路径
4. **备份**: 原始 mlx-community 路径仍可用作备份引用

---

## 迁移工具

已创建 `update_model_paths.py` 批量更新工具：

```bash
python3 update_model_paths.py
```

功能：
- 自动扫描 benchmarks/ 下所有 .py 文件
- 批量替换模型路径
- 生成更新报告

---

*迁移完成于: 2026-03-21 16:30*
*操作者: Solar*
*更新文件数: 16*
