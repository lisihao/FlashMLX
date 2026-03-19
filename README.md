# FlashMLX

**高性能 MLX 推理引擎 - 专注于 Flash Attention 和 Metal GPU 优化**

---

## 🎯 项目定位

FlashMLX 是基于 [MLX](https://github.com/ml-explore/mlx) 的性能增强层，专注于：

1. **Flash Attention 优化** - 优化注意力机制计算
2. **Metal GPU 加速** - 深度优化 Metal kernels
3. **量化推理优化** - 提升量化模型性能
4. **Apple Silicon 原生** - 充分利用 M 系列芯片

## 📦 项目结构

```
FlashMLX/
├── mlx-source/              # MLX 核心库源码 (0.31.2)
├── mlx-lm-source/           # MLX-LM 源码 (commit 4a21ffd)
├── src/                     # FlashMLX 优化层
├── tests/                   # 测试
├── benchmarks/              # 性能测试
└── docs/                    # 文档
```

## 🔥 核心优化方向

### 1. Flash Attention
- 优化 `scaled_dot_product_attention.metal`
- 减少内存带宽消耗
- 提升长序列性能

### 2. GEMV (矩阵-向量乘法)
- 优化 `gemv.metal` (31KB)
- 针对 MoE 模型优化
- 量化 GEMV 加速

### 3. MatMul 调度
- 优化 `matmul.cpp` (82KB)
- 改进 Metal 资源分配
- 减少 kernel 启动开销

## 🚀 快速开始

### 1. 编译 MLX 核心库

```bash
cd mlx-source
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

### 2. 安装 MLX-LM

```bash
cd mlx-lm-source
pip install -e .
```

### 3. 运行 Benchmark

```bash
cd benchmarks
python benchmark_mlx.py
```

## 📊 性能目标

| 指标 | MLX 基线 | FlashMLX 目标 |
|------|---------|--------------|
| Flash Attention | - | +15% TG |
| GEMV (量化) | - | +20% TG |
| 长序列 (8K+) | - | +25% TG |

## 🛠️ 技术栈

- **MLX**: 0.31.2
- **MLX-LM**: commit 4a21ffd
- **Metal**: Shading Language 3.1
- **Python**: 3.10+

## 📚 文档

- [源码结构](./SOURCE_CODE_STRUCTURE.md)
- [状态管理](./.solar/STATE.md)
- [MLX 官方文档](https://ml-explore.github.io/mlx/)

## 🎓 参考资料

- **MLX**: https://github.com/ml-explore/mlx
- **MLX-LM**: https://github.com/ml-explore/mlx-lm
- **Flash Attention 2**: https://arxiv.org/abs/2307.08691

## 📝 License

基于 MLX 的 MIT License

---

*FlashMLX - Blazing Fast MLX Inference on Apple Silicon* 🚀
