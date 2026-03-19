# FlashMLX 源码结构

**日期**: 2026-03-18
**状态**: ✅ 完整源码已就位

---

## 📦 完整源码树

```
~/FlashMLX/
├── mlx-source/              # MLX 核心库源码 (0.31.2)
│   ├── mlx/
│   │   ├── backend/
│   │   │   └── metal/       # Metal GPU 后端 ⭐
│   │   │       ├── kernels/ # Metal kernel 源码 (26个 .metal 文件)
│   │   │       ├── matmul.cpp (82KB)
│   │   │       └── ...
│   │   └── ops/             # 算子实现
│   ├── python/              # Python bindings
│   └── CMakeLists.txt       # 构建配置
│
└── mlx-lm-source/           # MLX-LM 源码 (commit 4a21ffd)
    ├── mlx_lm/
    │   ├── models/          # 模型实现 (Llama, Mistral, Qwen, etc.)
    │   ├── generate.py      # 生成逻辑 (54KB)
    │   ├── evaluate.py      # 评估
    │   ├── lora.py          # LoRA 支持
    │   └── gguf.py          # GGUF 格式
    └── setup.py
```

---

## 🔍 架构层级

```
┌─────────────────────────────────────────────┐
│  MLX-LM - 语言模型推理层                    │
│  ~/FlashMLX/mlx-lm-source/                  │
│  - Model Implementations                    │
│  - BatchGenerator                           │
│  - GGUF Support                             │
└─────────────────────────────────────────────┘
                    ↓ 依赖
┌─────────────────────────────────────────────┐
│  MLX - 核心框架                             │
│  ~/FlashMLX/mlx-source/                     │
│  - Metal GPU Backend                        │
│  - Array Operations                         │
│  - C++ Core + Metal Kernels                │
└─────────────────────────────────────────────┘
```

---

## 🔥 关键源码位置

### MLX Metal Kernels

| Kernel | 文件 | 大小 | 功能 |
|--------|------|------|------|
| **GEMV** | `mlx-source/mlx/backend/metal/kernels/gemv.metal` | 31KB | 矩阵-向量乘法 ⭐⭐⭐ |
| **Flash Attention** | `mlx-source/mlx/backend/metal/kernels/scaled_dot_product_attention.metal` | 1.9KB | Attention 优化 ⭐⭐⭐ |
| **Quantized** | `mlx-source/mlx/backend/metal/kernels/quantized.metal` | - | 量化 kernels ⭐⭐ |
| **RMSNorm** | `mlx-source/mlx/backend/metal/kernels/rms_norm.metal` | - | RMS 归一化 ⭐ |
| **RoPE** | `mlx-source/mlx/backend/metal/kernels/rope.metal` | - | 旋转位置编码 ⭐ |

**总计**: 26 个 `.metal` 文件，4,450 行代码

### MLX C++ 实现

| 文件 | 大小 | 功能 |
|------|------|------|
| `mlx-source/mlx/backend/metal/matmul.cpp` | 82KB | MatMul 调度 ⭐⭐⭐ |
| `mlx-source/mlx/backend/metal/device.cpp` | 25KB | Metal 设备管理 |
| `mlx-source/mlx/backend/metal/jit_kernels.cpp` | 37KB | JIT 编译 |

### MLX-LM 模型

```
mlx-lm-source/mlx_lm/models/
├── llama.py          # Llama 系列
├── mistral.py        # Mistral 系列
├── qwen2.py          # Qwen 系列
├── gemma.py          # Gemma 系列
└── ...               (总计 100+ 模型实现)
```

---

## 🛠️ 构建指南

### 1. 编译 MLX 核心库

```bash
cd ~/FlashMLX/mlx-source
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

**编译产物**:
- `libmlx.dylib` - C++ 动态库
- `mlx.metallib` - Metal kernels 库

### 2. 安装 MLX-LM (纯 Python，无需编译)

```bash
cd ~/FlashMLX/mlx-lm-source
pip install -e .
```

---

## 📊 版本信息

| 组件 | 版本 | 来源 |
|------|------|------|
| **MLX** | 0.31.2 | `mlx-source/mlx/version.h` |
| **MLX-LM** | commit 4a21ffd | `mlx-lm-source/.git` |

---

## 🔍 快速导航

### 查看 Flash Attention 实现
```bash
cat ~/FlashMLX/mlx-source/mlx/backend/metal/kernels/scaled_dot_product_attention.metal
```

### 查看量化 GEMV 实现
```bash
cat ~/FlashMLX/mlx-source/mlx/backend/metal/kernels/gemv.metal
```

### 查看 Qwen 模型实现
```bash
cat ~/FlashMLX/mlx-lm-source/mlx_lm/models/qwen2.py
```

### 查看 MatMul 调度
```bash
cat ~/FlashMLX/mlx-source/mlx/backend/metal/matmul.cpp
```

---

## 📚 文档

- **MLX 文档**: https://ml-explore.github.io/mlx/
- **MLX-LM 文档**: https://github.com/ml-explore/mlx-lm

---

*源码结构文档生成于: 2026-03-18*
*状态: 完整源码已就位，可以开始开发/调试*
