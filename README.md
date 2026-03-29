FlashMLX：三条路线吃掉 LLM 推理的每一块内存
在 Apple Silicon 上，三条独立优化路线同时出击：
参数内存：Expert Offloading 把 18 GB MoE 参数压到 10 GB，TG 速度不掉
PP 内存：Chunked Prefill + Streaming Eviction 把 O(N²) 打到 O(1)，32K PP 只用 774 MB
TG 内存：Scored P2 + Q8 Flat Buffer 把 4.6 GB KV Cache 压到 147 MB，TG 反而快 54%
这不是一个顺利的故事。每条路线都踩了坑，有些坑是论文挖的，有些是自己跳的。

先搞清楚：内存都花在哪了？
大模型推理的 GPU 内存开销分三大块，很多人只看到其中一块就以为搞定了：

┌─────────────────────────────────────────────────────────────────────┐
│                   LLM 推理内存三大块                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 参数内存 (Model Parameters)                                    │
│     模型权重常驻 GPU。密集模型 = 参数量 × 精度。                    │
│     MoE 模型 = 参数量巨大，但只有 top-k 专家被激活。               │
│     Qwen3.5-35B-A3B: 256 专家/层 × 40 层 = 18.21 GB              │
│                                                                     │
│  2. PP 内存 (Prefill / Prompt Processing)                          │
│     处理输入时的 KV Cache + 中间激活值。                            │
│     标准 Attention 是 O(N²)，32K 上下文 = 5 GB+ 峰值。            │
│     这是"首次输入"的内存高峰。                                      │
│                                                                     │
│  3. TG 内存 (Token Generation / Decode)                            │
│     逐 token 生成时的 KV Cache。                                   │
│     每个新 token 读全部历史 KV = 随对话长度线性增长。              │
│     32K 上下文 = 4.6 GB KV Cache，且拖慢 TG 速度。               │
│                                                                     │
│  总开销 = 参数 + max(PP 峰值, TG 累积)                             │
│  三块都得优化，只优化一块效果有限。                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
下面分别讲三条路线怎么把每一块吃下来的。

路线一：参数内存 — Expert Offloading Compact Pool
问题
MoE 模型的参数内存很大。Qwen3.5-35B-A3B 每层 256 个专家，40 层，Q4 量化后仍需 18.21 GB 常驻 GPU。

但一个关键事实是：推理时每层只激活 top-8 个专家。256 个里只用 8 个，其余 248 个纯占内存。

MLX-LM 社区有人做了 flash-moe 架构，把不活跃的专家卸载到 CPU。我在此基础上加了一个关键优化：两阶段 Compact Pool。

架构
PP 阶段: 全部 256 专家留在 GPU
         → identity 路径，零开销
         → 同时记录每个专家的激活次数

Compact: 统计 PP 中的"热门"专家 → 选 top-K
         → 非热门专家降级到 CPU cache (numpy, UMA 快速搬运)
         → mx.eval(compact pool) + gc.collect → 释放旧 pool
         → 预热 Metal kernel (新 pool shape 需要 JIT 编译)

TG 阶段: 只用 K 个热门专家 → remap + clamp 索引
         → 和 identity 路径走同一条 gather_qmm 代码
         → 零 .item()，零 GPU→CPU 同步，全 lazy evaluation
         → 极少数 miss → clamped 到最近的热门专家 (1/8 权重误差，可忽略)
踩坑记录
坑 1：Sentinel 检查杀死性能

最初的 miss 检测方案：每个 MoE 层用 mx.max(local_indices).item() 检查是否有索引越界。

结果：5.6 tok/s（vs identity 的 90+ tok/s）。

原因：.item() 强制 GPU→CPU 同步。40 个 MoE 层 × 每个 token = 40 次同步。MLX 的 lazy evaluation 完全失效——本来应该攒一大坨计算一起 flush 的 graph，被 .item() 切成 40 段串行执行。

修复：Speculative Execution

不检查，直接 clamp：

# 旧方案：每层检查一次（5.6 tok/s）
max_idx = mx.max(local_indices).item()  # GPU→CPU sync!
if max_idx < K:
    fast_path()
else:
    miss_path()

# 新方案：零检查（92.8 tok/s）
# remap 表的默认值设为 K-1（最后一个有效专家）
# 非 pool 专家自动 clamp 到有效范围，无需 mx.minimum
local_indices = self._pool_remap[indices]  # 一条 MLX op，全 lazy
为什么可以不检查？因为 compact 时选的是 PP 阶段激活次数最多的 top-K 专家。TG 阶段的路由高度集中在同一批专家上。coverage=100% 意味着所有 PP 期间出现过的专家都在 pool 里，TG 时几乎不会 miss。

坑 2：Metal Kernel JIT 的 Warmup

compact 后前 50 个 token 只有 ~40 tok/s，之后突然跳到 90+ tok/s。

原因不是 compact 本身慢，而是 Apple Metal 需要为新的 pool tensor shape JIT 编译 GPU kernel。编译完后就是全速。

验证：FORCE_REMAP=1 强制 pool=256 走 remap 路径 → 90.6 tok/s，和 identity 完全一致。remap 零开销，慢的只是 JIT。

最终结果
Qwen3.5-35B-A3B (Q4, 256 experts/layer, top-8), Apple M4 Pro 48GB：

Config	Steady TG	Memory	Saved	Coverage
pool=256 (identity)	90.0 tok/s	18.21 GB	—	100%
pool=192 (compact)	90.9 tok/s	13.99 GB	4.23 GB	100%
pool=128 (compact)	92.8 tok/s	9.77 GB	8.44 GB	100%
TG 速度零惩罚（warmup 后与 identity 一致），参数内存 减少 46%。

pool=128 甚至比 pool=256 快 3%——因为更小的 pool tensor = 更紧凑的内存布局 = 更好的 cache locality。

路线二：PP 内存 — Chunked Prefill + Streaming Eviction
问题
标准 Prefill 是 O(N²)。处理 32K token 的 prompt，Attention 要算 32K × 32K 的矩阵，KV Cache + 中间激活 = 5 GB+ 峰值。

这意味着：即使你的模型只占 8 GB，一个长 prompt 就再吃 5 GB，可能直接 OOM。

核心洞察
如果我们在 Prefill 阶段也做 eviction 呢？

不是等 prefill 结束后再压缩，而是边 prefill 边评估 token 重要性，边驱逐不重要的 token：

Chunk 1 (token 0-511):     prefill → cache: 512 tokens
Chunk 2 (token 512-1023):  prefill → cache: 1024 tokens
...
Chunk 9 (token 4096-4607): prefill → cache: 4608 tokens
                            ↓ cache > max_cache (4096)
                            AM eviction: 4608 → 1872 tokens
                            ↓ continue
Chunk 10 (token 4608-5119): prefill → cache: 2384 tokens
...
每次 cache 超过 4096 token 阈值，用 AM importance scoring 驱逐低分 token，保留 hot tokens + 最近 512 tokens。

为什么反而更快？
标准 attention 是 O(N²)。32K token 意味着 attention 矩阵是 32K × 32K = 10 亿次运算。

Chunked Prefill 把它变成 O(chunk × cache) = O(512 × 2048) ≈ 100 万次运算/chunk。复杂度从 O(N²) 变成 O(1)。

32K 时效果最明显：standard PP 被 O(N²) 拖到 213 tok/s，Scored Chunked 维持 369 tok/s (+73%)。

结果
Qwen3-8B-MLX (Q8), Apple M4 Pro 24GB：

指标	Standard	Scored Chunked	32K 变化
PP 速度	213.6 tok/s	369.1 tok/s	+73.0%
PP 峰值内存	5,079 MB	774 MB	-84.8%
PP Active	14,207 MB	526 MB	-96.3%
PP 峰值 = 774 MB，无论 16K 还是 32K 还是 128K。O(1) 内存复杂度。

这个数字不是巧合。max_cache=2048 + chunk_size=512 限制了物理 cache 上限。理论上你可以 prefill 128K token，PP 内存仍然在 ~800 MB 量级。

路线三：TG 内存 — Scored P2 + Q8 Flat Buffer
问题
TG 阶段的 KV Cache 随对话长度线性增长。32K 上下文 = 4.6 GB KV Cache。不仅吃内存，还拖慢 TG 速度（每个新 token 要读全部历史 KV）。

演化路线
这条路线走了最多弯路，也是教训最多的一条。

Round 1：信 AM，得永生？
Attention Matching——数学很优雅：最小化压缩前后 attention 分布的 KL 散度，用 β 系数补偿被丢弃的 Key。单层测试，完美：

Ratio 2.0x → 50% 压缩 → QA 100% ✅
Ratio 3.0x → 67% 压缩 → QA 100% ✅
36 层全上：QA 0.000，输出乱码。

教训 #1：单层压缩和全层压缩是两个完全不同的问题。 误差在 36 层 transformer 中逐层累积，到最后层已经不可控。

Round 2：差点放弃，然后找到了 Bug
准备转向 H2O 时，重新审视实现，发现两个关键 bug：

β solver 无界：论文只给 β = R_S† · target，没提需要 bound。β 飞到 [-171, 221]，毫无物理意义。加 bounded optimization 后修复。
Query 采样偏差：连续 10 个 query 偏向某个上下文窗口。改为均匀采样 594 个位置后修复。
教训 #2：不要过早放弃。修完 bug 后看到真实天花板在哪，再做决策。

Round 3：On-Policy 校准——论文没说的事
Bug 修完后，18 层 OK，36 层还是崩。加 45% 的数据，一层都没多压成功。

问题不在数据量，在数据分布。第 18 层以后看到的 KV 已经被前 17 层压缩过了——用原始分布的数据去校准被压缩分布上的行为，等于用城市路况数据训练越野自驾模型。

解法：分阶段 on-policy 校准。结果：36⁄36 层全部压缩，87.5% QA 准确率（与 baseline 持平）。

教训 #3：这是我读过的所有 KV Cache 压缩论文中都没有被充分讨论的一点。

Round 4：Scored P2 + Flat Buffer
最终架构抛弃了复杂的三层缓存（L0 Recent + L1 Quantized Warm + L2 AM Cold），简化为一次性 Promotion：

PP 阶段:    全 bf16 Recent buffer，不压缩
Promotion:  PP→TG 转换时一次性 AM 评分 → hot token 进 flat buffer / cold 丢弃
TG 阶段:    flat buffer(Q8) → O(1) per token append
为什么简化反而更好？因为 Pipeline 架构（PQ4+AM）的 PP 内存反而翻倍——Attention 必须收 bf16 数据，但 Pipeline 同时持有量化存储和 dequant 结果 = 双份内存。Scored P2 在 PP 阶段不做量化，PP 内存 = standard。

Round 5：Q8_0 Flat Buffer——几乎免费的 50%
flat buffer 本身还能压。per-token absmax int8 + bf16 scale，一条 Metal 指令的 dequant：

量化	TG 内存 (32K)	TG 速度	速度代价
bf16 (无量化)	288 MB	26.2 tok/s	—
Q8_0	147 MB	24.7 tok/s	-6%
Q4_0	81 MB	16.1 tok/s	-39%
Q8_0 是甜蜜点：6% 速度换 49% 内存。Q4_0 的 nibble unpack 是 compute-bound，KV 只占 TG 总带宽 6%，省带宽没意义但多出来的计算是实打实的。

教训 #4：KV Cache 量化的收益分析不能只看 cache 本身。TG 阶段 94% 的带宽在模型参数上。

混合架构上的 AM：完全失败
这是一个重要的负面结果。在 Qwen3.5（30 SSM + 10 Attention）上：

Compression ratio 2.0x → 乱码
Compression ratio 3.0x → 乱码
Compression ratio 5.0x → 乱码（和 2.0x 的乱码一模一样）
只压缩 10⁄40 层（25%）就完全崩溃。SSM 层放大了 Attention 层的压缩误差。这不是一个可以通过调参解决的问题——是架构级别的不兼容。

教训 #5：AM 不是通用记忆压缩器。即使一层是 softmax attention，也可能因架构交互而失效。混合架构的层间交互比单层特性更重要。

这个发现也解释了为什么我们在参数内存优化（Expert Offloading）上走了完全不同的路——不压缩，而是卸载。

最终结果
Qwen3-8B-MLX (Q8), Apple M4 Pro 24GB，Scored Q8 (推荐配置)：

指标	Standard	Scored Q8	16K 变化	32K 变化
TG 速度	18.9 / 16.0 tok/s	24.7 / 24.7 tok/s	+31%	+54%
KV TG 内存	2,268 / 4,572 MB	129 / 147 MB	-94%	-97%
质量	PASS	PASS	无损	无损
32K 上下文：TG 快 54%，KV 省 97%，质量无损。

三线汇合：完整架构
FlashMLX v1.0 — 三维内存优化
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ┌─── 路线一：参数内存 ───────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │  MoE Expert Offloading + Compact Pool                             │  │
│  │  Full Pool (256 experts) ──→ PP (zero overhead)                   │  │
│  │  Compact to hot-K ──→ CPU cache (non-hot experts)                 │  │
│  │  TG: remap + clamp, zero .item(), full lazy eval                  │  │
│  │                                                                    │  │
│  │  Model: Qwen3.5-35B-A3B (Q4)                                     │  │
│  │  Result: 18.21 GB → 9.77 GB (-46%), TG: 92.8 tok/s (zero loss)   │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─── 路线二：PP 内存 ───────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │  Chunked Prefill + Streaming AM Eviction                          │  │
│  │  chunk=512, max_cache=2048                                        │  │
│  │  if cache > threshold: AM scoring → evict cold tokens             │  │
│  │  O(chunk × cache) attention per chunk → O(1) total memory         │  │
│  │                                                                    │  │
│  │  Model: Qwen3-8B-MLX (Q8)                                        │  │
│  │  Result: PP peak 5,079 → 774 MB (-85%), PP speed +73% @ 32K      │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─── 路线三：TG 内存 ──────────────────────────────────────────────┐  │
│  │                                                                    │  │
│  │  Scored P2 + Q8_0 Flat Buffer Quantization                        │  │
│  │  PP: bf16 recent buffer (no quant overhead)                       │  │
│  │  Promotion: AM scoring → hot tokens → Q8 flat buffer              │  │
│  │  TG: int8 flat buffer, O(1) per token append                     │  │
│  │                                                                    │  │
│  │  Model: Qwen3-8B-MLX (Q8)                                        │  │
│  │  Result: KV 4,572 → 147 MB (-97%), TG speed +54% @ 32K           │  │
│  │                                                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─── 支撑系统 ─────────────────────────────────────────────────────┐  │
│  │  Auto-Calibration: 首次 ~26s, 缓存后 <1ms                       │  │
│  │  Pluggable Quantizers: Q4_0, Q8_0, PolarQuant, TurboQuant        │  │
│  │  On-Policy Calibration: 分阶段校准解决深层网络误差累积           │  │
│  │  UMA-Aware: Apple Silicon 统一内存 = numpy→mx 6μs memcpy         │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Platform: Apple M4 Pro 48GB                                             │
└──────────────────────────────────────────────────────────────────────────┘
总成绩单
参数内存 — Qwen3.5-35B-A3B (Q4, 256 experts/layer, MoE)
Config	TG (steady)	参数内存	Saved
No offload	90.0 tok/s	18.21 GB	—
Compact pool=192	90.9 tok/s	13.99 GB	-4.23 GB (-23%)
Compact pool=128	92.8 tok/s	9.77 GB	-8.44 GB (-46%)
PP 内存 — Qwen3-8B-MLX (Q8, Dense Transformer)
Config	PP Speed	PP Peak Memory	Change
Standard 16K	330.6 tok/s	8,399 MB	—
Standard 32K	264.6 tok/s	16,990 MB	—
Scored Chunked 16K	367.5 tok/s (+11%)	1,131 MB	-87%
Scored Chunked 32K	369.1 tok/s (+39%)	1,131 MB	-93%
TG 内存 — Qwen3-8B-MLX (Q8, Dense Transformer)
Config	TG Speed	KV TG Memory	Change
Standard 16K	18.9 tok/s	2,268 MB	—
Standard 32K	16.0 tok/s	4,572 MB	—
Scored Q8 16K	24.7 tok/s (+31%)	129 MB	-94%
Scored Q8 32K	24.7 tok/s (+54%)	147 MB	-97%
组合效果（PP + TG，Qwen3-8B, 32K 上下文）
阶段	Standard	FlashMLX	变化
PP 速度	213.6 tok/s	372.8 tok/s	+74.5%
PP 峰值内存	5,079 MB	774 MB	-84.8%
TG 速度	16.0 tok/s	24.7 tok/s	+54.4%
TG KV 内存	4,572 MB	147 MB	-96.8%
TTOF	151.7s	86.9s	-42.7%
质量	PASS	PASS	无损
技术创新总结
创新点	路线	类型	解决的问题
Two-Phase Compact Pool	参数	架构创新	MoE 参数内存 46% 节省，零 TG 惩罚
Speculative Execution (clamp, no sentinel)	参数	算法创新	消除 MoE 层的 GPU→CPU 同步瓶颈
UMA-Aware CPU Cache	参数	工程创新	Apple Silicon 统一内存的 numpy→mx 快速搬运
Chunked Prefill + Streaming Eviction	PP	架构创新	O(N²) → O(1) PP 内存
On-Policy 分阶段校准	TG	算法创新	深层网络的误差累积
Bounded β Optimization	TG	算法修复	AM 论文的隐含假设
Scored P2 一次性 Promotion	TG	架构创新	避免 Pipeline 的 PP 内存翻倍
Q8_0 Flat Buffer Quantization	TG	工程创新	Flat buffer 内存减半，6% 速度代价
可插拔量化策略	TG	架构创新	Q4_0 / Q8_0 / PolarQuant 统一接口
自动校准系统	TG	工程创新	新模型零配置使用
复盘：不太光彩但很有用的教训
论文复现的生存指南
先在最简单的设置上复现。单层、短序列、小模型。如果这都不 work，不用往下走了。
论文的 bound 和 constraint 往往不写在正文里。β 需要 bounded optimization 这件事，正文、附录、代码都没有。
论文的模型选择有 selection bias。在 Llama-7B 上 work 的 2-bit 量化，在 Qwen3-8B 上不 work。不要假设结论能跨架构迁移。
系统设计的教训
不要追求”一种算法统治所有”。参数用 offloading，PP 用 chunked eviction，TG 用 scored quantization。三条路线独立演进，互不干扰。
Lazy > Eager。不到内存不够，不做压缩。压缩有代价。
Bound 你的问题。chunk size 和 max cache 一限制，O(N²) 变 O(1)，质量不降。有时候最好的优化是”少做一点”。
MLX Lazy Evaluation 是把双刃剑。用好了 40 层计算攒一起 flush，性能爆炸；一个 .item() 就全毁——从 90 tok/s 掉到 5.6 tok/s。
研究方法论
快速失败比缓慢成功更有价值。混合架构上的 AM 失败用了 2 天发现，省去了可能几周的无用功。正是这个失败催生了 Expert Offloading 路线——不压缩，而是卸载。
保留所有实验记录。这个项目产出了 100+ 份实验报告。每一次失败都有完整的数据和分析。
让数据说话，不是直觉。”加更多数据应该能 work”是直觉；”query 分布不匹配”是数据告诉我的。”.item() 应该很快”是直觉；”40 次 GPU→CPU 同步”是 profiler 告诉我的。
使用方式
KV Cache 压缩（Dense Transformer）
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-8B-Instruct")

# One line — auto-calibration on first use (~26s), cached afterwards (<1ms)
result = generate(model, tokenizer, prompt="Your long prompt here...",
                  kv_cache="scored_pq", kv_flat_quant="q8_0")
Expert Offloading（MoE 模型）
from mlx_lm import load
from mlx_lm.models.expert_offload import patch_model_for_offload

model, tokenizer = load("qwen3.5-35b-mlx")
ctx = patch_model_for_offload(model, model_path, max_workers=4, cpu_cache_gb=2.0)

# PP with full pool → compact → TG with compact pool
# ... generate tokens ...
ctx.compact(pool_size=128)  # 18.21 → 9.77 GB, TG: 92.8 tok/s
代码
项目地址：github.com/lisihao/FlashMLX

核心文件：

KV Cache 压缩（路线二+三）:

triple_layer_cache.py — Scored P2 + Chunked Prefill + Q8/Q4 Flat Buffer
cache_factory.py — 策略工厂 + 自适应参数
am_calibrator.py — 自动校准系统
quantization_strategies.py — 可插拔量化（Q4_0, Q8_0, PolarQuant）
Expert Offloading（路线一）:

expert_offload.py — Two-Phase Compact Pool + Speculative Execution + CPU Cache
这篇文章基于 2026 年 3 月 18 日至 29 日的开发记录。 KV Cache 数据来自 Qwen3-8B-MLX (Q8) on Apple M4 Pro 24GB。 Expert Offloading 数据来自 Qwen3.5-35B-A3B (Q4) on Apple M4 Pro 48GB。 所有测试在独立子进程中运行，串行执行。 FlashMLX v1.0 — MIT License
