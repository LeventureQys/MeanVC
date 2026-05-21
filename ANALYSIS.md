# MeanVC 深度技术分析报告

## 1. 项目概述

**MeanVC** 是一个轻量级、流式、零样本语音转换（Zero-Shot Voice Conversion）系统。其核心目标是将任意源说话人的语音内容保留，同时将音色（timbre）转换为任意目标说话人的音色，无需对目标说话人进行额外训练。

### 核心能力
- **流式/实时推理**：支持逐块（chunk-wise）处理，可用于麦克风实时变声
- **单步/少步生成**：通过 "Mean Flows" 技术，仅需 1-2 步去噪即可完成语音生成（传统扩散模型需 50+ 步）
- **零样本**：无需对未见过的目标说话人重新训练
- **轻量化**：DiT 骨干仅 4 层 Transformer，隐藏维度 512，参数规模远小于同类模型

---

## 2. 完整数据流

### 2.1 数据预处理流水线

```
原始音频 (.wav)
    │
    ├──► [Mel 提取器] ──► Mel 频谱 (80维, 10ms hop)
    │    ├── src/preprocess/extrace_mel_10ms.py
    │    └── 输出: mel.npy  shape=(T, 80)
    │
    ├──► [FastU2++ ASR编码器] ──► 内容特征 BN (256维, 160ms/200ms窗口)
    │    ├── src/preprocess/extract_bn_160ms.py (chunk_size=16的配置)
    │    ├── src/preprocess/extract_bn_200ms.py (chunk_size=20的配置)
    │    └── 输出: bn.npy  shape=(T, 256)
    │        注意: BN 特征会在数据集加载时 4x 上采样以匹配 Mel 帧率
    │
    └──► [WavLM + ECAPA-TDNN] ──► 说话人嵌入向量 (256维)
         ├── src/preprocess/extract_spk_emb_wavlm.py
         └── 输出: xvector.npy  shape=(256,)

数据格式 (train.list):
utt_id|bn_path|mel_path|xvector_path|prompt_mel_path1|prompt_mel_path2|...
```

### 2.2 训练数据流

```
                       ┌──────────────────────────────────────────────────┐
                       │              DiffusionDataset                     │
                       │  (src/dataset/dataset.py:18)                     │
                       ├──────────────────────────────────────────────────┤
                       │  1. 加载 BN   → 4x线性插值上采样                  │
                       │  2. 加载 Mel  → 若 min < -1.5 则 /4 归一化       │
                       │  3. 加载 Xvector → squeeze 到 (256,)             │
                       │  4. 加载 Prompt Mels → 拼接至 ≥2000帧 → 随机截取  │
                       │  5. 截断到 max_len (500-1000帧)                   │
                       │  6. custom_collate_fn: 变长序列 padding            │
                       └──────────────┬───────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Trainer (trainer.py:396)                      │
├─────────────────────────────────────────────────────────────────────────┤
│  for epoch in epochs:                                                   │
│    for batch in dataloader:                                             │
│      features = {                                                       │
│        "mel":     (B, T, 80)     # 目标梅尔频谱                           │
│        "bn":      (B, T, 256)    # 内容特征                              │
│        "xvector": (B, 256)       # 目标说话人嵌入                         │
│        "prompt":  (B, PT, 80)    # 参考提示音频Mel                        │
│        "inputs_length": (B,)     # 实际序列长度                           │
│      }                                                                   │
│                                                                          │
│      diff_loss, mse_val = self.meanflow.loss(                           │
│          self.model,          # DiT 模型                                 │
│          x=features["mel"],   # 真值 Mel                                 │
│          bn=features["bn"],   # 内容特征                                 │
│          spks=features["xvector"],  # 说话人嵌入                          │
│          prompts=features["prompt"], # 提示音频                           │
│          inputs_length=features["inputs_length"]                         │
│      )                                                                   │
│                                                                          │
│      accelerator.backward(diff_loss)                                     │
│      梯度裁剪 → optimizer.step() → scheduler.step() → EMA.update()       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 MeanFlow 训练机制核心流程

```
输入: x (真值Mel), bn (内容), spks (说话人嵌入), prompts (提示Mel)

Step 1 — 采样时间步 (t, r):
  从 Lognormal(-0.4, 1.0) 采样, 经 sigmoid → (0,1) 
  t = max(a, b), r = min(a, b)
  以 flow_ratio=0.5 概率令 r = t (单步模式)

Step 2 — 构造噪声样本:
  ε ~ N(0, I)
  z = (1 - t) * x + t * ε    # 在数据和噪声之间插值
  v = ε - x                    # 噪声方向向量 (数据→噪声)
  
Step 3 — CFG (无分类器引导):
  u_t = model(z, t, t, is_uncondition=True)   # 无条件预测
  v_hat = w * v + (1-w) * u_t                  # 引导向量 (w=2.0)
  以 cfg_ratio=0.10 概率对 batch 样本做 timbre dropout

Step 4 — JVP (雅可比向量积):
  u, du/dt = jvp(model(z), (z,t,r), (v_hat, 1, 0))
  u_tgt = v_hat - (t - r) * du/dt    # 目标向量场

Step 5 — Adaptive L2 Loss:
  error = u - u_tgt
  w_adapt = 1 / (||error||^2 + 1e-3)^(1-p)    # p=0.5
  loss = mean( stopgrad(w_adapt) * ||error||^2 )
```

### 2.4 推理数据流

```
源音频 ──► [FastU2++ ASR] ──► 内容特征 BN ──┐
                                             │
目标音频 ──► [WavLM SV] ──► 说话人嵌入 ──────┤
     │                                       │
     └──► [Mel提取器] ──► Prompt Mel ────────┤
                                             │
               噪声 z ~ N(0,I) ──► [DiT + MeanFlow 逐块去噪] ◄──┘
                                      │
                                 Mel 频谱 (80维)
                                      │
                              [Vocos 声码器] ──► 转换后音频 (16kHz)
```

#### 逐块 (Chunk-wise) 推理细节

```
初始化:
  timesteps = [1.0, 0.8, 0.0] (2步) 或 [1.0, 0.0] (1步)
  cache = None, kv_cache = None, offset = 0

for each chunk (chunk_size=20帧=200ms):
    x = randn(1, chunk_len, 80)          # 噪声初始化

    for i in range(steps):
        t, r = timesteps[i], timesteps[i+1]
        u = DiT(x, t, r, cache, cond=bn_chunk, spks, prompts, offset, kv_cache)
        x = x - (t - r) * u              # 去噪更新

    kv_cache 更新 → 截断至 max_len=100
    offset += x.shape[1]
    cache = x                             # 当前块输出作为下一块的 cache
    收集 x

输出: concat 所有 chunk 的 x → mel → Vocos → wav
```

#### 实时推理流水线 (run_rt.py)

```
麦克风 (16kHz, int16)
    │
    ├── 缓存 720 采样点 (45ms)
    │
    ├──► [FastU2++ chunk解码] 
    │    ├── stride = 4 * 5 = 20帧采样点
    │    ├── 输出: 5帧 BN 特征 (加上缓存共6帧)
    │    └── 4x上采样 → 20帧 Mel 特征
    │
    ├──► [MeanVC DiT] (TorchScript)
    │    ├── 20帧 Mel 逐块生成
    │    ├── KV-Cache 跨块共享, 最大长度=100
    │    └── 缓存前一块 Mel 输出作为 vc_cache
    │
    ├──► [Vocos 声码器]
    │    ├── 3帧 overlap-add
    │    ├── 线性 crossfade 平滑
    │    └── 输出 16kHz 音频
    │
    └──► pyaudio 输出流
```

---

## 3. 模型设计方法

### 3.1 总体架构：DiT (Diffusion Transformer)

**文件**: `src/model/backbones/dit.py`

```
                         ┌──────────────────────┐
     x (B,T,80) ────────►│                      │
     timbre_cond ───────►│   InputEmbedding     │──► (B,T,512)
     spks_ ─────────────►│  (592 → 512)         │
                         └──────────────────────┘
                                  │
                         ┌────────▼────────────┐
                  ┌──────┤   TimestepEmbedding  │  (t, r → dim=512)
                  │      └─────────────────────┘
                  │
     ┌────────────▼──────────────────────────────┐
     │         ChunkDiTBlock × 4                  │
     │  ┌──────────────────────────────────────┐ │
     │  │  AdaLayerNorm (dim→6×dim gate)      │ │
     │  │    shift_msa, scale_msa, gate_msa,   │ │
     │  │    shift_mlp, scale_mlp, gate_mlp    │ │
     │  ├──────────────────────────────────────┤ │
     │  │  ChunkAttention                      │ │
     │  │    - 同chunk内 self-attention        │ │
     │  │    - 跨chunk cache attention (5块内) │ │
     │  │    - 训练: right_mask (self+cache)   │ │
     │  │    - 推理: left_mask (causal)        │ │
     │  │    - Rotary Position Embedding       │ │
     │  │    - RMS QK Norm                     │ │
     │  ├──────────────────────────────────────┤ │
     │  │  FeedForward (dim×ff_mult → dim)     │ │
     │  │    - GELU 激活, Dropout              │ │
     │  └──────────────────────────────────────┘ │
     └────────────────┬──────────────────────────┘
                      │
              ┌───────▼────────┐
              │ AdaLayerNorm   │
              │   _Final       │
              ├────────────────┤
              │ Linear(512→80) │──► 输出 Mel
              └────────────────┘
```

### 3.2 关键设计创新

#### 3.2.1 MeanFlow (均值流) — 核心训练框架

**文件**: `src/model/cfm_mean_flow.py`

MeanFlow 基于条件流匹配 (Conditional Flow Matching, CFM)，关键创新：

1. **Mean Flow 概念**: 从 batch 中随机选取 `flow_ratio=50%` 的样本，将其时间变量对设为 `t == r`（即一步映射），取 "均值" 以降低训练难度
2. **Lognormal 时间分布**: `t ~ Lognormal(-0.4, 1.0)` 经过 sigmoid 映射到 (0,1)，比均匀采样更关注中间信噪比区域
3. **JVP (雅可比向量积) 训练**: 不使用显式的速度场预测，而是通过计算模型输出对时间 t 的导数来获得目标向量场
4. **Adaptive L2 Loss**: 权重由 `1/||Δ||^(1-p)` 确定，`p=0.5`，使损失对大幅值误差不太敏感

数学公式：
```
z  = (1-t)·x + t·ε                    # 插值
v  = ε - x                            # 目标方向

u_tgt = v_hat - (t - r)·∂u/∂t        # 经JVP修正的目标输出

loss = Σ stopgrad(w_i) · ||u_i - u_tgt_i||²
w_i  = 1 / (||Δ_i||² + 1e-3)^{1-p}    # p = 0.5
```

#### 3.2.2 MRTE (Multi-Reference Timbre Encoder) — 多参考音色编码器

**文件**: `src/model/prompt_vp.py`

```
cond (B,T,256) ──────────────── Query ──┐
                                        │
prompts (B,PT,80) ──► Key, Value ───────┤
                                  │      │
spks (B,256) ──► vp_proj(256→n_feat) ──┤
                 └─► 拼接到 Key ────────┘   MRTE Layer × 2
                                           │
                    ┌──────────────────────┘
                    │
           ┌────────▼──────────────────────────┐
           │  MRTELayer (Cross-Attention)       │
           │   1. self_attn: Query  ↕  Key+speaker_emb │
           │   2. FeedForward: MLP + LayerNorm  │
           └───────────────────────────────────┘
```

- 通过交叉注意力将 **内容特征(Query)** 与 **提示Mel(Key/Value) + 说话人嵌入** 对齐
- 说话人嵌入通过 `vp_proj` 投影后拼接到 prompt key 中，使内容特征 "感知" 目标音色
- 训练时 8% 的 Query 帧随机 mask (Dropout 正则化)

#### 3.2.3 Chunk Attention — 块级注意力机制

**文件**: `src/model/modules.py:537 (ChunkAttnProcessor)`

```
序列按 chunk_size 分块:
  块索引: [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
  N = seq_len / 2 / chunk_size

Mask 策略:
  ┌─ 训练 (right_mask) ─────────────────┐
  │  self: 同块内可见 (ci == cj)        │
  │  cache: 前N块中的前max_lookback=5块 │
  │  公式: ci < rel_j && ci >= rel_j-5  │
  └──────────────────────────────────────┘
  ┌─ 推理 (left_mask) ──────────────────┐
  │  因果掩码: 只看见前lookback_k=5块   │
  │  公式: cj-ci in [0, 5]              │
  └──────────────────────────────────────┘
```

#### 3.2.4 AdaLayerNorm — 自适应层归一化

**文件**: `src/model/modules.py:301`

与标准 DiT 类似，时间步嵌入通过 AdaLayerNorm 调节每个 Transformer Block：

```python
emb → SiLU → Linear(dim, 6×dim) → chunk(6)
→ shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

MSA:  norm(x) * (1 + scale_msa) + shift_msa
输出:  x = x + gate_msa * attn(norm)

FF:   norm(x) * (1 + scale_mlp) + shift_mlp
输出:  x = x + gate_mlp * ffn(norm)
```

#### 3.2.5 CFG (Classifier-Free Guidance)

```python
# 训练时 (MeanFlow.loss):
cfg_mask = (random < 0.10)              # 10% 样本丢弃 timbre 条件
u_t = model(z, t, t, is_uncondition=True)  # 无条件预测
v_hat = w * v + (1 - w) * u_t           # w=2.0 混合

# 推理时 (chunk-wise):
# cfg_mask = torch.ones(...)  # 始终使用条件，由 cfg_strength 在训练时隐式学习
```

#### 3.2.6 判别器 (GAN 变体)

**文件**: `src/model/dit_discriminator.py`, `src/model/trainer_dis.py`

- 与生成器相同的 DiT 骨干，但输出标量分数替代 Mel
- 使用 `AttentionAggregation` + `MLPHead` 进行分类
- 提取中间层 [1, 3] 做 Feature Matching
- R1 梯度惩罚确保训练稳定性
- GAN 训练入口: `src/train/train_2.py`

---

## 4. 模型各类参数

### 4.1 DiT 模型结构参数

| 参数 | 160ms 配置 | 200ms 配置 | 说明 |
|------|-----------|-----------|------|
| `dim` | 512 | 512 | 隐藏维度 |
| `depth` | 4 | 4 | DiT Block 层数 |
| `heads` | 2 | 2 | 注意力头数 |
| `dim_head` | 64 | 64 | 每头维度 (2×64=128 inner_dim) |
| `ff_mult` | 2 | 2 | FF内部维度乘数 (inner=1024) |
| `mel_dim` | 80 | 80 | 输入/输出 Mel 通道数 |
| `bn_dim` | 256 | 256 | 内容特征 (BN) 维度 |
| `chunk_size` | 16 | 20 | 块大小 (帧) |
| `dropout` | 0.0 | 0.0 | Dropout 比率 |
| `qk_norm` | rms_norm | rms_norm | QK 归一化方式 |
| `conv_layers` | 4 | 4 | (配置文件声明，实际未使用) |

### 4.2 各模块参数数量估算

```
InputEmbedding:
  Linear(80 + 256*2, 512)          = 592 × 512 + 512    ≈ 303,616

TimestepEmbedding (×2, t + r):
  SinusoidEmb(256)
  Linear(256, 512) + Linear(512, 512)                    ≈ 131,072 × 2

CacheEmbedding:
  Linear(80, 512)                  = 80 × 512 + 512     ≈ 41,472

MRTE (Timbre Encoder, 2 layers, 4 heads, 256-dim):
  MRTELayer × 2:
    每层: QKV proj + FF + norms                          ≈ 1,050,624 × 2
  vp_proj: Linear(256, 256)                              ≈ 65,536

ChunkDiTBlock × 4:
  每块:
    AdaLayerNorm: Linear(512, 3072)                      ≈ 1,573,888
    ChunkAttention: QKV(512→128) + Out(128→512)         ≈ 197,120
    LayerNorm + FeedForward(512→1024→512)               ≈ 1,574,400
  总计 4块                                                ≈ 13,381,632

Final:
  AdaLayerNorm_Final: Linear(512, 1024)                  ≈ 524,800
  ProjOut: Linear(512, 80)                               ≈ 41,040

─────────────────────────────────────────────────────────
总参数量估算: ~17-18M (包含所有模块)
```

### 4.3 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 8 (每GPU) | 每 GPU 的批量大小 |
| `learning_rate` | 1e-4 | AdamW 初始学习率 |
| `epochs` | 110 | 最大训练轮数 |
| `num_warmup_updates` | 2000 | 线性 warmup 步数 |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值 |
| `grad_accumulation_steps` | 1 | 梯度累积步数 |
| `total_steps` | 3,000,000 × num_processes | 总训练步数 |
| `seed` | 42 | 随机种子 |

### 4.4 学习率调度

```
warmup_steps = 2000 × num_processes
total_steps  = 3,000,000 × num_processes
decay_steps  = total_steps - warmup_steps

├── Warmup: Linear 1e-4×LR → LR  (warmup_steps步)
└── Decay:  Linear LR → 0.1×LR   (decay_steps步)
```

### 4.5 MeanFlow 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `flow_ratio` | 0.50 | 单步映射样本比例 |
| `time_dist` | lognorm(-0.4, 1.0) | 时间步采样分布 |
| `cfg_ratio` | 0.10 | CFG 条件丢弃概率 |
| `cfg_scale` (w) | 2.0 | CFG 引导强度 |
| `p` | 0.5 | Adaptive L2 Loss 幂参数 |
| `cfg_uncond` | 'u' | 无条件模式 |
| `jvp_api` | 'autograd' | JVP 计算后端 |

### 4.6 推理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `chunk_size` | 16 或 20 | 每块帧数 (160ms/200ms) |
| `steps` | 1-2 | 去噪步数 |
| `cfg_strength` | 2.0 (或 4.0) | 推理时 CFG 强度 |
| `timesteps` | [1.0, 0.8, 0.0] | 2步模式时间点 |
| `timesteps` | [1.0, 0.0] | 1步模式时间点 |
| `VC_KV_CACHE_MAX_LEN` | 100 | KV 缓存最大长度 |
| `vocoder_overlap` | 3 | 声码器重叠帧数 |

### 4.7 数据配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_len` | 500 (训练) | 最大序列长度 (帧) |
| `prompt_min_frames` | 2000 | 提示 Mel 拼接最小长度 |
| `feature_list` | bn, mel, xvector, audio | 输入特征列表 |
| `feature_pad_values` | 0.0, -1.0, 0.0, 0.0 | 各特征填充值 |
| BN 4x 上采样 | linear 插值 | BN 帧率 40ms → Mel 帧率 10ms |
| Mel 归一化 | if min < -1.5 then /4 | 简单幅值归一化 |
| MRTE mask_prob | 0.08 | 训练时 Query 帧随机丢弃概率 |
| ChunkAttention max_lookback | 5 | 训练时最大回溯块数 |
| ChunkAttention lookback_k | 5 | 推理时最大前瞻块数 |

### 4.8 外部模型依赖

| 组件 | 模型 | 用途 |
|------|------|------|
| ASR 编码器 | FastU2++ | 提取内容特征 (BN) |
| 说话人验证 | WavLM Large + ECAPA-TDNN | 提取说话人嵌入 |
| 声码器 | Vocos | Mel → 波形合成 |
| 评估 ASR | Paraformer-zh (FunASR) | 计算 WER/CER |
| 评估 SV | WavLM Large finetuned | 计算 SSMI (说话人相似度) |

---

## 5. 性能效果评估

### 5.1 评估指标

#### SSMI (Speaker Similarity Metric)
- 计算方式：转换语音的 WavLM 嵌入与目标说话人嵌入的余弦相似度
- 值域：[-1, 1]，越高越好
- 评估代码：`src/eval/verification.py`

#### WER/CER (Word/Character Error Rate)
- 使用 FunASR paraformer-zh 做语音识别
- 移除标点符号后与真值文本比较
- 评估代码：`src/eval/run_wer.py`

#### RTF (Real-Time Factor)
- 计算：`处理时间 / 音频时长`
- 评估代码：`src/infer/infer.py:79`
- < 1.0 表示比实时更快

### 5.2 验证流程

```
Trainer.validate(step):
  1. 加载验证集说话人列表 + BN文件 + 真值文本
  2. 初始化评估模型: WavLM (SSMI) + Paraformer (WER)
  3. for each 目标说话人:
       for each BN文件:
         逐块生成 Mel → Vocos → wav
         计算 SSMI (与目标嵌入比较)
         做 ASR 转写 → 计算 WER
  4. 记录 mean SSMI 和 WER 到 wandb
```

### 5.3 性能指标总结

```
┌─────────────────────┬─────────────────┬──────────────────────┐
│ 指标                │ 含义            │ 预期/报告值          │
├─────────────────────┼─────────────────┼──────────────────────┤
│ 参数量              │ 模型大小        │ ~17-18M              │
│ 推理步数            │ 去噪步数        │ 1-2 (vs 传统 50+)    │
│ 单步延迟 (chunk)    │ 处理延迟        │ < 块时长 (实时)      │
│ RTF                 │ 实时因子        │ < 1.0                │
│ SSMI                │ 说话人相似度    │ 见论文 arXiv          │
│ WER/CER             │ 内容保留度      │ 见论文 arXiv          │
│ 音质                │ 主观/客观质量   │ 声称优于现有方法      │
└─────────────────────┴─────────────────┴──────────────────────┘
```

### 5.4 与其他方法的对比优势 (源自论文声称)

1. **速度**: 1-2 步完成生成 vs 扩散模型 50+ 步，推理速度提升 25-50 倍
2. **参数量**: ~18M vs 典型 DiT 模型 100M+，显著降低
3. **流式能力**: 原生支持 chunk-wise KV-cache 推理，无需特殊适配
4. **音质**: 通过 MeanFlow + GAN 微调，声称在语音自然度和说话人相似度上优于现有方法

### 5.5 已知限制

- 预训练模型从 Hugging Face (`ASLP-lab/MeanVC`) 下载
- WavLM 验证模型需从 Google Drive 手动下载
- 评估指标（ASR, SV）依赖外部大型模型
- GAN 训练需更小的学习率 (`5e-5`) 和较少的步数 (~3e6)
- 实时推理依赖 TorchScript 导出的模型文件

---

## 6. 文件索引

### 核心源码
| 文件 | 行数 | 功能 |
|------|------|------|
| `src/model/backbones/dit.py` | 208 | DiT 骨干网络定义 |
| `src/model/cfm_mean_flow.py` | 448 | MeanFlow 训练框架 + 损失函数 |
| `src/model/modules.py` | 904 | 所有 NN 构建块 (Attention, Block, Norm 等) |
| `src/model/prompt_vp.py` | 333 | MRTE 音色编码器 |
| `src/model/trainer.py` | 491 | 标准训练器 + 验证逻辑 |
| `src/model/trainer_dis.py` | 585 | GAN 训练器 + 判别器 |
| `src/model/dit_discriminator.py` | 283 | DiT 判别器 |
| `src/model/loss.py` | 36 | GAN 损失函数 |
| `src/model/utils.py` | 92 | 工具函数 |
| `src/dataset/dataset.py` | 248 | 数据集类 + collate |
| `src/infer/infer.py` | 173 | 离线推理 (预提取特征) |
| `src/infer/infer_ref.py` | 312 | 离线推理 (从参考音频) |
| `src/infer/dit_kvcache.py` | 204 | DiT KV-cache 推理骨干 |
| `src/infer/modules.py` | 625 | 推理用 Attention 模块 |
| `src/runtime/run_rt.py` | 325 | 实时变声主程序 |
| `src/eval/verification.py` | 91 | 说话人验证评估 |
| `src/eval/run_wer.py` | 43 | WER 计算 |
| `src/eval/ecapa_tdnn.py` | 301 | ECAPA-TDNN 模型 |
| `src/train/train.py` | 68 | 训练入口 |
| `src/train/train_2.py` | 68 | GAN 训练入口 |

### 预处理
| 文件 | 行数 | 功能 |
|------|------|------|
| `src/preprocess/extrace_mel_10ms.py` | 248 | Mel 频谱提取 |
| `src/preprocess/extract_bn_160ms.py` | 103 | BN 内容特征提取 (160ms) |
| `src/preprocess/extract_bn_200ms.py` | 101 | BN 内容特征提取 (200ms) |
| `src/preprocess/extract_spk_emb_wavlm.py` | 103 | 说话人嵌入提取 |

### 配置文件
| 文件 | 说明 |
|------|------|
| `defaults.ini` | 训练默认超参数 |
| `default_config.yaml` | HuggingFace Accelerate 配置 |
| `src/config/config_160ms.json` | 160ms chunk 模型配置 |
| `src/config/config_200ms.json` | 200ms chunk 模型配置 |
| `download_ckpt.py` | 预训练模型下载脚本 |
| `requirements.txt` | Python 依赖 |

### 脚本
| 文件 | 说明 |
|------|------|
| `scripts/train.sh` | 标准训练启动 |
| `scripts/train_dis.sh` | GAN 训练启动 |
| `scripts/infer.sh` | 离线推理 (预提取特征) |
| `scripts/infer_ref.sh` | 离线推理 (参考音频) |
