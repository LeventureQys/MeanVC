# MeanVC 项目架构文档

> **基于源代码阅读的独立分析，非 README 转述。**

---

## 1. 项目概述

**MeanVC** 是一个轻量级、流式、零样本语音转换（Zero-Shot Voice Conversion）系统，由西北工业大学 ASLP Lab 与吉利汽车研究院联合开发。论文发表于 arXiv:2510.08392。

### 核心能力

| 特性 | 含义 | 实现方式 |
|------|------|----------|
| **流式推理** | 逐块（chunk-wise）处理，支持麦克风实时变声 | Chunk Attention + KV-Cache |
| **少步生成** | 1-2 步去噪即可完成转换 | MeanFlow 条件流匹配 |
| **零样本** | 无需对未见过的目标说话人重新训练 | 说话人嵌入 + 参考音频 Prompt |
| **轻量化** | DiT 仅 4 层 Transformer，约 14.2M 参数 | 窄维度（dim=512）、浅层数 |

### 技术栈

| 组件 | 技术选型 | 在项目中的角色 |
|------|----------|---------------|
| 深度学习框架 | PyTorch 2.5.1 | 模型定义、训练、推理 |
| 分布式训练 | HuggingFace Accelerate | 多 NPU 并行训练 |
| 内容编码器 | WeNet FastU2++ (TorchScript) | ASR Bottleneck 特征提取 |
| 说话人验证 | WavLM-Large + ECAPA-TDNN | 目标说话人嵌入提取 |
| 声码器 | Vocos (TorchScript) | Mel 频谱 → 波形 |
| Transformer 组件 | x-transformers | RoPE、Attention 基础实现 |
| 实时音频 I/O | PyAudio | 麦克风输入 / 扬声器输出 |
| GUI 框架 | PyQt6 | 桌面实时变声应用 |
| 音频处理 | librosa, torchaudio | 音频加载、重采样、Mel 提取 |
| 评估 | FunASR (Paraformer), jiwer | WER/CER 计算 |

### 模型仓库

预训练权重托管在 HuggingFace: `ASLP-lab/MeanVC`，通过 `download_ckpt.py` 自动下载。说话人验证模型 `wavlm_large_finetune.pth` (约 1.2GB) 需从 Google Drive 手动下载。

---

## 2. 目录结构

```
MeanVC/
├── main.py                          # PyQt6 桌面 GUI（1361 行），含 ModelLoader、VCWorker、MetricsPanel
├── download_ckpt.py                 # 从 HuggingFace 下载预训练模型到 src/ckpt/
├── defaults.ini                     # 训练超参数默认值（batch_size, lr, epochs 等）
├── default_config.yaml              # HuggingFace Accelerate 配置（MULTI_NPU 分布式）
├── requirements.txt                 # Python 依赖清单
├── README.md / README_cn.md         # 用户入门文档（中/英）
├── ANALYSIS.md                      # 深度技术分析报告（中文，已有）
├── LICENSE                          # Apache 2.0
│
├── src/
│   ├── config/                      # DiT 模型结构配置（JSON）
│   │   ├── config_200ms.json        # chunk_size=20（200ms 窗口）
│   │   └── config_160ms.json        # chunk_size=16（160ms 窗口）
│   │
│   ├── ckpt/                        # 预训练模型权重（由 download_ckpt.py 填充）
│   │   ├── fastu2++.pt              # ASR 编码器（WeNet TorchScript, ~88MB）
│   │   ├── meanvc_200ms.pt          # VC 模型（TorchScript, ~54MB）
│   │   ├── model_200ms.safetensors  # VC 模型（SafeTensors, ~54MB，与上相同权重）
│   │   └── vocos.pt                 # Vocos 声码器（TorchScript, ~32MB）
│   │
│   ├── model/                       # ★ 核心模型定义（训练用）
│   │   ├── __init__.py              # 导出 MeanFlow, DiT, Trainer, TrainerDis
│   │   ├── backbones/dit.py         # DiT 骨干网络（训练版，无 KV-cache）
│   │   ├── cfm_mean_flow.py         # MeanFlow 条件流匹配训练框架 + 损失函数（447 行）
│   │   ├── prompt_vp.py             # MRTE 多参考音色编码器 + 辅助编码器类
│   │   ├── modules.py               # 所有 NN 构建块（904 行）
│   │   │                           #   - MelSpec, AdaLayerNorm, ChunkAttention
│   │   │                           #   - ChunkDiTBlock, FeedForward, TimestepEmbedding
│   │   │                           #   - AttnProcessor, ChunkAttnProcessor, JointAttnProcessor
│   │   ├── dit_discriminator.py     # DiT 判别器（GAN 阶段使用）
│   │   ├── trainer.py              # 标准训练器（纯 CFM 损失）
│   │   ├── trainer_dis.py          # GAN 训练器（生成器 + 判别器对抗训练）
│   │   ├── loss.py                 # GAN 损失函数（生成器/判别器/特征/蒸馏损失）
│   │   └── utils.py                # 工具函数（checkpoint 加载、seed、mask 生成）
│   │
│   ├── infer/                       # ★ 离线推理（含 KV-cache 版本 DiT）
│   │   ├── dit_kvcache.py           # DiT 推理版（支持 KV-cache，返回 (output, new_kv_cache)）
│   │   ├── modules.py               # 推理版 Attention 模块（ChunkAttnProcessor 含 cache 拼接逻辑）
│   │   ├── infer.py                 # 批处理推理（使用预提取的特征文件）
│   │   └── infer_ref.py             # 端到端推理（从音频文件提取特征 → 转换 → 保存）
│   │
│   ├── runtime/                     # 实时推理
│   │   ├── run_rt.py                # CLI 实时变声（命令行版 VCRunner）
│   │   ├── example/test.wav         # 示例参考音频
│   │   └── speaker_verification/    # 说话人验证模型（运行时使用）
│   │       ├── verification.py      # 模型初始化 + 嵌入提取
│   │       ├── ecapa_tdnn.py        # ECAPA-TDNN 模型定义
│   │       ├── utils.py             # WavLM 加载工具（S3PRL UpstreamExpert）
│   │       └── ckpt/wavlm_large_finetune.pth  # WavLM-Large 微调权重（~1.2GB）
│   │
│   ├── train/                       # 训练入口
│   │   ├── train.py                 # 标准训练（阶段 1：纯 CFM）
│   │   └── train_2.py               # GAN 训练（阶段 2：加判别器）
│   │
│   ├── dataset/
│   │   └── dataset.py               # DiffusionDataset + custom_collate_fn
│   │
│   ├── preprocess/                  # 数据预处理脚本
│   │   ├── extrace_mel_10ms.py      # Mel 频谱提取（10ms 帧移, 80 维）
│   │   ├── extract_bn_160ms.py      # BN 内容特征提取（160ms 窗口）
│   │   ├── extract_bn_200ms.py      # BN 内容特征提取（200ms 窗口）
│   │   ├── extract_spk_emb_wavlm.py # 说话人嵌入提取
│   │   └── models/                  # 预处理用的 ECAPA-TDNN + WavLM 工具
│   │       ├── ecapa_tdnn.py        # （与 runtime/ 中重复）
│   │       ├── utils.py             # （与 runtime/ 中重复）
│   │       └── __init__.py          # 空文件
│   │
│   └── eval/                        # 评估脚本
│       ├── verification.py          # 说话人相似度 (SSMI) 评估
│       ├── run_wer.py               # WER 计算（FunASR Paraformer 识别 → jiwer）
│       ├── ecapa_tdnn.py            # ECAPA-TDNN（与上面重复）
│       └── utils.py                 # WavLM 工具（与上面重复）
│
├── vocos/                           # Vocos 神经声码器（来自 gemelo-ai/vocos）
│   ├── __init__.py                  # 导出 Vocos, __version__ = "0.1.0"
│   ├── models.py                    # VocosBackbone (ConvNeXt) / VocosResNetBackbone
│   ├── modules.py                   # ConvNeXtBlock, ResBlock1, AdaLayerNorm
│   ├── heads.py                     # ISTFTHead, IMDCTSymExpHead, IMDCTCosHead
│   ├── spectral_ops.py              # ISTFT, MDCT, IMDCT 实现
│   ├── feature_extractors.py        # MelSpectrogramFeatures, EncodecFeatures
│   ├── discriminators.py            # MPD, MRD 判别器（Vocos 训练用）
│   ├── pretrained.py                # Vocos 类（load, decode, forward）
│   ├── loss.py                      # Mel 重建损失, 生成器/判别器损失
│   ├── helpers.py                   # 可视化, GradNormCallback
│   ├── dataset.py                   # Vocos 训练数据集
│   └── experiment.py                # Vocos 训练实验管理
│
├── scripts/                         # Shell 启动脚本
│   ├── train.sh                     # 标准训练（accelerate launch）
│   ├── train_dis.sh                 # GAN 训练
│   ├── infer.sh                     # 离线推理（预提取特征）
│   └── infer_ref.sh                 # 离线推理（参考音频）
│
├── temp/                            # 调试与测试脚本
│   ├── test_core.py                 # 单元测试（Mel 提取、FBANK、模型加载）
│   ├── test_vcworker.py             # 单元测试（VCWorker 状态管理）
│   ├── test_gui_smoke.py            # GUI 烟雾测试
│   ├── debug_conversion.py          # 离线管线对比（原版 vs GUI 版 VCWorker）
│   ├── debug_threading.py           # 主线程 vs QThread 加载测试
│   ├── debug_mic_loop.py            # 麦克风到文件实际测试
│   └── *.wav                        # 调试输出音频
│
├── target_voice/                    # 示例目标音色文件
│   ├── celeb_zh_zhenhuan.wav
│   └── celeb_zh_caixukun.wav
│
├── log/                             # 运行时日志
│   ├── vc_worker.log                # VCWorker 详细诊断
│   ├── device_info.txt              # 音频设备信息
│   ├── summary.txt                  # 处理摘要
│   └── chunk_*.wav                  # 分块调试音频
│
├── figs/                            # README 图片
│   ├── model.png                    # 模型架构图
│   ├── meanvc_QR.png                # 微信群二维码
│   ├── npu@aslp.jpeg                # NPU ASLP Lab logo
│   └── geely_logo.jpg               # 吉利 logo
│
└── Document/                        # 文档
    ├── LEARNING_GUIDE.md            # 6 周学习指南（以 MeanVC 为锚点学 VC）
    ├── MeanVC_轻量化分析报告.md       # 轻量化分析与内存估算
    └── PROJECT_ARCHITECTURE.md       # ★ 本文档
```

---

## 3. 系统架构与数据流

### 3.1 五组件流水线

```
                        ┌──────────────────────────────────────┐
                        │          ① 内容编码器                  │
  源音频 ──────────────►│   FastU2++ ASR Encoder               │
  (16kHz)               │   输出: BN 特征 (T_bn, 256)           │
                        │   窗口: 160ms 或 200ms                │
                        └──────────────┬───────────────────────┘
                                       │
                        ┌──────────────▼───────────────────────┐
                        │          ② 说话人嵌入                  │
  参考音频 ────────────►│   WavLM-Large + ECAPA-TDNN            │
  (目标音色)            │   输出: 说话人向量 (256,)              │
                        │   仅启动时运行一次，不在推理循环中     │
                        └──────────────┬───────────────────────┘
                                       │
                        ┌──────────────▼───────────────────────┐
                        │        ③ 音色编码器 (MRTE)            │
                        │   交叉注意力:                          │
                        │   Q = BN 内容特征                      │
                        │   K = Prompt Mel + 投影后说话人嵌入    │
                        │   V = Prompt Mel                       │
                        │   输出: 音色条件特征 (T, 256)          │
                        └──────────────┬───────────────────────┘
                                       │
                        ┌──────────────▼───────────────────────┐
                        │     ④ DiT 骨干网络 (流匹配)           │
                        │   输入: 噪声 Mel + 音色条件 + 说话人   │
                        │   4× ChunkDiTBlock                    │
                        │   输出: 预测速度场 u (T, 80)           │
                        │   1-2 步 Euler ODE 求解               │
                        └──────────────┬───────────────────────┘
                                       │
                        ┌──────────────▼───────────────────────┐
                        │         ⑤ Vocos 声码器                │
                        │   Mel 频谱 (80 维) → 波形 (16kHz)     │
                        │   ConvNeXt Backbone + ISTFT Head      │
                        │   3 帧 overlap-add 平滑               │
                        └──────────────┬───────────────────────┘
                                       │
                                       ▼
                                  输出音频 (16kHz)
```

### 3.2 训练数据流

#### 3.2.1 预处理阶段

三个特征分别从原始音频中提取：

```
原始音频 (.wav, 16kHz)
    │
    ├──► [Mel 提取] ──► mel.npy  shape=(T_mel, 80)
    │    src/preprocess/extrace_mel_10ms.py
    │    参数: n_fft=1024, hop_length=160 (10ms), n_mels=80
    │
    ├──► [FastU2++] ──► bn.npy  shape=(T_bn, 256)
    │    src/preprocess/extract_bn_200ms.py (或 _160ms.py)
    │    chunked 编码: decoding_chunk_size=5 (或 4)
    │    帧移: 200ms (5×4×10ms) 或 160ms (4×4×10ms)
    │
    └──► [WavLM+ECAPA] ──► xvector.npy  shape=(256,)
         src/preprocess/extract_spk_emb_wavlm.py
         全局话语级嵌入向量
```

#### 3.2.2 数据集加载

文件格式 (`train.list`):
```
utt_id|bn_path|mel_path|xvector_path|prompt_mel1|prompt_mel2|...
```

[DiffusionDataset](src/dataset/dataset.py) 加载时的处理：
1. **BN 上采样**: `F.interpolate(..., scale_factor=4, mode='linear')` — 从 ASR 帧率对齐到 Mel 帧率
2. **Mel 归一化**: 若 `min < -1.5`，整个 Mel 除以 4
3. **Prompt 拼接**: 多段参考 Mel 拼接至 ≥ 2000 帧，然后随机截取
4. **截断**: 所有序列截断到 `max_len`（默认 500-1000 帧）
5. **Collation**: `custom_collate_fn` 对变长序列 zero-padding

#### 3.2.3 MeanFlow 训练机制

MeanFlow 的核心思想是**在训练时就让模型熟悉单步映射**，缩小训练-推理差距。

```
训练循环 (src/model/trainer.py):

for batch in dataloader:
    features = {
        "mel":     (B, T, 80)     # 目标 Mel 频谱
        "bn":      (B, T, 256)    # 内容特征（已 4x 上采样）
        "xvector": (B, 256)       # 说话人嵌入
        "prompt":  (B, PT, 80)    # 参考提示 Mel
        "inputs_length": (B,)     # 实际长度
    }

    # MeanFlow.loss() 内部:
    t, r = sample_t_r(batch_size)      # 对数正态分布 → sigmoid
    # flow_ratio=0.5: 一半样本设 t=r（极端情况，单步映射）

    e ~ N(0,I)
    z = (1-t)·x + t·e                  # 噪声插值
    v = e - x                          # 目标速度场

    u_t = model(z, t, t, ..., is_uncondition=True)  # 无条件预测
    v_hat = w·v + (1-w)·u_t            # CFG 引导向量 (w=cfg_scale)

    # JVP 计算 du/dt:
    u, du/dt = jvp(model, (z, t, r), (v_hat, 1, 0))

    u_tgt = v_hat - (t-r)·du/dt        # 轨迹匹配目标
    loss = adaptive_l2_loss(u - sg(u_tgt), mask)
```

**关键设计点：**

- **时间采样**: 从 Lognormal(-0.4, 1.0) 采样，经 sigmoid 得到 (0,1) 范围内的 (t, r) 对。强制 `r ≤ t`。
- **flow_ratio=0.5**: 一半训练样本设置 `t=r`（直接从 t=1 映射到 t=0），使模型学会单步生成。
- **JVP 训练**: 使用 `torch.autograd.functional.jvp` 计算 Jacobian-vector product 得到 `du/dt`，构造轨迹匹配目标。
- **自适应 L2 损失**: `w * ||Δ||²`，其中 `w = 1/(||Δ||² + 1e-3)^(1-p)`，`p=0.5`。通过 `stopgrad(w)` 防止权重梯度影响损失尺度。
- **CFG (Classifier-Free Guidance)**: 训练时 10% 概率丢弃音色条件（`cfg_mask`），推理时使用 `w=2.0` 进行引导。
- **CFG 实现细节**: 代码中有一个 `cfg_uncond` 选项（默认 `'u'`），当设为 `'v'` 时会用原始速度场 v 替代 CFG 向量中的无条件部分。代码中有 TODO 注释质疑为何用 `r` 而非 `random_tensor` 生成 cfg_mask。

### 3.3 流式推理数据流

```
麦克风输入
    │
    ▼
┌─────────────────────────────────────────────┐
│ PyAudio 读取: 3200 采样点/块 (200ms @16kHz) │
│ 首块额外 720 采样点 (45ms) 用于缓存预热      │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ Kaldi FBANK: 80 维, 帧移 10ms               │
│ 200ms 窗口 → 20 帧 fbank                     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ FastU2++ 流式编码器                          │
│ forward_encoder_chunk(chunk, offset, ...)    │
│ decoding_chunk_size=5, subsampling=4         │
│ 每块输出 5 帧 BN 特征 (256 维)               │
│ 维护 att_cache, cnn_cache                    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ BN 上采样: 4× 线性插值 (5 → 20 帧)          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ DiT 流匹配去噪 (1-2 步 Euler)                │
│ x = randn(B, 20, 80)                         │
│ for t, r in [(1.0, 0.8), (0.8, 0.0)]:       │
│     u, new_kv = model(x, t, r, cond=bn,      │
│         spks=emb, prompts=prompt_mel,        │
│         cache=prev_x, kv_cache=kv_cache)      │
│     x = x - (t-r)·u                          │
│ KV-cache 截断: 保留最近 100 帧               │
│ CFG: 推理时 w=2.0 引导                       │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ Vocos 解码                                    │
│ Mel (1, 80, T) → 波形 (1, T_audio)           │
│ 3 帧 overlap-add + cross-fade 平滑           │
│ cache 保留最后 3 帧用于下次拼接               │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
              扬声器输出 (16kHz float32)
```

**缓存管理（三级缓存）:**

| 缓存 | 存储内容 | 截断策略 |
|------|---------|---------|
| ASR cache | `att_cache`, `cnn_cache` (编码器状态) | 不截断，流式编码器自动管理 |
| VC cache | `x_pred` (前一帧 Mel 输出) + `kv_cache` (各层 KV) | KV 截断到最近 100 帧 |
| Vocoder cache | 最后 3 帧 Mel | 固定 3 帧 overlap |

**Chunk 定时复位**: 每 50 个 chunk 复位 ASR/VC/Vocoder 缓存，防止状态累积漂移。

---

## 4. 核心模型组件详解

### 4.1 DiT 骨干网络

**文件**: [src/model/backbones/dit.py](src/model/backbones/dit.py)（训练版）、[src/infer/dit_kvcache.py](src/infer/dit_kvcache.py)（推理版）

```
DiT(
  dim=512,          # 隐藏维度
  depth=4,          # Transformer 层数
  heads=2,          # 注意力头数
  dim_head=64,      # 每头维度（总注意维度 = heads × dim_head = 128）
  ff_mult=2,        # FFN 扩展倍率（FFN 维度 = 512 × 2 = 1024）
  mel_dim=80,       # 输入/输出 Mel 维度
  bn_dim=256,       # 内容特征维度
  chunk_size=20,    # 块大小（200ms 配置）
  qk_norm="rms_norm" # QK 归一化
)
```

**前向传播流程:**

```
输入: x(B,T,80) + t(B,) + r(B,) + cache(B,T_prev,80) + cond(B,T,256) + spks(B,256) + prompts(B,PT,80)

1. spks → unsqueeze + repeat → (B,T,256)

2. t → t_time_embed (Sinusoidal + MLP) → (B,dim)
   r → r_time_embed → (B,dim)
   t = t + r                            # 时间嵌入求和

3. timbre_cond = MRTE(cond, prompts, spks)  # (B,T,bn_dim=256)
   # CFG 掩码: 若 cfg_mask 为 True，timbre_cond 和 spks 清零

4. x = InputEmbedding(x, timbre_cond, spks)  # cat([x, cond, spks]) → Linear → (B,T,512)
   # 若 cache 不为空: x = cat([cache_embed(cache), x])  # (B, T_prev+T, 512)

5. RoPE 位置编码（训练 vs 推理有不同 offset 逻辑）

6. for block in 4× ChunkDiTBlock:
       x = block(x, t, mask, rope)
       # 推理版同时返回 new_kv_cache

7. x = x[:, -seq_len:, :]              # 只保留新帧
8. x = AdaLayerNorm_Final(x, t)
9. output = Linear(x) → (B,T,80)
```

**参数初始化策略:**
- 所有 AdaLayerNorm 的线性层 → 零初始化 (`weight=0, bias=0`)
- `norm_out` 和 `proj_out` → 零初始化
- 这意味着训练初期输出为零，模型逐步学会非零预测

**训练版 vs 推理版差异:**

| 差异点 | 训练版 (backbones/dit.py) | 推理版 (infer/dit_kvcache.py) |
|--------|--------------------------|------------------------------|
| KV-Cache | 不支持 | 支持，返回 `new_kv_cache` |
| RoPE 计算 | 区分 cache/无 cache 两种情况 | 统一用 offset 计算，截断到 140 帧 |
| 导入路径 | `from src.model.modules import ...` | `from modules import ...`（相对导入） |
| `long_skip_connection` | 支持 | 不支持 |

### 4.2 Chunk Attention 机制

**文件**: [src/model/modules.py](src/model/modules.py) L493-580（训练版）、[src/infer/modules.py](src/infer/modules.py) L400-500（推理版）

```
ChunkAttnProcessor:
  - chunk_size=20 (或 16)
  - max_lookback=5 (训练时，最多看 5 个历史块)
  
训练时注意力掩码:
  ┌──────────────┬─────────┐
  │ 块内全注意力  │ 跨块=0  │
  │ (chunk_size) │  (其余) │
  ├──────────────┼─────────┤
  │ 历史块最多5个 │ 其余=0  │
  │ (5×chunk)    │         │
  └──────────────┴─────────┘

推理时注意力掩码:
  - 因果掩码 (当前及过去可看，未来不可看)
  - KV-cache: 每层保存 (key, value) 对
  - 增量更新: 每块添加新 key/value，保留最近 100 帧
```

**关键实现细节:**
- 训练时的 `right_mask` 使用块内自注意力 + 有限历史窗口
- 推理时的 `left_mask` 使用因果掩码 + KV-cache
- `ChunkAttnProcessor` 通过 `is_inference` 标志切换行为
- 注释中有被注释掉的 `F.scaled_dot_product_attention` 调用，当前使用手动实现
- NTK-aware RoPE 缩放被提及但未启用（注释中参考了 RoPE 长度外推技术）

### 4.3 MeanFlow 损失框架

**文件**: [src/model/cfm_mean_flow.py](src/model/cfm_mean_flow.py) (448 行)

**核心类 `MeanFlow`:**

```python
MeanFlow(
    flow_ratio=0.50,         # 单步映射训练比例
    time_dist=['lognorm', -0.4, 1.0],  # 时间分布
    cfg_ratio=0.10,          # CFG 丢弃概率
    cfg_scale=2.0,           # CFG 引导强度
    p=0.5,                   # 自适应损失指数
    jvp_api='autograd',      # JVP 实现方式
)
```

**损失函数族:**

| 方法 | 用途 | 训练阶段 |
|------|------|---------|
| `loss()` | 完整 MeanFlow 损失（JVP + 轨迹匹配） | 阶段 1 主损失 |
| `loss_one_step_only()` | 单步直接预测损失（t=1, r=0） | 阶段 1 辅助 |
| `loss_ema_one_step_only()` | 含 EMA 模型的单步损失 | 阶段 2 蒸馏 |
| `discrimi()` | 判别器真实/生成/扰动评分 | 阶段 2 GAN |

**自适应 L2 损失:**
```python
def adaptive_l2_loss(error, mask, gamma=0., c=1e-3):
    delta_sq = mean(error², dim=-1)       # 每个 token 的 MSE
    p = 1.0 - gamma                        # = 0.5
    w = 1.0 / (delta_sq + c)^p            # 权重：误差越大，权重越小
    return mean(stopgrad(w) * delta_sq)    # stopgrad 防止权重梯度干扰
```

这种损失设计的直觉是：对已经预测很好的 token 给予更高权重（精细调优），对预测差的 token 降低权重（避免被异常值主导）。

### 4.4 MRTE 音色编码器

**文件**: [src/model/prompt_vp.py](src/model/prompt_vp.py) L209-233

```
MRTE(n_head=4, n_feat=256, dropout_rate=0.,
     q_in_dim=256,    # BN 内容特征维度
     k_in_dim=80,     # Prompt Mel 维度
     v_in_dim=80,     # Prompt Mel 维度
     num_blocks=2)

前向传播:
  query = cond               # (B, T1, 256) — ASR 内容特征
  key   = prompts            # (B, T2, 80)  — 参考音频 Mel
  value = prompts            # (B, T2, 80)
  GE    = vp_proj(spks)      # (256,) → (n_feat,)
  key   = cat([key, GE.unsqueeze(1).repeat(1, T2, 1)], dim=-1)  # (B, T2, 80+256)

  for layer in 2× MRTELayer:
      query = query + dropout(cross_attn(query, key, value))
      query = LayerNorm(query)
      query = query + dropout(FFN(query))
      query = LayerNorm(query)
  
  return query               # (B, T1, 256) — 音色条件特征
```

**设计要点:**
- 说话人嵌入通过线性投影 (`vp_proj`) 从 256 维映射到 `n_feat=256` 维，然后拼接到 Key 上
- 这相当于在交叉注意力中同时注入帧级（prompt mel）和全局（说话人嵌入）音色信息
- 不使用 dropout（`dropout_rate=0.`）
- 代码中还包含其他辅助编码器类（`PromptVPEncoder`, `PromptEncoder`, `TextEncoder`, `TransformerEncoder`, `CrossAttentionEncoder`），但这些在 DiT 中未被使用，可能是实验残留

### 4.5 DiT 判别器（GAN 阶段）

**文件**: [src/model/dit_discriminator.py](src/model/dit_discriminator.py) (283 行)

```
DiT_dis:
  - 结构与生成器 DiT 完全相同（4 层 ChunkDiTBlock）
  - 在第 1 层和第 3 层后提取中间特征
  - AttentionAggregation: 对每层中间特征做注意力池化
  - MLPHead: cat(聚合特征) → Linear+ReLU → Linear → 标量评分
```

**GAN 损失** ([src/model/loss.py](src/model/loss.py)):
- `disc_loss`: hinge loss（真实样本 > 0，生成样本 < 0）+ R1 梯度惩罚 (γ=100)
- `gen_loss`: 对抗损失 + 蒸馏损失 (EMA 模型输出作为软标签)
- `feature_loss`: 判别器中间层特征的 L1 匹配

### 4.6 Vocos 声码器

**文件**: [vocos/](vocos/) 目录

Vocos 是一个基于 ConvNeXt 的神经声码器，将 Mel 频谱转换为波形。

**推理时使用的组件:**
- `VocosBackbone`: ConvNeXt 块堆叠（深度可配置，默认 6 层）
- `ISTFTHead`: 预测 STFT 幅度和相位，通过逆 STFT 合成波形

**部署格式**: TorchScript (`vocos.pt`, ~32MB)，通过 `torch.jit.load` 加载。

---

## 5. 训练流程

### 5.1 阶段 1：标准 CFM 训练

**入口**: [src/train/train.py](src/train/train.py) → [src/model/trainer.py](src/model/trainer.py)

```bash
bash scripts/train.sh "0,1,2,3"   # 指定 GPU ID
```

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| batch_size | 16 (per GPU) | 总批次 = 16 × GPU 数 |
| learning_rate | 1e-4 | AdamW |
| epochs | 1000 | 实际由收敛情况决定 |
| warmup | 2000 steps | 线性 warmup |
| lr decay | linear → 0.1× lr | warmup 后线性衰减 |
| max_grad_norm | 1.0 | 梯度裁剪 |
| grad_accumulation | 1 | 梯度累积步数 |
| flow_ratio | 0.50 | MeanFlow t=r 概率 |
| cfg_ratio | 0.1 | CFG 丢弃概率 |
| cfg_scale | 2.0 | CFG 引导强度 |
| steps | 1 (训练时) | 训练时模拟 1 步推理 |
| max_len | 1000 | 最大序列帧数 |
| seed | 42 | 随机种子（resumable_with_seed=666） |

**训练循环关键逻辑:**
1. 加载 batch → 前向传播 `meanflow.loss()` → `accelerator.backward(loss)`
2. 梯度裁剪 → optimizer.step() → scheduler.step() → EMA.update()
3. 每 `save_per_updates=10000` 步保存 checkpoint
4. 验证时生成样本音频 → 计算 SSMI (说话人相似度) 和 WER
5. 支持 HuggingFace Accelerate 的多 NPU 分布式训练

### 5.2 阶段 2：GAN 对抗训练

**入口**: [src/train/train_2.py](src/train/train_2.py) → [src/model/trainer_dis.py](src/model/trainer_dis.py)

在阶段 1 收敛后，加载最佳 checkpoint 继续训练：
- 新增 `DiT_dis` 判别器，独立 optimizer 和 scheduler
- 判别器学习率: 5e-5（低于生成器）
- 生成器损失 = CFM 损失 + 对抗损失 + 蒸馏损失
- 判别器损失 = Hinge loss + R1 梯度惩罚 (γ=100)
- 蒸馏: EMA 模型的单步预测作为软标签

### 5.3 评估指标

| 指标 | 实现 | 含义 |
|------|------|------|
| SSMI | [src/eval/verification.py](src/eval/verification.py) | 转换后语音与目标说话人的余弦相似度 |
| WER | [src/eval/run_wer.py](src/eval/run_wer.py) | FunASR Paraformer 识别 → jiwer 计算词错误率 |
| RTF | [src/infer/infer_ref.py](src/infer/infer_ref.py) L234 | 推理时间 / 音频时长（< 1 表示实时） |

---

## 6. 推理流水线

### 6.1 实时 CLI

**文件**: [src/runtime/run_rt.py](src/runtime/run_rt.py) (325 行)

```bash
python src/runtime/run_rt.py --target-path "target_voice/celeb_zh_caixukun.wav"
```

`VCRunner` 类的推理循环：
1. 加载 4 个模型（SV, ASR, VC, Vocoder）
2. 提取目标说话人嵌入和 Prompt Mel（一次性）
3. 列出音频设备，用户选择输入/输出
4. 麦克风循环: 读取 3200 采样点 → 提取 FBANK → ASR 编码 → BN 上采样 → DiT 去噪 → Vocos 解码 → 扬声器输出

### 6.2 桌面 GUI

**文件**: [main.py](main.py) (1362 行)

```bash
python main.py
```

基于 PyQt6 的桌面应用，包含以下组件：

| 组件 | 类型 | 功能 |
|------|------|------|
| `ModelLoader` | QThread | 后台加载 4 个模型 + 提取目标说话人特征 |
| `VCWorker` | QThread | 实时 VC 推理循环（同 CLI 逻辑） |
| `DeviceScanner` | QThread | 枚举 PyAudio 输入/输出设备 |
| `MetricsPanel` | QWidget | 滚动显示 chunk 耗时、总延迟、内存、RTF |
| `MeanVCApp` | QMainWindow | 主窗口：文件选择、设备选择、控制按钮、指标面板 |

**GUI 特有设计:**
- **macOS 兼容**: PyAudio 流必须在主线程打开（`main.py:1216` 注释），因此音频 I/O 在主线程，VCWorker 通过信号传递数据
- **深色主题**: 使用暗色调 QPalette
- **设备发现**: 启动时自动扫描音频设备，也可手动刷新
- **去噪步数**: 用户可选 1-8 步（默认 2）
- **首块预热**: 第一个 chunk 额外附加 720 采样点（45ms）用于流式编码器缓存预热
- **周期复位**: 每 50 个 chunk 复位所有缓存

### 6.3 离线推理

#### infer_ref.py（端到端）

```bash
python src/infer/infer_ref.py \
    --model-config src/config/config_200ms.json \
    --ckpt-path src/ckpt/model_200ms.safetensors \
    --vocoder-ckpt-path src/ckpt/vocos.pt \
    --source-path path/to/source.wav \
    --reference-path path/to/reference.wav \
    --output-dir src/outputs \
    --chunk-size 20 --steps 2
```

流程:
1. 加载 4 个模型
2. 对每个源音频:
   - 提取 FBANK → ASR 流式编码 → BN 特征
   - 提取参考音频的说话人嵌入和 Prompt Mel
   - DiT 流式去噪（逐块，含 KV-cache）
   - Vocos 解码 → 保存 .wav
3. 输出 RTF（实时因子）统计

#### infer.py（预提取特征）

从已保存的 .npy 特征文件推理，跳过 ASR/SV 提取步骤。适用于大规模评估。

---

## 7. 代码观察与注意事项

以下是阅读代码过程中发现的值得注意的细节，部分在 README 中未被提及：

### 7.1 代码重复

ECAPA-TDNN 和 WavLM 加载工具在项目中重复出现在 3 个位置：
- `src/preprocess/models/` — 预处理阶段使用
- `src/eval/` — 评估阶段使用
- `src/runtime/speaker_verification/` — 运行时使用

这些副本并非完全相同，可能存在版本差异，重构时需谨慎。

### 7.2 双 DiT 实现

存在两个 DiT 实现：
- [src/model/backbones/dit.py](src/model/backbones/dit.py) — 训练用，无 KV-cache
- [src/infer/dit_kvcache.py](src/infer/dit_kvcache.py) — 推理用，支持 KV-cache

两者的 `ChunkDiTBlock` 和 `ChunkAttnProcessor` 也分别在 [src/model/modules.py](src/model/modules.py) 和 [src/infer/modules.py](src/infer/modules.py) 中有独立实现。推理版返回 `(output, new_kv_cache)` 元组，训练版只返回 `output`。

### 7.3 CPU 推理

代码中观察到推理默认使用 CPU (`device='cpu'`, `torch.set_num_threads(1)`)。这意味着项目在设计上优先考虑了跨平台兼容性和部署便利性，而非 GPU 加速。对于实时推理场景（RTF 约 0.3-0.5），CPU 性能已足够。

### 7.4 硬编码值

| 位置 | 值 | 含义 |
|------|-----|------|
| `main.py:1220` | `CHUNK = 160 * 20 = 3200` | GUI 音频块大小固定为 3200 采样点 |
| `src/infer/dit_kvcache.py:181` | `rope[0][:, -140:, :]` | RoPE 截断到 140 帧 |
| `src/infer/infer_ref.py:18` | `C_KV_CACHE_MAX_LEN = 100` | KV-cache 最大帧数 |
| `main.py:386` | 首块额外 720 采样点 | 流式编码器缓存预热 |
| `src/model/prompt_vp.py:119` | `self.mask_prob = 0.08` | MRTE 训练 dropout |

### 7.5 未解决的 TODO

- `src/model/cfm_mean_flow.py:184`: `# todo: why use r generate cfg_mask?` — 质疑 CFG 掩码生成方式
- `src/model/cfm_mean_flow.py:183`: `# as v = wv - (1-w)v = wv - (1-w)u in the unconditional case, should we directly use v instead?` — 对 CFG 实现正确性的疑问
- `src/runtime/speaker_verification/ecapa_tdnn.py:153`: `# DON'T use ReLU here! In experiments, I find ReLU hard to converge.` — AttentiveStatsPool 中明确禁用 ReLU（该注释在三处 ecapa_tdnn.py 副本中均存在）
- `src/model/modules.py:493`: 注释掉的 `F.scaled_dot_product_attention`（causal attention 尝试）

### 7.6 内存使用

根据 [Document/MeanVC_轻量化分析报告.md](Document/MeanVC_轻量化分析报告.md) 的分析：
- 运行时总内存约 1.8-2.2 GB
- WavLM-Large 约占 1.2 GB（315M 参数），但仅在启动时用于提取嵌入，之后可卸载
- 不含 WavLM 的自建框架内存约 400-450 MB
- FP16 量化后可降至约 180-200 MB

### 7.7 两种 checkpoint 格式

`src/ckpt/` 中同时存在 `meanvc_200ms.pt`（TorchScript, 54MB）和 `model_200ms.safetensors`（SafeTensors, 54MB），包含相同的权重。`.pt` 文件用于 `torch.jit.load` 加载，`.safetensors` 用于标准 `load_checkpoint` 加载。两者并存可能是为了兼容不同的部署场景。

---

## 8. 依赖关系

[requirements.txt](requirements.txt) 中每个关键依赖在项目中的具体用途：

| 包名 | 在项目中的角色 |
|------|---------------|
| `torch==2.5.1` | 深度学习框架核心 |
| `torchaudio==2.5.1` | Kaldi FBANK 特征提取、音频 I/O |
| `torchvision==0.20.1` | （未直接使用，torch 依赖） |
| `librosa` | 音频加载 (librosa.load)、Mel 滤波器组 (librosa.filters.mel) |
| `einops` | 张量重排 (rearrange, repeat) |
| `x-transformers` | RotaryEmbedding (RoPE), apply_rotary_pos_emb |
| `tqdm` | 进度条（训练、预处理） |
| `PyYAML` | 读取 default_config.yaml |
| `omegaconf` | 结构化配置管理（trainer 中使用） |
| `transformers` | （可能用于 HuggingFace 模型加载） |
| `pyaudio` | 实时音频 I/O（麦克风、扬声器） |
| `accelerate` | HuggingFace Accelerate 分布式训练 |
| `matplotlib` | 频谱图可视化 |
| `wandb` | 训练日志与实验跟踪 |
| `ema_pytorch` | EMA（指数移动平均）模型维护 |
| `jiwer==3.1.0` | WER/CER 计算 |
| `zhon` | 中文文本处理（WER 评估用） |
| `zhconv` | 中文繁简转换 |
| `funasr` | FunASR Paraformer 语音识别（WER 评估） |
| `encodec` | Meta EnCodec 神经音频编解码器（备选特征提取） |
| `prefigure` | 训练参数管理 |

---

## 9. 已有文档索引

| 文档 | 内容 | 适用对象 |
|------|------|---------|
| [README.md](../README.md) / [README_cn.md](../README_cn.md) | 项目入门指南，环境配置，快速开始 | 初次使用者 |
| [ANALYSIS.md](../ANALYSIS.md) | 深度中文技术分析，含完整数据流、参数估算 | 研究人员 |
| [Document/LEARNING_GUIDE.md](LEARNING_GUIDE.md) | 6 周 VC 编解码器学习指南 | 学习者 |
| [Document/MeanVC_轻量化分析报告.md](MeanVC_轻量化分析报告.md) | 内存分析与轻量化方案 | 工程部署 |
| **Document/PROJECT_ARCHITECTURE.md**（本文档） | 基于代码阅读的项目架构完整说明 | 开发者、贡献者 |

---

> **文档生成说明**: 本文档完全基于对项目源代码的独立阅读和分析编写，所有结论均有代码依据。文件路径、参数数值、数据流描述均来自对实际代码的验证。最后更新：2026-05-31。
