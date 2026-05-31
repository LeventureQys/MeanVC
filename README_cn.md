# MeanVC: 基于均值流的轻量级流式零样本语音转换系统

<div align="center">

[![论文](https://img.shields.io/badge/arXiv-2510.08392-b31b1b.svg)](https://arxiv.org/pdf/2510.08392)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-模型-yellow)](https://huggingface.co/ASLP-lab/MeanVC)
[![演示页面](https://img.shields.io/badge/演示-音频样本-green)](https://aslp-lab.github.io/MeanVC/)

</div>

**MeanVC** 是一个轻量级、流式的零样本语音转换系统，能够将任意源说话人的音色实时转换为任意目标说话人的音色，同时保留语言内容。该系统引入了一种扩散Transformer，结合逐块自回归去噪策略和均值流（mean flows）技术，实现了高效的单步推理。

![img](figs/model.png)

## ✨ 核心特性

-   **🚀 流式推理**：通过逐块处理实现实时语音转换。
-   **⚡ 单步生成**：利用均值流直接从起点映射到终点，实现快速生成。
-   **🎯 零样本能力**：无需重新训练，即可转换到任意未见过的目标说话人。
-   **💾 轻量级**：参数量显著少于现有方法。
-   **🔊 高保真**：卓越的语音质量和说话人相似度。

## 🚀 快速开始

### 1. 环境配置

按照以下步骤克隆仓库并安装所需环境。

```bash
# 克隆仓库并进入目录
git clone https://github.com/ASLP-lab/MeanVC.git
cd MeanVC

# 创建并激活 Conda 环境
conda create -n meanvc python=3.11 -y
conda activate meanvc

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载预训练模型

运行提供的脚本自动下载所有必要的预训练模型。

```bash
python download_ckpt.py
```

这会将主VC模型、声码器和ASR模型下载到 `src/ckpt/` 目录中。

说话人验证模型（`wavlm_large_finetune.pth`）需要从 Google Drive 手动下载。请从 [此链接](https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view) 下载文件，并将下载的 `wavlm_large_finetune.pth` 文件放入 `src/runtime/speaker_verification/ckpt/` 目录中。

### 3. 实时语音转换

此脚本从麦克风捕获音频，并将其实时转换为目标说话人的声音。

```bash
python src/runtime/run_rt.py --target-path "path/to/target_voice.wav"
```

-   `--target-path`：目标说话人干净音频文件的路径。该声音将用作转换目标。示例文件位于 `src/runtime/example/test.wav`。

运行脚本时，系统会提示您从列表中选择音频输入（麦克风）和输出（扬声器）设备。

### 4. 离线语音转换

对于批量处理或转换预录制的音频文件，请使用离线转换脚本。

```bash
bash scripts/infer_ref.sh
```

在运行脚本之前，需要在 `scripts/infer_ref.sh` 中配置以下路径：

-   `source_path`：待转换的源音频文件或包含多个音频文件的目录路径
-   `reference_path`：目标说话人干净音频文件的路径（用作声音参考）
-   `output_dir`：转换后音频文件的保存目录（默认：`src/outputs`）
-   `steps`：去噪步数（默认：2）

## 🏋️‍♀️ 训练

要在自己的数据集上训练模型，请按照以下步骤操作。

### 1. 数据预处理

首先，需要从音频数据中提取梅尔频谱、内容特征（BN）和说话人嵌入。

```bash
# 1. 提取梅尔频谱（10ms 帧移）
python src/preprocess/extrace_mel_10ms.py --input_dir path/to/wavs --output_dir path/to/mels

# 2. 提取内容特征（160ms 窗口）
python src/preprocess/extract_bn_160ms.py --input_dir path/to/wavs --output_dir path/to/bns

# 3. 提取说话人嵌入
python src/preprocess/extract_spk_emb_wavlm.py --input_dir path/to/wavs --output_dir path/to/xvectors
```

### 2. 准备数据列表

创建训练用的文件列表（例如 `train.list`）。每行应遵循以下格式：

```
# 格式：utt|bn_path|mel_path|xvector_path|prompt_mel_path1|prompt_mel_path2|...
utterance_id_001|/path/to/bns/utt001.npy|/path/to/mels/utt001.npy|/path/to/xvectors/utt001.npy|/path/to/mels/prompt01.npy
```

### 3. 开始训练

修改 `script/train.sh` 中的配置（例如数据路径、模型目录），然后运行脚本。

```bash
bash script/train.sh
```

## 📋 待办事项

-   [x] 🌐 **演示网站**
-   [x] 📝 **论文发布**
-   [x] 🤗 **HuggingFace 模型发布**
-   [x] 🔓 **发布推理代码**
-   [x] 🔓 **发布训练代码**
-   [ ] 📱 **Android 部署包**

## 📜 许可与免责声明

MeanVC 基于 Apache License 2.0 协议发布。本开源许可证允许您自由使用、修改和分发模型，前提是您包含适当的版权声明和免责声明。

MeanVC 专为语音转换技术的研究和合法应用而设计。使用者必须获得被转换或用作参考声音的个人的适当同意。我们强烈反对任何恶意用途，包括冒充、欺诈或制作误导性音频内容。使用者有责任确保其使用场景符合伦理标准和法律要求。

## ❤️ 致谢

我们的工作基于以下开源项目：[MeanFlow](https://github.com/haidog-yaqub/MeanFlow)、[F5-TTS](https://github.com/SWivid/F5-TTS) 和 [Vocos](https://github.com/gemelo-ai/vocos)。感谢作者们的出色工作，如果您有任何问题，可以先查阅他们各自的 issues。

## 📄 引用

如果您觉得我们的工作有帮助，请引用我们的论文：

```bibtex
@article{ma2025meanvc,
  title={MeanVC: Lightweight and Streaming Zero-Shot Voice Conversion via Mean Flows},
  author={Ma, Guobin and Yao, Jixun and Ning, Ziqian and Jiang, Yuepeng and Xiong, Lingxin and Xie, Lei and Zhu, Pengcheng},
  journal={arXiv preprint arXiv:2510.08392},
  year={2025}
}
```

## 📧 联系我们

如果您有兴趣向我们的研究团队留言，欢迎发送邮件至 guobin.ma@mail.nwpu.edu.cn

欢迎加入我们的微信群进行技术讨论和交流更新。

<p align="center">
    <img src="figs/meanvc_QR.png" width="300"/>
</p>

<p align="center">
    <img src="figs/npu@aslp.jpeg" height="120" style="margin-right: 20px;"/>
    <img src="figs/geely_logo.jpg" height="120"/>
</p>
