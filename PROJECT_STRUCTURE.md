# CAM++ 说话人识别项目 — 结构与执行流程

## 1. 项目目录树

```
camplusplus/
├── requirements.txt                          # Python 依赖
├── note/                                     # 实验笔记
│   ├── codec_test.md
│   └── Experiment_Table_Template.md
│
├── pretrained/                               # 预训练模型
│   └── speech_campplus_sv_cn_cnceleb_16k/
│
├── egs/3dspeaker/sv-cam++/                   # 实验配方（recipe）
│   ├── run.sh                                # 主实验驱动脚本（入口）
│   ├── path.sh                               # 环境变量设置
│   ├── conf/
│   │   ├── cam++.yaml                        # 基线训练配置
│   │   └── cam++_cnceleb_mixed_ft.yaml       # CN-Celeb 微调配置
│   ├── local/
│   │   ├── download_data.sh                  # 数据下载
│   │   ├── prepare_data.sh                   # 数据准备
│   │   ├── prepare_data_csv.py               # 生成训练 CSV 文件
│   │   ├── download_data_vctk.py             # 下载 VCTK 数据集
│   │   ├── prepare_vctk_split.py             # VCTK 数据集划分
│   │   ├── resample_vctk.py                  # VCTK 重采样到 16kHz
│   │   ├── simulate_codec.py                 # 编解码器仿真（Opus/G.711/AMR-WB）
│   │   ├── evaluate_eer.py                   # 计算 EER 评测指标
│   │   ├── run_pretrained_codec_eval.py      # 预训练模型编解码器评测
│   │   ├── build_mixed_codec_trainset.py     # 构建混合编解码器训练集
│   │   ├── prepare_cnceleb_mixed_data.py     # CN-Celeb 混合数据准备
│   │   └── check_cnceleb2_archive.py         # CN-Celeb2 压缩包校验
│   ├── utils/
│   │   └── parse_options.sh                  # Shell 参数解析工具
│   └── exp/                                  # 实验输出目录（模型/日志）
│       ├── campp_vctk/config.yaml
│       └── campp_cnceleb_mixed_ft/config.yaml
│
└── speakerlab/                               # 核心库
    ├── bin/                                  # 可执行入口脚本
    │   ├── train.py                          # 训练入口（核心）
    │   ├── train_para.py                     # 并行训练
    │   ├── extract.py                        # 说话人嵌入提取
    │   ├── compute_score_metrics.py          # EER/minDCF 计算
    │   ├── infer_sv.py                       # 单对推理
    │   ├── infer_sv_batch.py                 # 批量推理
    │   └── ...                               # 其他训练变体
    ├── dataset/
    │   └── dataset.py                        # WavSVDataset 数据集类
    ├── models/campplus/
    │   ├── DTDNN.py                          # CAMPPlus 模型主体
    │   ├── layers.py                         # 基础层（TDNN/CAM/池化等）
    │   └── classifier.py                     # 分类器（Cosine/Linear）
    ├── process/
    │   ├── processor.py                      # 数据处理器（读音频/提特征/增强/编码标签）
    │   ├── augmentation.py                   # 数据增强（加噪/混响）
    │   └── scheduler.py                      # 学习率/Margin 调度器
    ├── loss/
    │   └── margin_loss.py                    # ArcMarginLoss 等损失函数
    └── utils/
        ├── config.py                         # YAML 配置加载
        ├── builder.py                        # 动态对象构建器
        ├── checkpoint.py                     # 检查点保存/恢复
        ├── epoch.py                          # Epoch 计数/日志
        ├── fileio.py                         # 文件 I/O 工具
        ├── utils.py                          # 通用工具（日志/种子/指标）
        └── score_metrics.py                  # EER/minDCF 计算函数
```

---

## 2. 整体执行流程

```
┌─────────────────────────────────────────────────────────┐
│                      run.sh (主入口)                     │
│                                                         │
│  Stage 1 ─→ 数据下载与准备                                │
│  Stage 2 ─→ 生成训练 CSV                                 │
│  Stage 3 ─→ 模型训练 (train.py)                          │
│  Stage 4 ─→ 嵌入提取 (extract.py)                        │
│  Stage 5 ─→ 评分与指标计算 (compute_score_metrics.py)      │
└─────────────────────────────────────────────────────────┘
```

### Stage 1 — 数据下载与准备

```
run.sh
  └─→ local/prepare_data.sh
        └─→ local/download_data.sh    # 下载原始数据集
```

### Stage 2 — 生成训练 CSV

```
run.sh
  └─→ local/prepare_data_csv.py
        读取 wav.scp，遍历所有音频文件
        输出 train.csv (格式: ID, 时长, 路径, 说话人ID)
```

### Stage 3 — 模型训练（核心）

```
run.sh
  └─→ python -m speakerlab.bin.train --config conf/cam++.yaml
        │
        ├── config.py:build_config()     ← 加载并解析 YAML 配置
        ├── builder.py:build()           ← 根据配置动态实例化所有组件
        │
        │   构建顺序:
        │   ① WavReader         (processor.py)    读音频 + 截取
        │   ② SpkLabelEncoder   (processor.py)    说话人 ID → 整数标签
        │   ③ SpkVeriAug        (processor.py)    数据增强
        │   │   └─→ NoiseReverbCorrupter (augmentation.py)  加噪/混响
        │   ④ FBank             (processor.py)    提取梅尔滤波器组特征
        │   ⑤ WavSVDataset      (dataset.py)      PyTorch Dataset
        │   ⑥ DataLoader        (torch)           批数据加载
        │   ⑦ CAMPPlus          (DTDNN.py)        说话人嵌入模型
        │   │   └─→ layers.py   (TDNN/CAM/池化层)
        │   ⑧ CosineClassifier  (classifier.py)   余弦分类器
        │   ⑨ SGD Optimizer     (torch)           优化器
        │   ⑩ ArcMarginLoss    (margin_loss.py)   ArcFace 损失
        │   ⑪ WarmupCosineScheduler (scheduler.py) 学习率调度
        │   ⑫ MarginScheduler   (scheduler.py)    Margin 调度
        │   ⑬ Checkpointer      (checkpoint.py)   模型保存/恢复
        │
        └── 训练循环:
            for epoch in epochs:
                for batch in dataloader:
                    features, labels = batch
                    embeddings = CAMPPlus(features)
                    logits = CosineClassifier(embeddings)
                    loss = ArcMarginLoss(logits, labels)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                margin_scheduler.step()
                checkpointer.save()
```

### Stage 4 — 嵌入提取

```
run.sh
  └─→ python -m speakerlab.bin.extract
        ├── 加载训练好的 CAMPPlus 模型
        ├── 对测试集每条音频提取 512 维嵌入向量
        └── 输出 .ark/.scp 文件 (Kaldi 格式)
```

### Stage 5 — 评分与指标

```
run.sh
  └─→ python -m speakerlab.bin.compute_score_metrics
        ├── 读取嵌入向量 (.ark)
        ├── 读取试验对列表 (trials)
        ├── 计算余弦相似度
        └── 计算 EER / minDCF (score_metrics.py)
```

---

## 3. 各文件详细说明

### 3.1 实验入口层 (`egs/3dspeaker/sv-cam++/`)

| 文件 | 作用 |
|------|------|
| `run.sh` | 主实验脚本，串联五个阶段，支持 `stage` 参数控制从哪一步开始 |
| `path.sh` | 设置 `PYTHONPATH`，使 `speakerlab` 包可被导入 |
| `conf/cam++.yaml` | 基线配置：30 epoch, 800类, lr=0.02, batch=128, embedding=512 |
| `conf/cam++_cnceleb_mixed_ft.yaml` | CN-Celeb 微调配置：10000类, lr=0.01, batch=64, embedding=192 |

### 3.2 数据工具脚本 (`egs/.../local/`)

| 文件 | 作用 | 依赖 |
|------|------|------|
| `prepare_data_csv.py` | 遍历音频文件生成训练 CSV | torchaudio |
| `download_data_vctk.py` | 从 Kaggle 下载 VCTK 数据集 | kagglehub |
| `prepare_vctk_split.py` | 将 VCTK 划分为训练/测试/验证集 | — |
| `resample_vctk.py` | 将 VCTK 音频从 48kHz 重采样到 16kHz | torchaudio |
| `simulate_codec.py` | 通过 FFmpeg 仿真各种编解码器降质 | ffmpeg (subprocess) |
| `evaluate_eer.py` | 使用 ModelScope 预训练模型计算 EER | modelscope, scipy, sklearn |
| `run_pretrained_codec_eval.py` | 完整编解码器评测流水线 | simulate_codec, evaluate_eer |
| `build_mixed_codec_trainset.py` | 构建包含原始+编解码器处理的混合训练集 | simulate_codec |
| `prepare_cnceleb_mixed_data.py` | CN-Celeb 数据集全流程准备 | subprocess 调用其他脚本 |
| `check_cnceleb2_archive.py` | 校验 CN-Celeb2 压缩包完整性 | — |

### 3.3 核心训练库 (`speakerlab/bin/`)

| 文件 | 作用 | 主要依赖 |
|------|------|----------|
| `train.py` | **训练入口**：加载配置 → 构建组件 → 运行训练循环 | config, builder, epoch, checkpoint |
| `extract.py` | 嵌入提取：加载模型 → 逐条提取 → 输出 ark/scp | builder, config, fileio, kaldiio |
| `compute_score_metrics.py` | 评分：读嵌入 → 余弦相似度 → EER/minDCF | kaldiio, sklearn, score_metrics |
| `infer_sv.py` | 单对推理：两段音频 → 相似度分数 | processor, builder, modelscope |
| `infer_sv_batch.py` | 批量推理 | 同上 + multiprocessing |

### 3.4 数据处理 (`speakerlab/dataset/`, `speakerlab/process/`)

| 文件 | 类/函数 | 作用 |
|------|---------|------|
| `dataset.py` | `WavSVDataset` | PyTorch Dataset，按 CSV 索引读取音频并经 preprocessor 处理 |
| `processor.py` | `WavReader` | 读取 WAV、随机截取指定时长、可选语速扰动 |
| | `SpkLabelEncoder` | 从 CSV 中收集所有说话人 ID，映射为整数标签 |
| | `FBank` | 提取 Mel 滤波器组特征 (torchaudio Kaldi 兼容) |
| | `SpkVeriAug` | 以一定概率对音频施加噪声或混响增强 |
| `augmentation.py` | `NoiseReverbCorrupter` | 加载噪声/RIR 文件，执行加噪和混响卷积 |
| `scheduler.py` | `WarmupCosineScheduler` | 预热 + 余弦退火学习率调度 |
| | `MarginScheduler` | 线性递增 ArcFace margin |

### 3.5 模型定义 (`speakerlab/models/campplus/`)

| 文件 | 类 | 作用 |
|------|-----|------|
| `layers.py` | `TDNNLayer` | 时延神经网络层 (1D 卷积 + BN + ReLU) |
| | `CAMLayer` | 上下文感知掩蔽层 — 通过全局统计产生频率注意力掩模 |
| | `CAMDenseTDNNLayer` | CAM + TDNN 的密集连接单元 |
| | `CAMDenseTDNNBlock` | 多个 CAMDenseTDNNLayer 组成的密集块 |
| | `TransitLayer` | 通道转换层 (降维) |
| | `StatsPool` | 统计池化 (均值 + 标准差) |
| | `DenseLayer` | 全连接 + BN + ReLU |
| | `BasicResBlock` | 2D 残差块 (用于前端 FCM) |
| `DTDNN.py` | `FCM` | 前端卷积模块：2D 卷积 + 残差块，处理时频特征 |
| | `CAMPPlus` | **完整模型**：FCM → TDNN → 3×CAMDenseTDNNBlock → StatsPool → 嵌入 |
| `classifier.py` | `CosineClassifier` | 权重归一化余弦相似度分类器 |
| | `LinearClassifier` | 标准线性分类器 |

### 3.6 损失函数 (`speakerlab/loss/`)

| 文件 | 类 | 作用 |
|------|-----|------|
| `margin_loss.py` | `ArcMarginLoss` | ArcFace 损失：在角度空间添加加性 margin，增强类间区分度 |
| | `AddMarginLoss` | CosFace 损失 (备选) |
| | `EntropyLoss` | 交叉熵损失 (备选) |

### 3.7 工具库 (`speakerlab/utils/`)

| 文件 | 作用 |
|------|------|
| `config.py` | 加载 YAML 配置，将 `<ref>` 占位符解析为嵌套的 Config 对象 |
| `builder.py` | **动态对象构建**：根据 `obj` 字段用 `importlib` 导入类，用 `args` 实例化 |
| `checkpoint.py` | 模型检查点保存与恢复，管理最佳/最新模型 |
| `epoch.py` | `EpochCounter` 维护当前 epoch；`EpochLogger` 记录训练日志 |
| `fileio.py` | 文件读写工具：`load_data_csv`, `load_wav_scp`, `load_yaml` 等 |
| `utils.py` | 通用工具：`set_seed`, `get_logger`, `AverageMeters`, `accuracy` |
| `score_metrics.py` | `compute_eer`, `compute_c_norm`, DET 曲线绘制 |

---

## 4. 文件引用关系图

```
                           ┌──────────────┐
                           │   run.sh     │
                           │   (主入口)    │
                           └──────┬───────┘
                      ┌───────────┼───────────┐
                      ▼           ▼           ▼
               prepare_data   train.py    extract.py ──→ compute_score_metrics.py
                  .sh/.py        │                              │
                                 │                              ▼
                      ┌──────────┴──────────┐           score_metrics.py
                      ▼                     ▼
                config.py              builder.py
                (加载YAML)        (动态实例化对象)
                      │                     │
                      ▼                     ▼
               cam++.yaml ─────────→ 实例化以下组件：
                                            │
            ┌───────────┬───────────┬───────┴───────┬──────────┐
            ▼           ▼           ▼               ▼          ▼
       dataset.py  processor.py  DTDNN.py    classifier.py  margin_loss.py
      (WavSVDataset)    │       (CAMPPlus)  (CosineClassifier) (ArcMarginLoss)
            │           │           │
            ▼           ▼           ▼
       fileio.py  augmentation.py  layers.py
     (load_data_csv) (加噪/混响)  (TDNN/CAM/池化)
```

```
辅助关系:

train.py ──→ utils.py       (日志, 种子, 指标显示)
         ──→ epoch.py        (EpochCounter, EpochLogger)
         ──→ checkpoint.py   (模型保存/恢复)
         ──→ scheduler.py    (学习率/Margin 调度)

processor.py ──→ augmentation.py ──→ fileio.py
             ──→ fileio.py
```

---

## 5. YAML 配置的动态构建机制

YAML 配置文件不仅是参数文件，更是**对象组装清单**。每个组件定义为：

```yaml
component_name:
  obj: fully.qualified.ClassName    # 要实例化的类
  args:
    param1: value1
    param2: <other_component>       # 引用其他组件（自动递归构建）
```

`builder.py` 的 `build()` 函数执行以下流程：
1. 解析 `obj` 字段，通过 `importlib` 动态导入对应的类
2. 递归解析 `args` 中的 `<ref>` 引用，先构建被依赖的组件
3. 用解析后的参数调用类构造函数，返回实例

这种机制使得整个训练流水线可以仅通过修改 YAML 文件来重新配置，无需修改任何 Python 代码。

---

## 6. 两套实验配置对比

| 参数 | `cam++.yaml` (基线) | `cam++_cnceleb_mixed_ft.yaml` (微调) |
|------|---------------------|--------------------------------------|
| embedding_size | 512 | 192 |
| num_classes | 800 | 10000 |
| num_epoch | 30 | 30 |
| batch_size | 128 | 64 |
| lr | 0.02 | 0.01 |
| aug_prob | 0.5 | 0.8 |
| speed_pertub | False | True |
| 用途 | 从零训练 | 基于预训练模型微调 |

---

## 7. 外部依赖

### Python 包 (requirements.txt)
- `torch >= 1.10.1`, `torchaudio >= 0.10.1` — 深度学习框架
- `scipy`, `numpy`, `scikit-learn` — 科学计算与指标计算
- `kaldiio` — Kaldi ark/scp 格式读写
- `pyyaml` — YAML 配置解析
- `tqdm` — 进度条
- `soundfile`, `matplotlib`, `pandas`, `openpyxl` — 音频/可视化/表格

### 可选依赖
- `modelscope` — 预训练模型下载与推理
- `kagglehub` — VCTK 数据集下载
- `ffmpeg` — 编解码器仿真 (系统工具)
