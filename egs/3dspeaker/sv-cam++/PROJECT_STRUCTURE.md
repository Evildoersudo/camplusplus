# CAM++ 项目结构与调用层级说明

本文档用于说明当前仓库 `camplusplus` 的目录结构、关键文件职责，以及从数据准备到训练、提取 embedding、评测的调用层级。

适用范围：

- 当前仓库：`E:\Speaker_recognition\Graduation_Project\camplusplus`
- 当前 recipe：`egs/3dspeaker/sv-cam++`
- 当前主任务：基于 CAM++ 的说话人验证（speaker verification）

## 1. 项目总体分层

这个仓库可以分成两层：

1. `egs/3dspeaker/sv-cam++`
   负责实验编排，也就是 recipe 层。这里控制数据准备、训练、提 embedding、评测。
2. `speakerlab/`
   负责通用能力实现，也就是框架层。这里放训练脚本、模型、数据处理、损失函数、调度器、工具函数。

可以把它理解为：

`recipe 层负责“怎么跑实验”`  
`speakerlab 层负责“实验真正调用了哪些代码模块”`

---

## 2. 顶层目录说明

### 2.1 根目录

- `requirements.txt`
  项目依赖列表，包含 `torch`、`torchaudio`、`kaldiio`、`scikit-learn`、`pyyaml` 等。

- `.vscode/`
  编辑器配置目录，与模型训练逻辑无关。

- `egs/`
  放具体实验 recipe。当前仓库只保留了 `3dspeaker/sv-cam++` 这个实验目录。

- `speakerlab/`
  核心 Python 包。训练、推理、模型定义、数据集、损失函数都在这里。

---

## 3. `egs/3dspeaker/sv-cam++` 目录说明

这个目录是 CAM++ baseline 的实验入口。

### 3.1 文件与子目录

- `run.sh`
  主控脚本。按 stage 串起完整实验流程：
  `数据准备 -> 生成训练 CSV -> 模型训练 -> embedding 提取 -> 指标评测`

- `path.sh`
  设置运行环境变量。
  主要作用是把项目根目录加入 `PYTHONPATH`，保证 `speakerlab` 能被导入。

- `README.md`
  官方 recipe 说明，包含模型简介、公开结果、预训练模型链接。

- `Directory_Explanation.md`
  你当前仓库里的额外说明文档。内容方向是对 recipe 的中文解释，但当前终端读取时有编码问题。

- `speakerlab`
  一个路径指针文件，内容指向上层的 `speakerlab/` 包，方便 recipe 从该目录访问框架代码。

- `conf/`
  训练配置目录。当前核心配置文件是 `cam++.yaml`。

- `local/`
  recipe 专用的数据准备脚本目录。

- `utils/`
  recipe 依赖的 shell/perl 工具脚本目录。

- `data/`
  数据目录。当前会存放准备后的数据索引，以及你后来下载的 VCTK 数据。

### 3.2 `conf/` 目录

- `conf/cam++.yaml`
  当前 CAM++ baseline 的核心配置文件。
  它定义了：
  - 输入数据与增强方式
  - 数据集与 DataLoader
  - embedding 模型
  - 分类头
  - 优化器
  - 学习率调度器
  - margin 调度器
  - loss
  - checkpoint 保存对象

这个文件不是“静态参数表”，而是“对象装配清单”。  
训练脚本会根据其中的 `obj` 和 `args` 动态实例化模块。

### 3.3 `local/` 目录

- `local/download_data.sh`
  下载官方 baseline 使用的 3D-Speaker、musan、RIRS_NOISES 等数据压缩包。

- `local/prepare_data.sh`
  数据准备主脚本。
  分为三个阶段：
  - 下载压缩包
  - 解压数据
  - 生成 `wav.scp`、`utt2spk`、`spk2utt`、`trials_*`

- `local/prepare_data_csv.py`
  将 Kaldi 风格索引转换成训练使用的 `train.csv`。
  核心逻辑：
  - 读取 `wav.scp`
  - 读取 `utt2spk`
  - 检查采样率和声道数
  - 按固定时长切 chunk
  - 输出 `train.csv`

- `local/download_data_vctk.py`
  你新增的 VCTK 下载脚本。
  当前逻辑是：
  - 用 `kagglehub` 下载 VCTK 到默认缓存
  - 再复制到 `egs/3dspeaker/sv-cam++/data/`

### 3.4 `utils/` 目录

- `utils/parse_options.sh`
  解析 shell 命令行参数，允许用 `--stage`、`--gpus` 等覆盖 `run.sh` 里的默认变量。

- `utils/utt2spk_to_spk2utt.pl`
  把 `utt2spk` 转成 `spk2utt`。
  这是 Kaldi 风格数据准备中的常见工具。

- `utils/m4a2wav.pl`
  用于音频格式转换的辅助脚本，当前 CAM++ baseline 主流程并不依赖它。

---

## 4. `speakerlab/` 目录说明

`speakerlab` 是框架层，真正的训练和推理逻辑都在这里。

### 4.1 `speakerlab/bin/`

这是命令行入口脚本目录。

- `train.py`
  当前 CAM++ baseline 的主训练入口。
  它会：
  - 读取 YAML 配置
  - 构建数据集和 DataLoader
  - 构建 embedding 模型与分类头
  - 构建 optimizer、loss、scheduler
  - 启动分布式训练
  - 保存 checkpoint

- `extract.py`
  用训练好的 embedding 模型批量提取测试集 embedding，输出为 Kaldi ark/scp。

- `compute_score_metrics.py`
  读取 embedding 和 trial 列表，计算 cosine score、EER、minDCF。

- `infer_sv.py`
  单独做 speaker verification 推理用的脚本。
  可直接从 ModelScope 下载预训练模型，对单个或两个 wav 做 embedding 提取和相似度计算。

- `export_speaker_embedding_onnx.py`
  导出 speaker embedding 模型为 ONNX。

- `infer_sv_batch.py`
  批量 speaker verification 推理脚本。

- `infer_diarization.py`
  与说话人分离/聚类相关的推理入口。

- `extract_ssl.py`
- `infer_sv_ssl.py`
- `train_sdpn.py`
- `train_rdino.py`
- `train_para.py`
- `train_asd.py`
  这些脚本面向仓库里其他训练范式或自监督/半监督分支，不属于当前 CAM++ baseline 主链路。

### 4.2 `speakerlab/dataset/`

- `dataset.py`
  当前 CAM++ recipe 直接使用的数据集定义。
  核心类是 `WavSVDataset`，返回 `(feat, spkid)`。

- `dataset_asd.py`
- `dataset_rdino.py`
- `dataset_sdpn.py`
  其他任务或训练范式使用的数据集定义，不属于当前 CAM++ 主链路。

### 4.3 `speakerlab/models/campplus/`

这是当前模型的核心目录。

- `DTDNN.py`
  定义 `CAMPPlus` 主模型。
  核心结构：
  - `FCM` 前端卷积模块
  - TDNN 层
  - 多个 `CAMDenseTDNNBlock`
  - `TransitLayer`
  - `StatsPool`
  - 最终 `DenseLayer` 输出 speaker embedding

- `classifier.py`
  定义分类头。
  包含：
  - `CosineClassifier`
    当前 recipe 默认使用，输出归一化后的余弦相似度分类 logits
  - `LinearClassifier`
    普通线性分类头，当前默认未使用

- `layers.py`
  放 CAM++ 依赖的基础模块，例如：
  - `DenseLayer`
  - `StatsPool`
  - `TDNNLayer`
  - `CAMDenseTDNNBlock`
  - `TransitLayer`
  - `BasicResBlock`

### 4.4 `speakerlab/process/`

- `processor.py`
  当前 CAM++ recipe 的数据预处理核心模块。
  主要类：
  - `WavReader`：读 wav、裁固定时长、可做 speed perturb
  - `SpkLabelEncoder`：把说话人标签映射为类别 ID
  - `SpkVeriAug`：加噪、加混响增强
  - `FBank`：提取 80 维 fbank 特征

- `augmentation.py`
  具体的数据增强实现，供 `SpkVeriAug` 调用。

- `scheduler.py`
  学习率和 margin 调度器实现。
  当前 CAM++ recipe 用到了：
  - `WarmupCosineScheduler`
  - `MarginScheduler`

- `cluster.py`
- `processor_para.py`
  其他场景或并行处理相关模块，不属于当前 CAM++ 主训练链路。

### 4.5 `speakerlab/loss/`

- `margin_loss.py`
  当前 CAM++ recipe 直接使用的 loss 实现。
  包括：
  - `ArcMarginLoss`
  - `AddMarginLoss`
  - `EntropyLoss`

- `dino_loss.py`
- `keleo_loss.py`
- `sdpn_loss.py`
  其他训练范式使用的损失函数，不属于当前 CAM++ baseline 主链路。

### 4.6 `speakerlab/utils/`

- `config.py`
  读取 YAML 配置，并把命令行 override 合并到配置中。

- `builder.py`
  项目里非常关键的动态构建器。
  它会解析配置中的：
  - `obj: xxx.xxx.ClassName`
  - `<ref_name>` 引用
  然后递归构造对象。

- `checkpoint.py`
  负责 checkpoint 保存与恢复。

- `epoch.py`
  提供 epoch 计数和训练日志记录器。

- `fileio.py`
  负责读取 `wav.scp`、CSV 等文件。

- `score_metrics.py`
  负责计算 EER、DCF 等指标。

- `utils.py`
  常用工具函数，包括日志、平均统计、准确率、随机种子设置等。

- `utils_rdino.py`
  其他训练分支使用的工具函数。

---

## 5. 当前 CAM++ baseline 的主调用链

下面是当前最重要的一条链路。

### 5.1 实验入口层

`egs/3dspeaker/sv-cam++/run.sh`

按 stage 调用：

1. `local/prepare_data.sh`
2. `local/prepare_data_csv.py`
3. `speakerlab/bin/train.py`
4. `speakerlab/bin/extract.py`
5. `speakerlab/bin/compute_score_metrics.py`

### 5.2 数据准备层

`run.sh`
-> `local/prepare_data.sh`
-> `local/download_data.sh`

`prepare_data.sh` 最终生成：

- `data/musan/wav.scp`
- `data/rirs/wav.scp`
- `data/3dspeaker/train/wav.scp`
- `data/3dspeaker/train/utt2spk`
- `data/3dspeaker/test/wav.scp`
- `data/3dspeaker/test/utt2spk`
- `data/3dspeaker/trials/*`

然后：

`run.sh`
-> `local/prepare_data_csv.py`
-> 生成 `data/3dspeaker/train/train.csv`

### 5.3 训练层

`run.sh`
-> `speakerlab/bin/train.py`
-> `speakerlab/utils/config.py`
-> `speakerlab/utils/builder.py`
-> 根据 `conf/cam++.yaml` 动态构建对象

构建顺序可以概括为：

1. `dataset`
   - `speakerlab.dataset.dataset.WavSVDataset`
2. `preprocessor`
   - `WavReader`
   - `SpkLabelEncoder`
   - `SpkVeriAug`
   - `FBank`
3. `dataloader`
4. `embedding_model`
   - `speakerlab.models.campplus.DTDNN.CAMPPlus`
5. `classifier`
   - `speakerlab.models.campplus.classifier.CosineClassifier`
6. `optimizer`
   - `torch.optim.SGD`
7. `loss`
   - `speakerlab.loss.margin_loss.ArcMarginLoss`
8. `lr_scheduler`
   - `speakerlab.process.scheduler.WarmupCosineScheduler`
9. `margin_scheduler`
   - `speakerlab.process.scheduler.MarginScheduler`
10. `checkpointer`
    - `speakerlab.utils.checkpoint.Checkpointer`

### 5.4 训练时单个 batch 的调用链

`DataLoader`
-> `WavSVDataset.__getitem__`
-> `WavReader`
-> `SpkLabelEncoder`
-> `SpkVeriAug`
-> `FBank`
-> 返回 `(feat, spkid)`

然后进入模型：

`feat`
-> `CAMPPlus`
-> `CosineClassifier`
-> `ArcMarginLoss`
-> `backward()`
-> `optimizer.step()`

### 5.5 提取 embedding 层

`run.sh`
-> `speakerlab/bin/extract.py`
-> 读取 `exp_dir/config.yaml`
-> 构建 `embedding_model`
-> 从 `models/` 恢复 checkpoint
-> 读取测试集 `wav.scp`
-> 提取 embedding
-> 输出到 `exp_dir/embeddings/*.ark` 和 `*.scp`

### 5.6 评测层

`run.sh`
-> `speakerlab/bin/compute_score_metrics.py`
-> 读取 `embeddings/*.ark`
-> 读取 `trials_*`
-> 计算 cosine similarity
-> 输出：
  - 每个 trial 的 `.score`
  - 汇总指标 `result.metrics`

---

## 6. 当前 CAM++ 模型内部层级

以 `speakerlab/models/campplus/DTDNN.py` 为主：

1. 输入特征形状：`[B, T, F]`
2. 转置为：`[B, F, T]`
3. 进入 `FCM`
   - 2D 卷积
   - 残差块
   - 压缩频率维
4. reshape 成时序特征
5. 进入 `TDNNLayer`
6. 进入 3 个 `CAMDenseTDNNBlock + TransitLayer`
7. 进入 `StatsPool`
   - 聚合时序统计量
8. 进入最终 `DenseLayer`
9. 输出 speaker embedding

训练时 embedding 再接：

`embedding`
-> `CosineClassifier`
-> `ArcMarginLoss`

推理时通常只保留：

`wav -> FBank -> CAMPPlus -> embedding`

---

## 7. 哪些文件是“当前主链路必须读”的

如果你是为了后续改 baseline，建议优先读这些文件：

1. `egs/3dspeaker/sv-cam++/run.sh`
2. `egs/3dspeaker/sv-cam++/conf/cam++.yaml`
3. `speakerlab/bin/train.py`
4. `speakerlab/dataset/dataset.py`
5. `speakerlab/process/processor.py`
6. `speakerlab/models/campplus/DTDNN.py`
7. `speakerlab/models/campplus/classifier.py`
8. `speakerlab/loss/margin_loss.py`
9. `speakerlab/process/scheduler.py`
10. `speakerlab/bin/extract.py`
11. `speakerlab/bin/compute_score_metrics.py`

---

## 8. 哪些文件是“当前 CAM++ baseline 可以后读”的

这些文件存在，但不属于当前主实验链路：

- `speakerlab/bin/train_asd.py`
- `speakerlab/bin/train_para.py`
- `speakerlab/bin/train_rdino.py`
- `speakerlab/bin/train_sdpn.py`
- `speakerlab/bin/extract_ssl.py`
- `speakerlab/bin/infer_sv_ssl.py`
- `speakerlab/dataset/dataset_asd.py`
- `speakerlab/dataset/dataset_rdino.py`
- `speakerlab/dataset/dataset_sdpn.py`
- `speakerlab/loss/dino_loss.py`
- `speakerlab/loss/keleo_loss.py`
- `speakerlab/loss/sdpn_loss.py`
- `speakerlab/process/cluster.py`
- `speakerlab/process/processor_para.py`
- `speakerlab/utils/utils_rdino.py`

这些更像是框架里保留下来的其他训练分支。

---

## 9. 你后续做毕业设计时最可能改动的位置

### 9.1 改实验配置

优先改：

- `egs/3dspeaker/sv-cam++/conf/cam++.yaml`

适合修改的内容：

- `embedding_size`
- `batch_size`
- `lr`
- `augmentations`
- `classifier`
- `loss`
- `scheduler`

### 9.2 改数据来源或数据格式

优先改：

- `egs/3dspeaker/sv-cam++/local/prepare_data.sh`
- `egs/3dspeaker/sv-cam++/local/prepare_data_csv.py`
- `egs/3dspeaker/sv-cam++/local/download_data_vctk.py`

### 9.3 改模型结构

优先改：

- `speakerlab/models/campplus/DTDNN.py`
- `speakerlab/models/campplus/classifier.py`
- `speakerlab/models/campplus/layers.py`

### 9.4 改训练目标

优先改：

- `speakerlab/loss/margin_loss.py`
- `speakerlab/process/scheduler.py`
- `speakerlab/bin/train.py`

---

## 10. 一句话总结

这个仓库的核心关系可以概括为：

`run.sh` 负责组织实验流程，`cam++.yaml` 负责决定训练时实例化哪些模块，`speakerlab/` 负责真正执行数据处理、模型前向、loss 计算、checkpoint 保存和最终评测。
