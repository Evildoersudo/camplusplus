# `sv-cam++` 目录说明

本文档用于解释 [run.sh](e:/Speaker_recognition/Graduation_Project/3D-Speaker/egs/3dspeaker/sv-cam++/run.sh) 以及 `egs/3dspeaker/sv-cam++` 目录下各文件的作用，方便后续作为毕业设计中的 `CAM++ baseline` 工程入口。

## 1. 目录定位

`egs/3dspeaker/sv-cam++` 是 3D-Speaker 项目中基于 `CAM++` 的说话人验证 recipe。

它的主要职责不是定义模型本身，而是把一整条实验链路串起来：

- 下载和整理数据
- 生成训练索引
- 调用通用训练脚本训练 CAM++
- 提取测试 embedding
- 计算 EER / minDCF

因此，这个目录更像是一个 `实验入口目录`，而不是模型源码目录。

## 2. 目录结构总览

### 2.1 顶层文件

| 文件/目录 | 作用 |
|---|---|
| `run.sh` | 主控脚本，按 stage 串起数据准备、训练、提取 embedding、评测 |
| `README.md` | 该 recipe 的简要说明、公开结果、预训练模型说明 |
| `path.sh` | 配置运行环境变量，确保脚本能找到项目根目录下的 Python 包 |
| `speakerlab` | 一个文本指针文件，内容为 `../../../speakerlab/`，用于从 recipe 目录访问项目主代码 |
| `conf/` | 训练配置文件目录 |
| `local/` | 当前 recipe 专用的数据准备脚本 |
| `utils/` | 通用辅助脚本，如参数解析、`utt2spk` 转 `spk2utt` |

### 2.2 子目录职责

| 子目录 | 作用 |
|---|---|
| `conf/` | 定义训练参数、数据处理模块、模型对象、优化器、损失函数等 |
| `local/` | 负责下载数据、解压数据、生成 `wav.scp` / `utt2spk` / `train.csv` |
| `utils/` | 提供 shell/perl 工具脚本，支撑 recipe 运行 |

## 3. `run.sh` 详细解读

文件位置：
[run.sh](e:/Speaker_recognition/Graduation_Project/3D-Speaker/egs/3dspeaker/sv-cam++/run.sh)

这个脚本是整个 baseline 的总入口。

### 3.1 开头部分

```bash
set -e
. ./path.sh || exit 1

stage=1
stop_stage=5

data=data
exp=exp
exp_name=cam++
gpus="0 1 2 3"
```

这里定义了几个核心控制项：

- `set -e`
  只要某一步报错，脚本就立即退出，避免后续阶段在错误状态下继续执行。

- `. ./path.sh`
  加载环境变量，确保当前目录和项目根目录的 `speakerlab` Python 包可被找到。

- `stage` 和 `stop_stage`
  用于控制运行到哪一步。比如只想做训练，可以设置 `stage=3 stop_stage=3`。

- `data=data`
  数据输出目录根路径。最终整理好的 `wav.scp`、`train.csv` 等都放在这个目录下。

- `exp=exp`
  实验输出根目录。

- `exp_name=cam++`
  当前实验名，最终实验路径是 `exp/cam++`。

- `gpus="0 1 2 3"`
  指定训练和提取 embedding 使用的 GPU 编号。

### 3.2 参数覆盖机制

```bash
. utils/parse_options.sh || exit 1
```

这行允许你用命令行覆盖上面的变量，例如：

```bash
bash run.sh --stage 3 --stop_stage 3 --gpus "0"
```

这是 Kaldi 风格 recipe 常见写法。

### 3.3 实验目录

```bash
exp_dir=$exp/$exp_name
```

最终模型、日志、embedding、分数文件都存到这里，例如：

- `exp/cam++/models`
- `exp/cam++/embeddings`
- `exp/cam++/scores`

### 3.4 Stage 1: 数据下载与基础准备

```bash
./local/prepare_data.sh --stage 1 --stop_stage 3 --data ${data}
```

这里会调用本目录下的 [prepare_data.sh](e:/Speaker_recognition/Graduation_Project/3D-Speaker/egs/3dspeaker/sv-cam++/local/prepare_data.sh)，完成：

- 下载 `musan`
- 下载 `RIRS_NOISES`
- 下载 `3D-Speaker train/test`
- 解压归档
- 整理出训练和测试所需的 Kaldi 风格文件

输出的关键文件包括：

- `data/3dspeaker/train/wav.scp`
- `data/3dspeaker/train/utt2spk`
- `data/3dspeaker/test/wav.scp`
- `data/3dspeaker/trials/*`
- `data/musan/wav.scp`
- `data/rirs/wav.scp`

### 3.5 Stage 2: 生成训练 CSV

```bash
python local/prepare_data_csv.py --data_dir $data/3dspeaker/train
```

这里调用 [prepare_data_csv.py](e:/Speaker_recognition/Graduation_Project/3D-Speaker/egs/3dspeaker/sv-cam++/local/prepare_data_csv.py)，把 `wav.scp + utt2spk` 转成训练脚本使用的 `train.csv`。

这个 CSV 每一行包含：

- 片段 ID
- 音频总时长
- 文件路径
- 片段起点
- 片段终点
- 说话人 ID

默认按 `3 秒` 分块，把长语音拆成多个训练样本。

### 3.6 Stage 3: 训练 CAM++

```bash
torchrun --nproc_per_node=$num_gpu speakerlab/bin/train.py --config conf/cam++.yaml --gpu $gpus \
         --data $data/3dspeaker/train/train.csv --noise $data/musan/wav.scp --reverb $data/rirs/wav.scp --exp_dir $exp_dir
```

这是最关键的一步。

它调用项目公共训练入口：
[train.py](e:/Speaker_recognition/Graduation_Project/3D-Speaker/speakerlab/bin/train.py)

并加载配置文件：
[cam++.yaml](e:/Speaker_recognition/Graduation_Project/3D-Speaker/egs/3dspeaker/sv-cam++/conf/cam++.yaml)

传入的几个动态参数分别是：

- `--data`
  指向 `train.csv`

- `--noise`
  指向 `musan` 噪声列表

- `--reverb`
  指向 `RIRS` 混响列表

- `--exp_dir`
  指定实验输出目录

这里的训练逻辑本质是：

1. 读取 wav
2. 做 speed perturb、加噪、加混响
3. 提取 80 维 fbank
4. 输入 CAM++ 提取 embedding
5. 接余弦分类头做说话人分类训练
6. 用 ArcMarginLoss 优化

### 3.7 Stage 4: 提取测试 embedding

```bash
torchrun --nproc_per_node=8 speakerlab/bin/extract.py --exp_dir $exp_dir \
         --data $data/3dspeaker/test/wav.scp --use_gpu --gpu $gpus
```

这一步用训练好的模型对测试集所有语音提取 embedding。

输出一般在：

- `exp/cam++/embeddings`

后续评测并不直接重新跑模型，而是基于提取好的 embedding 计算分数。

### 3.8 Stage 5: 计算 EER / minDCF

```bash
python speakerlab/bin/compute_score_metrics.py --enrol_data $exp_dir/embeddings --test_data $exp_dir/embeddings \
                                               --scores_dir $exp_dir/scores --trials $trials
```

这里读取 trials 文件，对 enrol/test embedding 两两做 cosine 相似度，最后输出：

- `EER`
- `minDCF`

评测覆盖三种测试设置：

- `trials_cross_device`
- `trials_cross_distance`
- `trials_cross_dialect`

这也是你后续做 codec 鲁棒性研究时最应该保留的 baseline 评测风格。

## 4. `conf/cam++.yaml` 说明

文件位置：
[cam++.yaml](e:/Speaker_recognition/Graduation_Project/3D-Speaker/egs/3dspeaker/sv-cam++/conf/cam++.yaml)

这是 baseline 的核心配置文件，定义了训练过程中“构建什么对象、用什么参数”。

### 4.1 基础超参数

- `num_epoch: 60`
- `wav_len: 3.0`
- `sample_rate: 16000`
- `batch_size: 256`
- `fbank_dim: 80`
- `embedding_size: 512`
- `lr: 0.1`
- `min_lr: 1e-4`

### 4.2 数据处理链

配置中定义了以下模块：

- `wav_reader`
  负责读取音频，并按 `3 秒` 取片段，可做 speed perturb

- `feature_extractor`
  提取 `FBank` 特征，`n_mels=80`，开启均值归一化

- `augmentations`
  使用 `SpkVeriAug` 做噪声和混响增强

- `dataset`
  使用 `WavSVDataset`

- `dataloader`
  使用 PyTorch `DataLoader`

### 4.3 模型与分类头

```yaml
embedding_model:
  obj: speakerlab.models.campplus.DTDNN.CAMPPlus
```

说明 embedding 提取器是 `CAMPPlus`，模型源码在：
[DTDNN.py](e:/Speaker_recognition/Graduation_Project/3D-Speaker/speakerlab/models/campplus/DTDNN.py)

```yaml
classifier:
  obj: speakerlab.models.campplus.classifier.CosineClassifier
```

说明训练时使用的是余弦分类器。

### 4.4 损失与调度

- `optimizer`: SGD + momentum + nesterov
- `lr_scheduler`: WarmupCosineScheduler
- `loss`: ArcMarginLoss
- `margin_scheduler`: 动态调整 margin

这套组合是典型的说话人验证分类训练范式。

## 5. `local/` 目录说明

### 5.1 `download_data.sh`

文件位置：
[download_data.sh](e:/Speaker_recognition/Graduation_Project/3D-Speaker/egs/3dspeaker/sv-cam++/local/download_data.sh)

职责：

- 下载 `musan.tar.gz`
- 下载 `rirs_noises.zip`
- 下载 `3D-Speaker` 的 `train.tar.gz`、`test.tar.gz`
- 下载 `3dspeaker_files.tar.gz`
- 做 md5 校验

这个脚本只负责下载，不负责解压和整理。

### 5.2 `prepare_data.sh`

文件位置：
[prepare_data.sh](e:/Speaker_recognition/Graduation_Project/3D-Speaker/egs/3dspeaker/sv-cam++/local/prepare_data.sh)

这是数据准备主脚本，按 stage 分三步：

1. `stage 1`
   调用 `download_data.sh` 下载数据包

2. `stage 2`
   解压数据包到 `data/raw_data`

3. `stage 3`
   生成训练和测试所需的元数据文件

最终生成：

- `wav.scp`
- `utt2spk`
- `spk2utt`
- `trials_*`

其中有一个重要细节：

```bash
grep -v "Device09" ...
```

训练集和测试集都显式排除了 `Device09` 的数据。这意味着官方 baseline 不是直接用所有原始条目，而是按作者规定过滤了一部分数据。

### 5.3 `prepare_data_csv.py`

文件位置：
[prepare_data_csv.py](e:/Speaker_recognition/Graduation_Project/3D-Speaker/egs/3dspeaker/sv-cam++/local/prepare_data_csv.py)

职责是把 Kaldi 风格索引转成训练 CSV。

它做了几件事：

- 读取 `wav.scp`
- 读取 `utt2spk`
- 检查采样率是否为 `16k`
- 检查是否单声道
- 将语音按固定长度切块
- 生成 `train.csv`

对你毕设很重要的点有两个：

1. 这里是以后接入 `codec 数据版本` 的一个关键入口
   如果你要把 clean / opus / amrwb 全部纳入训练，可以从这里或它的上游数据索引逻辑入手。

2. 这里的输出格式决定了 `train.py` 能否直接复用
   所以你后面做多条件训练，优先保证 CSV 格式不变，而不是去改训练器。

## 6. `utils/` 目录说明

### 6.1 `parse_options.sh`

文件位置：
[parse_options.sh](e:/Speaker_recognition/Graduation_Project/3D-Speaker/egs/3dspeaker/sv-cam++/utils/parse_options.sh)

作用是解析命令行参数，例如：

```bash
bash run.sh --stage 3 --stop_stage 3 --exp_name campp_thesis
```

它会把：

- `--stage` 映射为 shell 变量 `stage`
- `--stop-stage` 映射为 `stop_stage`

这是 recipe 灵活运行的关键工具。

### 6.2 `utt2spk_to_spk2utt.pl`

作用是把：

- `utt2spk`: 每条语音属于哪个说话人

转换为：

- `spk2utt`: 每个说话人有哪些语音

这是 Kaldi 数据组织中的标准辅助脚本。

### 6.3 `m4a2wav.pl`

文件位置：
[m4a2wav.pl](e:/Speaker_recognition/Graduation_Project/3D-Speaker/egs/3dspeaker/sv-cam++/utils/m4a2wav.pl)

这个脚本并不直接参与当前 `3D-Speaker + CAM++` 训练流程。

它主要是一个通用的音频格式转换工具，用 `ffmpeg` 把 `m4a` 转成 `wav`，更像是复用工具，常见于 VoxCeleb 数据准备场景。

在你的毕业设计当前路线里，这个脚本不是主入口。

## 7. `path.sh` 说明

文件位置：
[path.sh](e:/Speaker_recognition/Graduation_Project/3D-Speaker/egs/3dspeaker/sv-cam++/path.sh)

内容很短，但很关键：

```bash
export PATH=$PWD:$PATH
export PYTHONPATH=../../../:$PYTHONPATH
export OMP_NUM_THREADS=1
```

含义：

- 把当前 recipe 目录加入 `PATH`
- 把项目根目录加入 `PYTHONPATH`
- 限制 OpenMP 线程数，避免 CPU 线程过多

没有这一步，`speakerlab/bin/train.py` 可能无法正确 import 项目源码。

## 8. `README.md` 说明

文件位置：
[README.md](e:/Speaker_recognition/Graduation_Project/3D-Speaker/egs/3dspeaker/sv-cam++/README.md)

这个文件主要提供：

- baseline 训练配置概览
- 在 3D-Speaker 测试集上的公开结果
- ModelScope 预训练模型链接
- 论文引用信息

它不是运行依赖文件，但很适合用于核对 baseline 目标指标。

## 9. `speakerlab` 文件说明

文件位置：
[speakerlab](e:/Speaker_recognition/Graduation_Project/3D-Speaker/egs/3dspeaker/sv-cam++/speakerlab)

这个文件内容是：

```text
../../../speakerlab/
```

它本身不是模型代码，只是一个路径指向，表明真正的训练、推理、模型定义都在项目根目录的 `speakerlab/` 中。

因此：

- recipe 目录负责编排实验流程
- 根目录 `speakerlab/` 负责提供模型、数据处理、训练器、评测器

## 10. 这个目录和你的毕业设计的关系

对你当前的毕设路线，这个目录可以视为 `baseline 总控目录`。

你后续最可能改动的是以下位置：

### 10.1 建议保留不动的部分

- `run.sh` 的整体 stage 结构
- `prepare_data.sh` 的 clean 数据准备逻辑
- `compute_score_metrics.py` 的 EER/minDCF 评测逻辑

### 10.2 建议复制后再改的部分

- `conf/cam++.yaml`
  建议复制为你自己的：
  `campp_thesis_baseline.yaml`
  `campp_thesis_multicond.yaml`
  `campp_thesis_distill.yaml`

### 10.3 你毕业设计最可能新增的部分

- `local/make_codec_data.py`
  生成 Opus / AMR-WB 编码后的 wav 数据

- `local/make_codec_trials.py`
  生成 clean-codec、codec-codec、cross-codec 评测协议

- 新的训练入口
  如 `speakerlab/bin/train_distill.py`

## 11. 一句话总结

`sv-cam++` 目录本质上是一个 CAM++ 说话人验证 baseline 的实验配方目录。

它负责把：

`数据准备 -> 训练 -> embedding 提取 -> 指标评测`

完整串起来。

对你的毕业设计来说，最合理的做法不是重写这个目录，而是在保留其 baseline 结构的前提下，新增：

- thesis 专用配置
- codec 数据生成脚本
- multi-condition 训练版本
- teacher-student 蒸馏版本

这样可以最大限度保证 baseline 清晰、对比公平、实验可复现。
