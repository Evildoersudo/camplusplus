# GPU 服务器运行手册

> 本文档为历史记录。当前统一执行版本请使用 `note/container_linux_runbook.md`（容器内 Linux 命令）。

## 1. 文档用途

本文档用于记录当前项目在 Linux GPU 服务器上的常用运行命令，方便后续重复调试、训练、续训与评测。

当前机器的约束是：

- 代码只能放在：`/root/my_project/camplusplus`
- 数据、解压结果、混合音频、训练输出都应放在：`/root/autodl-tmp`

因此本文档统一采用“代码和数据完全分离”的运行方式。

## 2. 路径约定

建议固定使用以下目录：

- 项目根目录：`/root/my_project/camplusplus`
- 下载数据目录：`/root/autodl-tmp/download_data`
- 数据工作区根目录：`/root/autodl-tmp/camplusplus_data`
- 训练输出根目录：`/root/autodl-tmp/camplusplus_exp`
- 预训练模型目录：`/root/autodl-tmp/pretrained`

建议先手动创建：

```bash
mkdir -p /root/autodl-tmp/download_data
mkdir -p /root/autodl-tmp/camplusplus_data
mkdir -p /root/autodl-tmp/camplusplus_exp
mkdir -p /root/autodl-tmp/pretrained
```

## 3. 环境准备

### 3.1 进入项目目录

```bash
cd /root/my_project/camplusplus
```

### 3.2 创建 Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3.3 安装项目依赖

```bash
pip install -r requirements.txt
pip install soundfile
```

### 3.4 安装 GPU 版 PyTorch

如果服务器使用 CUDA 12.1，可执行：

```bash
pip uninstall -y torch torchaudio
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3.5 检查 GPU 是否可用

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

## 4. 数据准备

### 4.1 说明

当前数据准备脚本：

- `egs/3dspeaker/sv-cam++/local/prepare_cnceleb_mixed_data.py`

该脚本会完成：

1. 使用已经下载好的 `CN-Celeb`、`musan`、`rirs`
2. 生成 clean 训练集索引
3. 生成 mixed codec 训练集
4. 生成测试集索引
5. 生成 `musan/wav.scp`
6. 生成 `rirs/wav.scp`
7. 生成 `train.csv`

### 4.2 生成混合训练数据

注意：这里的 `data_root` 和 `raw_root` 都放到 `/root/autodl-tmp`。

#### 4.2.1 首次全量生成（覆盖模式）

使用 `--overwrite` 会强制重生成已存在的 degraded 音频，适合首次跑或你确认要全量重做。

```bash
cd /root/my_project/camplusplus

python egs/3dspeaker/sv-cam++/local/prepare_cnceleb_mixed_data.py \
  --download_dir /root/autodl-tmp/download_data \
  --data_root /root/autodl-tmp/camplusplus_data \
  --workspace_name CN_celeb_database \
  --raw_root /root/autodl-tmp/camplusplus_data/raw_data \
  --clean_ratio 0.3 \
  --opus_ratio 0.3 \
  --amrwb_ratio 0.2 \
  --g711_ratio 0.2 \
  --opus_bitrates 4k,6k,8k \
  --amrwb_bitrates 8.85k,8.85k,8.85k,12.65k,23.85k \
  --g711_variants g711_mulaw,g711_alaw \
  --sample_rate 16000 \
  --num_workers 8 \
  --prepare_csv_nj 8 \
  --overwrite
```

#### 4.2.2 意外中断后续跑（推荐）

如果 mixed codec 音频已经生成了一部分，续跑时去掉 `--overwrite`，脚本会复用已存在文件，只补齐缺失部分。

```bash
cd /root/my_project/camplusplus

python egs/3dspeaker/sv-cam++/local/prepare_cnceleb_mixed_data.py \
  --download_dir /root/autodl-tmp/download_data \
  --data_root /root/autodl-tmp/camplusplus_data \
  --workspace_name CN_celeb_database \
  --raw_root /root/autodl-tmp/camplusplus_data/raw_data \
  --clean_ratio 0.3 \
  --opus_ratio 0.3 \
  --amrwb_ratio 0.2 \
  --g711_ratio 0.2 \
  --opus_bitrates 4k,6k,8k \
  --amrwb_bitrates 8.85k,8.85k,8.85k,12.65k,23.85k \
  --g711_variants g711_mulaw,g711_alaw \
  --sample_rate 16000 \
  --num_workers 8 \
  --prepare_csv_nj 8
```

续跑注意事项：

- 保持 `clean_ratio / opus_ratio / amrwb_ratio / g711_ratio` 不变
- 保持 `opus_bitrates / amrwb_bitrates / g711_variants` 不变
- 保持 `seed` 不变（默认是 42）

可用下面命令快速查看当前 mixed 音频数量：

```bash
find /root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb_mixed_audio -name "*.wav" | wc -l
```

### 4.3 生成后的关键目录

- Clean 训练集：
  - `/root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb/clean_train`
- Mixed 训练集：
  - `/root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb_mixed/train`
- Mixed 音频根目录：
  - `/root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb_mixed_audio`
- 测试集：
  - `/root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb/test`
- Trials：
  - `/root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb/trials`
- MUSAN：
  - `/root/autodl-tmp/camplusplus_data/CN_celeb_database/musan/wav.scp`
- RIRS：
  - `/root/autodl-tmp/camplusplus_data/CN_celeb_database/rirs/wav.scp`

### 4.4 从本地上传已生成数据后的路径修复

如果你在本地已经生成了 `CN_celeb_database`，再打包上传到服务器，这个方案完全可行。

需要额外做的一步是：把索引文件中的本地绝对路径批量改成服务器路径。

#### 4.4.1 典型需要改写的文件

- `cnceleb/clean_train/wav.scp`
- `cnceleb/test/wav.scp`
- `cnceleb_mixed/train/wav.scp`
- `cnceleb_mixed/train/train.csv`
- `musan/wav.scp`
- `rirs/wav.scp`

其中：

- `utt2spk` 和 `spk2utt` 不含文件路径，一般不需要改
- `train.csv` 只需要改 `path` 列

#### 4.4.2 一键改写脚本（推荐）

下面脚本会把 Windows 路径前缀映射到服务器路径，并统一路径分隔符为 `/`。

```bash
cd /root/my_project/camplusplus

python - <<'PY'
from pathlib import Path
import csv

base = Path('/root/autodl-tmp/camplusplus_data/CN_celeb_database')

# 按你的本地实际路径调整旧前缀（可保留多个候选）
prefix_map = {
  'E:/Speaker_recognition/Graduation_Project/camplusplus/egs/3dspeaker/sv-cam++/data/raw_data': '/root/autodl-tmp/camplusplus_data/raw_data',
  'E:/Speaker_recognition/Graduation_Project/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database': '/root/autodl-tmp/camplusplus_data/CN_celeb_database',
}

wav_scp_files = [
  base / 'cnceleb/clean_train/wav.scp',
  base / 'cnceleb/test/wav.scp',
  base / 'cnceleb_mixed/train/wav.scp',
  base / 'musan/wav.scp',
  base / 'rirs/wav.scp',
]
train_csv = base / 'cnceleb_mixed/train/train.csv'

def rewrite_path(path: str) -> str:
  p = path.replace('\\\\', '/')
  for old, new in prefix_map.items():
    old_norm = old.replace('\\\\', '/')
    if p.startswith(old_norm):
      return new + p[len(old_norm):]
  return p

for f in wav_scp_files:
  if not f.exists():
    continue
  out = []
  with f.open('r', encoding='utf-8') as fin:
    for line in fin:
      line = line.strip()
      if not line:
        continue
      utt, path = line.split(maxsplit=1)
      out.append(f"{utt} {rewrite_path(path)}")
  with f.open('w', encoding='utf-8', newline='\n') as fout:
    fout.write('\n'.join(out) + ('\n' if out else ''))
  print(f'rewritten wav.scp: {f}')

if train_csv.exists():
  rows = []
  with train_csv.open('r', encoding='utf-8', newline='') as fin:
    reader = csv.reader(fin)
    header = next(reader)
    rows.append(header)
    path_idx = header.index('path')
    for row in reader:
      row[path_idx] = rewrite_path(row[path_idx])
      rows.append(row)
  with train_csv.open('w', encoding='utf-8', newline='') as fout:
    writer = csv.writer(fout)
    writer.writerows(rows)
  print(f'rewritten train.csv: {train_csv}')
PY
```

#### 4.4.3 改写后快速自检

```bash
head -n 3 /root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb_mixed/train/wav.scp
head -n 3 /root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb_mixed/train/train.csv
```

如果输出路径都以 `/root/autodl-tmp/...` 开头，说明索引修复完成。

## 5. 训练配置建议

当前服务器上建议优先修改：

- `egs/3dspeaker/sv-cam++/conf/cam++.yaml`

推荐设置：

```yaml
batch_size: 64
num_workers: 8
save_epoch_freq: 1
aug_prob: 0.8
```

如果显存不足：

- 将 `batch_size` 改为 `32`

如果显存较大：

- 可尝试 `batch_size: 128`

## 6. 预训练模型存放建议

为了避免占用代码目录空间，建议把预训练权重也放在 `/root/autodl-tmp`。

例如：

- `/root/autodl-tmp/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin`

该权重对应：

- `embedding_size = 512`

因此可以直接配合：

- `egs/3dspeaker/sv-cam++/conf/cam++.yaml`

## 7. 微调训练

### 7.1 Mixed 数据微调

训练输出目录也放在 `/root/autodl-tmp/camplusplus_exp`。

```bash
cd /root/my_project/camplusplus

python -m speakerlab.bin.train \
  --config /root/my_project/camplusplus/egs/3dspeaker/sv-cam++/conf/cam++.yaml \
  --gpu 0 \
  --init_model /root/autodl-tmp/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --data /root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb_mixed/train/train.csv \
  --noise /root/autodl-tmp/camplusplus_data/CN_celeb_database/musan/wav.scp \
  --reverb /root/autodl-tmp/camplusplus_data/CN_celeb_database/rirs/wav.scp \
  --aug_prob 0.8 \
  --exp_dir /root/autodl-tmp/camplusplus_exp/campp_cnceleb_mixed_ft
```

### 7.2 Clean 数据微调

如果你要做对照实验，只用 clean 训练集：

```bash
cd /root/my_project/camplusplus

python -m speakerlab.bin.train \
  --config /root/my_project/camplusplus/egs/3dspeaker/sv-cam++/conf/cam++.yaml \
  --gpu 0 \
  --init_model /root/autodl-tmp/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --data /root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb/clean_train/train.csv \
  --noise /root/autodl-tmp/camplusplus_data/CN_celeb_database/musan/wav.scp \
  --reverb /root/autodl-tmp/camplusplus_data/CN_celeb_database/rirs/wav.scp \
  --aug_prob 0.8 \
  --exp_dir /root/autodl-tmp/camplusplus_exp/campp_cnceleb_clean_ft
```

## 8. 断点续训

当前训练脚本支持自动续训。

只要重新执行**相同的训练命令**，并保持：

- `exp_dir` 不变

它就会自动从该目录下最新的 checkpoint 继续训练。

例如：

```bash
cd /root/my_project/camplusplus

python -m speakerlab.bin.train \
  --config /root/my_project/camplusplus/egs/3dspeaker/sv-cam++/conf/cam++.yaml \
  --gpu 0 \
  --init_model /root/autodl-tmp/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --data /root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb_mixed/train/train.csv \
  --noise /root/autodl-tmp/camplusplus_data/CN_celeb_database/musan/wav.scp \
  --reverb /root/autodl-tmp/camplusplus_data/CN_celeb_database/rirs/wav.scp \
  --aug_prob 0.8 \
  --exp_dir /root/autodl-tmp/camplusplus_exp/campp_cnceleb_mixed_ft
```

## 9. 提取测试集 embedding

训练完成后先提取测试集 embedding：

```bash
cd /root/my_project/camplusplus

python -m speakerlab.bin.extract \
  --exp_dir /root/autodl-tmp/camplusplus_exp/campp_cnceleb_mixed_ft \
  --data /root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb/test/wav.scp \
  --use_gpu \
  --gpu 0
```

如果是 clean 对照模型，则改成：

```bash
cd /root/my_project/camplusplus

python -m speakerlab.bin.extract \
  --exp_dir /root/autodl-tmp/camplusplus_exp/campp_cnceleb_clean_ft \
  --data /root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb/test/wav.scp \
  --use_gpu \
  --gpu 0
```

## 10. 计算测试集分数

先进入 trials 目录看有哪些官方协议文件：

```bash
ls /root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb/trials
```

然后选择实际使用的 trial 文件，例如：

```bash
cd /root/my_project/camplusplus

python -m speakerlab.bin.compute_score_metrics \
  --enrol_data /root/autodl-tmp/camplusplus_exp/campp_cnceleb_mixed_ft/embeddings \
  --test_data /root/autodl-tmp/camplusplus_exp/campp_cnceleb_mixed_ft/embeddings \
  --scores_dir /root/autodl-tmp/camplusplus_exp/campp_cnceleb_mixed_ft/scores \
  --trials /root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb/trials/<trial_file>
```

## 11. 常用检查命令

### 11.1 检查训练集文件是否生成

```bash
ls /root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb_mixed/train
```

### 11.2 检查 train.csv 行数

```bash
wc -l /root/autodl-tmp/camplusplus_data/CN_celeb_database/cnceleb_mixed/train/train.csv
```

### 11.3 检查 checkpoint 是否保存

```bash
ls /root/autodl-tmp/camplusplus_exp/campp_cnceleb_mixed_ft/models
```

### 11.4 查看训练日志

```bash
tail -f /root/autodl-tmp/camplusplus_exp/campp_cnceleb_mixed_ft/train.log
```

## 12. 推荐实验顺序

建议按下面顺序跑：

1. 先生成 mixed 数据
2. 先跑 `clean_train` 微调
3. 再跑 `mixed_train` 微调
4. 分别提 embedding
5. 分别在同一份 trials 上打分
6. 对比：
   - clean EER
   - degraded EER
   - 不同 codec 下的鲁棒性变化

## 13. 关键原则

在这台服务器上，后续都按下面的原则执行：

- `/root/my_project/camplusplus` 只放代码
- `/root/autodl-tmp` 放：
  - 下载数据
  - 解压数据
  - mixed 音频
  - 训练索引
  - checkpoint
  - embeddings
  - scores
  - 预训练模型

不要把大文件写回代码目录，否则很快会占满系统盘。
