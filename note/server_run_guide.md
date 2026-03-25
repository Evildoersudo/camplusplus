# GPU 服务器运行手册

## 1. 文档用途

本文档记录当前项目在 Linux GPU 服务器上的常用运行命令，目标是让仓库可以在不同机器上直接复用，而不依赖固定的机器路径。

默认做法：

- 仓库路径不写死，统一用 `PROJECT_ROOT`
- 数据、实验输出、预训练权重等运行时目录统一放在 `PROJECT_ROOT` 下的相对目录
- 如果你想把大文件放到别处，只需要改开头的环境变量，不需要改后面的命令

## 2. 路径约定

先进入仓库根目录，然后统一导出路径变量：

```bash
cd "$(git rev-parse --show-toplevel)"

export PROJECT_ROOT="$(pwd)"
export WORKSPACE_ROOT="../camplusplus_runtime"
export DOWNLOAD_DIR="${WORKSPACE_ROOT}/download_data"
export DATA_ROOT="${WORKSPACE_ROOT}/camplusplus_data"
export EXP_ROOT="${WORKSPACE_ROOT}/camplusplus_exp"
export PRETRAINED_ROOT="${WORKSPACE_ROOT}/pretrained"
export RECIPE_DIR="${PROJECT_ROOT}/egs/3dspeaker/sv-cam++"
export WORKSPACE_NAME="CN_celeb_database"
```

建议先创建这些目录：

```bash
mkdir -p "${DOWNLOAD_DIR}"
mkdir -p "${DATA_ROOT}"
mkdir -p "${EXP_ROOT}"
mkdir -p "${PRETRAINED_ROOT}"
```

如果你的服务器不适合把大文件放在仓库目录内，只需要把 `WORKSPACE_ROOT` 改成你自己的挂载目录，例如：

```bash
export WORKSPACE_ROOT="../camplusplus_runtime"
```

后面的命令不用再改。

## 3. 环境准备

### 3.1 进入项目目录

```bash
cd "${PROJECT_ROOT}"
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

注意：这里的 `data_root` 和 `raw_root` 都基于前面定义的相对目录变量。

#### 4.2.1 首次全量生成

```bash
cd "${PROJECT_ROOT}"

python "${RECIPE_DIR}/local/prepare_cnceleb_mixed_data.py" \
  --download_dir "${DOWNLOAD_DIR}" \
  --data_root "${DATA_ROOT}" \
  --workspace_name "${WORKSPACE_NAME}" \
  --raw_root "${DATA_ROOT}/raw_data" \
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

#### 4.2.2 意外中断后续跑

如果 mixed codec 音频已经生成了一部分，续跑时去掉 `--overwrite`，脚本会复用已存在文件，只补齐缺失部分。

```bash
cd "${PROJECT_ROOT}"

python "${RECIPE_DIR}/local/prepare_cnceleb_mixed_data.py" \
  --download_dir "${DOWNLOAD_DIR}" \
  --data_root "${DATA_ROOT}" \
  --workspace_name "${WORKSPACE_NAME}" \
  --raw_root "${DATA_ROOT}/raw_data" \
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
- 保持 `seed` 不变（默认 42）

可用下面命令快速查看当前 mixed 音频数量：

```bash
find "${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb_mixed_audio" -name "*.wav" | wc -l
```

### 4.3 生成后的关键目录

- Clean 训练集：`${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb/clean_train`
- Mixed 训练集：`${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb_mixed/train`
- Mixed 音频根目录：`${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb_mixed_audio`
- 测试集：`${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb/test`
- Trials：`${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb/trials`
- MUSAN：`${DATA_ROOT}/${WORKSPACE_NAME}/musan/wav.scp`
- RIRS：`${DATA_ROOT}/${WORKSPACE_NAME}/rirs/wav.scp`

### 4.4 从本地上传已生成数据后的路径修复

如果你在本地已经生成了 `CN_celeb_database`，再打包上传到服务器，这个方案完全可行。

需要额外做的一步是：把索引文件中的旧绝对路径批量改成当前服务器上的运行路径。

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

#### 4.4.2 一键改写脚本

下面脚本不会依赖固定的 Windows 盘符或固定的 Linux 根目录，而是根据当前仓库和工作区变量生成映射。

```bash
cd "${PROJECT_ROOT}"

python - <<'PY'
from pathlib import Path
import csv
import os

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
data_root = Path(os.environ["DATA_ROOT"]).resolve()
workspace_name = os.environ.get("WORKSPACE_NAME", "CN_celeb_database")
base = data_root / workspace_name

old_prefix_candidates = {
    (project_root / "egs/3dspeaker/sv-cam++/data/raw_data").as_posix(): (data_root / "raw_data").as_posix(),
    (project_root / "egs/3dspeaker/sv-cam++/data/CN_celeb_database").as_posix(): base.as_posix(),
}

wav_scp_files = [
    base / "cnceleb/clean_train/wav.scp",
    base / "cnceleb/test/wav.scp",
    base / "cnceleb_mixed/train/wav.scp",
    base / "musan/wav.scp",
    base / "rirs/wav.scp",
]
train_csv = base / "cnceleb_mixed/train/train.csv"

def rewrite_path(path: str) -> str:
    p = path.replace("\\\\", "/")
    for old, new in old_prefix_candidates.items():
        if p.startswith(old):
            return new + p[len(old):]
    return p

for f in wav_scp_files:
    if not f.exists():
        continue
    out = []
    with f.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            utt, path = line.split(maxsplit=1)
            out.append(f"{utt} {rewrite_path(path)}")
    with f.open("w", encoding="utf-8", newline="\n") as fout:
        fout.write("\n".join(out) + ("\n" if out else ""))
    print(f"rewritten wav.scp: {f}")

if train_csv.exists():
    rows = []
    with train_csv.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        rows.append(header)
        path_idx = header.index("path")
        for row in reader:
            row[path_idx] = rewrite_path(row[path_idx])
            rows.append(row)
    with train_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerows(rows)
    print(f"rewritten train.csv: {train_csv}")
PY
```

#### 4.4.3 改写后快速自检

```bash
head -n 3 "${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb_mixed/train/wav.scp"
head -n 3 "${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb_mixed/train/train.csv"
```

如果输出路径都已经切到当前 `${DATA_ROOT}` 下，说明索引修复完成。

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

建议把预训练权重放在 `${PRETRAINED_ROOT}` 下，例如：

- `${PRETRAINED_ROOT}/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin`

该权重对应：

- `embedding_size = 512`

因此可以直接配合：

- `egs/3dspeaker/sv-cam++/conf/cam++.yaml`

## 7. 微调训练

### 7.1 Mixed 数据微调

```bash
cd "${PROJECT_ROOT}"

python -m speakerlab.bin.train \
  --config "${RECIPE_DIR}/conf/cam++.yaml" \
  --gpu 0 \
  --init_model "${PRETRAINED_ROOT}/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin" \
  --data "${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb_mixed/train/train.csv" \
  --noise "${DATA_ROOT}/${WORKSPACE_NAME}/musan/wav.scp" \
  --reverb "${DATA_ROOT}/${WORKSPACE_NAME}/rirs/wav.scp" \
  --aug_prob 0.8 \
  --exp_dir "${EXP_ROOT}/campp_cnceleb_mixed_ft"
```

### 7.2 Clean 数据微调

如果你要做对照实验，只用 clean 训练集：

```bash
cd "${PROJECT_ROOT}"

python -m speakerlab.bin.train \
  --config "${RECIPE_DIR}/conf/cam++.yaml" \
  --gpu 0 \
  --init_model "${PRETRAINED_ROOT}/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin" \
  --data "${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb/clean_train/train.csv" \
  --noise "${DATA_ROOT}/${WORKSPACE_NAME}/musan/wav.scp" \
  --reverb "${DATA_ROOT}/${WORKSPACE_NAME}/rirs/wav.scp" \
  --aug_prob 0.8 \
  --exp_dir "${EXP_ROOT}/campp_cnceleb_clean_ft"
```

## 8. 断点续训

当前训练脚本支持自动续训。

只要重新执行相同的训练命令，并保持：

- `exp_dir` 不变

它就会自动从该目录下最新的 checkpoint 继续训练。

例如：

```bash
cd "${PROJECT_ROOT}"

python -m speakerlab.bin.train \
  --config "${RECIPE_DIR}/conf/cam++.yaml" \
  --gpu 0 \
  --init_model "${PRETRAINED_ROOT}/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin" \
  --data "${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb_mixed/train/train.csv" \
  --noise "${DATA_ROOT}/${WORKSPACE_NAME}/musan/wav.scp" \
  --reverb "${DATA_ROOT}/${WORKSPACE_NAME}/rirs/wav.scp" \
  --aug_prob 0.8 \
  --exp_dir "${EXP_ROOT}/campp_cnceleb_mixed_ft"
```

## 9. 提取测试集 embedding

训练完成后先提取测试集 embedding：

```bash
cd "${PROJECT_ROOT}"

python -m speakerlab.bin.extract \
  --exp_dir "${EXP_ROOT}/campp_cnceleb_mixed_ft" \
  --data "${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb/test/wav.scp" \
  --use_gpu \
  --gpu 0
```

如果是 clean 对照模型，则改成：

```bash
cd "${PROJECT_ROOT}"

python -m speakerlab.bin.extract \
  --exp_dir "${EXP_ROOT}/campp_cnceleb_clean_ft" \
  --data "${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb/test/wav.scp" \
  --use_gpu \
  --gpu 0
```

## 10. 计算测试集分数

先进入 trials 目录看有哪些官方协议文件：

```bash
ls "${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb/trials"
```

然后选择实际使用的 trial 文件，例如：

```bash
cd "${PROJECT_ROOT}"

python -m speakerlab.bin.compute_score_metrics \
  --enrol_data "${EXP_ROOT}/campp_cnceleb_mixed_ft/embeddings" \
  --test_data "${EXP_ROOT}/campp_cnceleb_mixed_ft/embeddings" \
  --scores_dir "${EXP_ROOT}/campp_cnceleb_mixed_ft/scores" \
  --trials "${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb/trials/<trial_file>"
```

## 11. 常用检查命令

### 11.1 检查训练集文件是否生成

```bash
ls "${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb_mixed/train"
```

### 11.2 检查 train.csv 行数

```bash
wc -l "${DATA_ROOT}/${WORKSPACE_NAME}/cnceleb_mixed/train/train.csv"
```

### 11.3 检查 checkpoint 是否保存

```bash
ls "${EXP_ROOT}/campp_cnceleb_mixed_ft/models"
```

### 11.4 查看训练日志

```bash
tail -f "${EXP_ROOT}/campp_cnceleb_mixed_ft/train.log"
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

后续都按下面的原则执行：

- 仓库内只保留代码、配置和文档
- 运行时数据路径全部通过变量控制
- 需要迁移机器时，只改第 2 节的变量定义，不改正文命令
