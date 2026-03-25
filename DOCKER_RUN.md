# Docker 运行说明

本文档面向 `NVIDIA DGX Spark` 一类 ARM64 (`aarch64`) GPU 服务器。

当前仓库的 Docker 配置特点：

- 使用 NVIDIA NGC PyTorch 基础镜像，而不是手动安装 x86_64 CUDA wheel
- 额外安装 `ffmpeg`
- 按 [requirements.txt](/e:/Graduation_project/camplusplus/requirements.txt) 安装项目依赖
- 默认把代码和运行时数据分开挂载

## 1. 前提检查

先在宿主机确认 Docker 能看到 GPU：

```bash
docker run --rm --runtime=nvidia --gpus all \
  nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 \
  nvidia-smi
```

如果这一步失败，不要继续构建项目镜像，先修宿主机的 NVIDIA Container Runtime。

## 2. 构建镜像

在仓库根目录执行：

```bash
cd "$(git rev-parse --show-toplevel)"
docker compose build
```

如果你所在环境访问 `nvcr.io` 需要登录，先执行：

```bash
docker login nvcr.io
```

## 3. 启动交互式容器

```bash
docker compose run --rm camplusplus
```

容器内默认目录：

- 代码目录：`/workspace/camplusplus`
- 运行时目录：`/workspace/runtime`

对应宿主机挂载：

- 当前仓库目录 -> `/workspace/camplusplus`
- `../camplusplus_runtime` -> `/workspace/runtime`

## 4. 容器内快速验收

进入容器后，先检查这几项：

```bash
python3 -c "import torch, torchaudio; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
ffmpeg -version
python3 -c "import numpy, scipy, sklearn, soundfile, yaml, kaldiio, pandas; print('python deps ok')"
```

如果 `torch.cuda.is_available()` 返回 `True`，说明 GPU 直通正常。

## 5. 运行目录约定

`docker-compose.yml` 里已经预设这些环境变量：

```bash
echo "$PROJECT_ROOT"
echo "$WORKSPACE_ROOT"
echo "$DOWNLOAD_DIR"
echo "$DATA_ROOT"
echo "$EXP_ROOT"
echo "$PRETRAINED_ROOT"
echo "$WORKSPACE_NAME"
```

默认值为：

- `PROJECT_ROOT=/workspace/camplusplus`
- `WORKSPACE_ROOT=/workspace/runtime`
- `DOWNLOAD_DIR=/workspace/runtime/download_data`
- `DATA_ROOT=/workspace/runtime/camplusplus_data`
- `EXP_ROOT=/workspace/runtime/camplusplus_exp`
- `PRETRAINED_ROOT=/workspace/runtime/pretrained`
- `WORKSPACE_NAME=CN_celeb_database`

先创建运行目录：

```bash
mkdir -p "$DOWNLOAD_DIR" "$DATA_ROOT" "$EXP_ROOT" "$PRETRAINED_ROOT"
```

## 6. 容器内准备项目路径

如果你使用 recipe 脚本：

```bash
cd /workspace/camplusplus/egs/3dspeaker/sv-cam++
source ./path.sh
```

如果你使用 `python -m ...` 的方式，一般直接在仓库根目录即可：

```bash
cd /workspace/camplusplus
```

## 7. 数据准备示例

```bash
cd /workspace/camplusplus

python3 egs/3dspeaker/sv-cam++/local/prepare_cnceleb_mixed_data.py \
  --download_dir "$DOWNLOAD_DIR" \
  --data_root "$DATA_ROOT" \
  --workspace_name "$WORKSPACE_NAME" \
  --raw_root "$DATA_ROOT/raw_data" \
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

## 8. 训练示例

```bash
cd /workspace/camplusplus

python3 -m speakerlab.bin.train \
  --config egs/3dspeaker/sv-cam++/conf/cam++.yaml \
  --gpu 0 \
  --init_model "$PRETRAINED_ROOT/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin" \
  --data "$DATA_ROOT/$WORKSPACE_NAME/cnceleb_mixed/train/train.csv" \
  --noise "$DATA_ROOT/$WORKSPACE_NAME/musan/wav.scp" \
  --reverb "$DATA_ROOT/$WORKSPACE_NAME/rirs/wav.scp" \
  --aug_prob 0.8 \
  --exp_dir "$EXP_ROOT/campp_cnceleb_mixed_ft"
```

## 9. 常见问题

- `docker build` 卡在 `scikit-learn` 或 `scipy`
  说明 ARM64 下没有命中可用 wheel，当前 Dockerfile 已经补了 `build-essential`、`gfortran`、`openblas`、`lapack` 等编译依赖，继续等待即可；如果仍失败，再单独看报错。

- `nvcr.io` 拉镜像失败
  一般是没有登录 NGC，或者外网访问受限。先执行 `docker login nvcr.io`。

- 容器里 `torch.cuda.is_available()` 为 `False`
  先回到宿主机检查 `nvidia-smi` 和 NVIDIA Container Runtime，再确认 `docker compose` 版本支持 `gpus: all`。

- 输出文件找不到
  训练、数据、预训练模型默认都不写到代码目录，而是写到挂载的 `/workspace/runtime`。

## 10. 当前建议

- 先 `docker compose build`
- 然后 `docker compose run --rm camplusplus`
- 容器内先做 GPU 和依赖验收
- 再开始跑数据准备和训练
