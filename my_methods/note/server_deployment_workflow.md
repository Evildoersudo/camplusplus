# CA-AFC 服务器部署与运行流程

> 统一说明：本文件维护当前服务器实跑命令。每次新增或修改脚本后，需同步把可执行命令更新到本文件对应章节。

本文档按当前 DGX Spark + Docker 方案整理，默认约定如下：

- 项目根目录：`~/lkj/camplusplus`
- 容器内项目目录：`/workspace/camplusplus`
- 数据目录：`~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data`
- 预训练 CAM++：`pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin`

本文档覆盖：

1. Docker 环境构建
2. CN-Celeb mixed codec 数据准备
3. pair manifest 与 feature manifest 构建
4. CA-AFC 预提特征训练
5. 断点续训与评测

## 1. Docker 环境

### 1.1 基础配置

当前部署使用：

- [Dockerfile](/e:/Graduation_project/camplusplus/Dockerfile)
- [docker-compose.yml](/e:/Graduation_project/camplusplus/docker-compose.yml)
- [req_no_torch.txt](/e:/Graduation_project/camplusplus/req_no_torch.txt)

设计原则：

- 基础镜像使用 NGC PyTorch 镜像，直接复用镜像内 `torch`
- 非 `torch/torchaudio` 依赖单独安装
- `ffmpeg` 从源码编译，并显式启用：
  - `aac`
  - `libopus`
  - `libvo_amrwbenc`

### 1.2 登录 NGC

```bash
docker login nvcr.io
```

用户名填写：

```text
$oauthtoken
```

密码填写你的 NGC API Key。

### 1.3 构建容器

首次构建：

```bash
cd ~/lkj/camplusplus
docker compose build
```

如果你修改了基础镜像、`req_no_torch.txt` 或 `Dockerfile` 中的核心层，再考虑：

```bash
docker compose build --no-cache
```

### 1.4 启动并进入容器

```bash
cd ~/lkj/camplusplus
docker compose up -d
docker compose exec camplusplus bash
```

### 1.5 容器内验证

先确认 `torch/torchaudio`：

```bash
python -c "import torch, torchaudio; print(torch.__version__); print(torchaudio.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

再确认 `ffmpeg` 编码器：

```bash
ffmpeg -hide_banner -encoders | grep -E 'libopus|aac|libvo_amrwbenc|pcm_alaw|pcm_mulaw'
```

## 2. 数据目录约定

容器内统一使用：

- 下载目录：`/workspace/camplusplus/egs/3dspeaker/sv-cam++/data/download_data`
- 原始解压目录：`/workspace/camplusplus/egs/3dspeaker/sv-cam++/data/raw_data`
- 工作目录：`/workspace/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database`

先创建目录：

```bash
mkdir -p /workspace/camplusplus/egs/3dspeaker/sv-cam++/data/download_data
mkdir -p /workspace/camplusplus/egs/3dspeaker/sv-cam++/data/raw_data
mkdir -p /workspace/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database
```

## 3. 生成 CN-Celeb mixed codec 数据

使用脚本：

- [prepare_cnceleb_mixed_data.py](/e:/Graduation_project/camplusplus/egs/3dspeaker/sv-cam++/local/prepare_cnceleb_mixed_data.py)

在容器内执行：

```bash
cd /workspace/camplusplus

python egs/3dspeaker/sv-cam++/local/prepare_cnceleb_mixed_data.py \
  --download_dir /workspace/camplusplus/egs/3dspeaker/sv-cam++/data/download_data \
  --data_root egs/3dspeaker/sv-cam++/data \
  --workspace_name CN_celeb_database \
  --raw_root egs/3dspeaker/sv-cam++/data/raw_data \
  --clean_ratio 0.25 \
  --opus_ratio 0.25 \
  --aac_ratio 0.25 \
  --amrwb_ratio 0.25 \
  --g711_ratio 0.0 \
  --opus_bitrates 16k \
  --aac_bitrates 16k \
  --amrwb_bitrates 15.85k \
  --sample_rate 16000 \
  --num_workers 8 \
  --prepare_csv_nj 8
```

首次强制重建 degraded 音频可加：

```bash
--overwrite
```

关键输出目录：

- clean train：
  - `/workspace/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/clean_train`
- test：
  - `/workspace/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/test`
- trials：
  - `/workspace/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/trials`
- mixed train：
  - `/workspace/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_mixed/train`
- mixed audio：
  - `/workspace/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_mixed_audio`

说明：

- `prepare_cnceleb_mixed_data.py` 默认输出是 `cnceleb_mixed/train` 和 `cnceleb_mixed_audio`
- `cnceleb_fixedrate_mixed/train` 与 `cnceleb_fixedrate_mixed_audio` 属于 fixed-rate 封装脚本默认命名，不是这个脚本的默认输出

## 4. 构建 CA-AFC pair manifest

使用脚本：

- [build_pair_manifest.py](/e:/Graduation_project/camplusplus/my_methods/tools/build_pair_manifest.py)

容器内执行：

```bash
cd /workspace/camplusplus

python my_methods/tools/build_pair_manifest.py \
  --clean_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/clean_train/wav.scp \
  --codec_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_mixed/train/wav.scp \
  --utt2spk egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/clean_train/utt2spk \
  --codec_assignment_csv egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_mixed/train/codec_assignment.csv \
  --output_csv my_methods/exp/cnceleb_fixedrate_pair_manifest.csv
```

输出：

- `/workspace/camplusplus/my_methods/exp/cnceleb_fixedrate_pair_manifest.csv`

## 5. 预提特征

使用脚本：

- [precompute_pair_features.py](/e:/Graduation_project/camplusplus/my_methods/tools/precompute_pair_features.py)

容器内执行：

```bash
cd /workspace/camplusplus

python my_methods/tools/precompute_pair_features.py \
  --pair_manifest my_methods/exp/cnceleb_fixedrate_pair_manifest.csv \
  --output_root my_methods/exp/ca_afc_features_cnceleb_fixedrate \
  --output_manifest my_methods/exp/cnceleb_fixedrate_feature_manifest.csv \
  --sample_rate 16000
```

可选并行：

```bash
--num_workers 4
```

输出：

- 特征目录：`/workspace/camplusplus/my_methods/exp/ca_afc_features_cnceleb_fixedrate`
- 特征清单：`/workspace/camplusplus/my_methods/exp/cnceleb_fixedrate_feature_manifest.csv`

## 6. smoke test

先用小样本确认：

- 数据读取正常
- 训练链路正常
- checkpoint 正常保存
- loss、lr、ETA 日志正常

```bash
cd /workspace/camplusplus

python -u my_methods/scripts/train_ca_afc.py \
  --train_feature_manifest my_methods/exp/cnceleb_fixedrate_feature_manifest.csv \
  --output_dir my_methods/exp/ca_afc_cnceleb_fixedrate_smoke \
  --campplus_model_bin pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin \
  --max_frames 300 \
  --batch_size 32 \
  --num_workers 0 \
  --pretrain_epochs 1 \
  --finetune_epochs 1 \
  --max_train_samples 512 \
  --max_valid_samples 64 \
  --optimizer adamw \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --scheduler cosine \
  --warmup_steps 50 \
  --min_lr 1e-5 \
  --grad_clip_norm 5.0 \
  --log_interval 1 \
  --lambda_rec 1.0 \
  --lambda_emb 0.3 \
  --lambda_smooth 0.01 \
  --device cuda
```

## 7. 正式训练 CA-AFC

使用脚本：

- [train_ca_afc.py](/e:/Graduation_project/camplusplus/my_methods/scripts/train_ca_afc.py)

当前推荐先用 `AdamW + warmup + cosine`：

```bash
cd /workspace/camplusplus

python my_methods/scripts/train_ca_afc.py \
  --train_feature_manifest my_methods/exp/cnceleb_fixedrate_feature_manifest.csv \
  --output_dir my_methods/exp/ca_afc_cnceleb_fixedrate \
  --campplus_model_bin pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin \
  --max_frames 300 \
  --batch_size 64 \
  --num_workers 4 \
  --pretrain_epochs 5 \
  --finetune_epochs 10 \
  --optimizer adamw \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --scheduler cosine \
  --warmup_steps 1000 \
  --min_lr 1e-5 \
  --grad_clip_norm 5.0 \
  --log_interval 20 \
  --lambda_rec 1.0 \
  --lambda_emb 0.3 \
  --lambda_smooth 0.01 \
  --device cuda
```

当前版本建议先用“稳定收敛 + 可观测诊断”配置（含梯度探针与 finetune codec 加权采样）：

```bash
cd /workspace/camplusplus

python my_methods/scripts/train_ca_afc.py \
  --train_feature_manifest my_methods/exp/cnceleb_fixedrate_feature_manifest.csv \
  --output_dir my_methods/exp/ca_afc_cnceleb_fixedrate \
  --campplus_model_bin pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin \
  --max_frames 300 \
  --batch_size 64 \
  --num_workers 4 \
  --pretrain_epochs 15 \
  --finetune_epochs 20 \
  --optimizer adamw \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --scheduler none \
  --grad_clip_norm 5.0 \
  --log_interval 20 \
  --lambda_rec 1.0 \
  --lambda_emb 0.3 \
  --lambda_smooth 0.01 \
  --emb_term_target_ratio 0.10 \
  --emb_lambda_scale_max 20.0 \
  --grad_probe_interval 100 \
  --finetune_codec_weights clean=0.4,aac=1.0,opus=1.5,amrwb=1.8 \
  --device cuda 2>&1 | tee my_methods/exp/ca_afc_cnceleb_fixedrate/train.log
```

如果要做 `SGD + momentum` 对照实验：

```bash
cd /workspace/camplusplus

python my_methods/scripts/train_ca_afc.py \
  --train_feature_manifest my_methods/exp/cnceleb_fixedrate_feature_manifest.csv \
  --output_dir my_methods/exp/ca_afc_cnceleb_fixedrate_sgd \
  --campplus_model_bin pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin \
  --max_frames 300 \
  --batch_size 64 \
  --num_workers 4 \
  --pretrain_epochs 15 \
  --finetune_epochs 20 \
  --optimizer sgd \
  --momentum 0.9 \
  --nesterov \
  --lr 0.01 \
  --weight_decay 1e-4 \
  --scheduler cosine \
  --warmup_steps 1000 \
  --min_lr 1e-4 \
  --grad_clip_norm 5.0 \
  --log_interval 20 \
  --lambda_rec 1.0 \
  --lambda_emb 0.3 \
  --lambda_smooth 0.01 \
  --device cuda
```

说明：

- 每个 epoch 都会保存一次 checkpoint 到 `output_dir/checkpoints/`
- `best_frontend.pt` 保存当前最佳验证集权重
- checkpoint 现在包含：
  - `frontend_state`
  - `optimizer_state`
  - `scheduler_state`
  - `epoch`
  - `stage`
  - `global_step`

### 7.1 断点续训

```bash
cd /workspace/camplusplus

python my_methods/scripts/train_ca_afc.py \
  --train_feature_manifest my_methods/exp/cnceleb_fixedrate_feature_manifest.csv \
  --output_dir my_methods/exp/ca_afc_cnceleb_fixedrate \
  --campplus_model_bin pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin \
  --resume \
  --device cuda
```

## 7.2 训练日志自动汇总与收敛曲线

使用脚本：

- `my_methods/tools/summarize_train_log.py`

功能：

- 从 `train.log` 提取每个 epoch 的 `loss / rec_loss / emb_loss / emb/rec / grad_emb/rec / lr`
- 终端打印统计表
- 导出 JSON
- 生成收敛曲线 PNG

```bash
cd /workspace/camplusplus

python my_methods/tools/summarize_train_log.py \
  --log my_methods/exp/ca_afc_cnceleb_fixedrate/train.log \
  --split both \
  --trend_width 30 \
  --json_out my_methods/exp/ca_afc_cnceleb_fixedrate/epoch_curve_summary.json \
  --plot_dir my_methods/exp/ca_afc_cnceleb_fixedrate/plots
```

仅查看训练集统计：

```bash
python my_methods/tools/summarize_train_log.py \
  --log my_methods/exp/ca_afc_cnceleb_fixedrate/train.log \
  --split train
```

## 8. 直接从音频训练（可选）

如果你不想预提特征，可使用：

- [train_ca_afc_from_audio.py](/e:/Graduation_project/camplusplus/my_methods/scripts/train_ca_afc_from_audio.py)

```bash
cd /workspace/camplusplus

python my_methods/scripts/train_ca_afc_from_audio.py \
  --train_manifest my_methods/exp/cnceleb_fixedrate_pair_manifest.csv \
  --output_dir my_methods/exp/ca_afc_cnceleb_fixedrate_audio \
  --campplus_model_bin pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin \
  --max_frames 300 \
  --batch_size 32 \
  --num_workers 4 \
  --pretrain_epochs 15 \
  --finetune_epochs 20 \
  --optimizer adamw \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --scheduler cosine \
  --warmup_steps 1000 \
  --min_lr 1e-5 \
  --grad_clip_norm 5.0 \
  --log_interval 20 \
  --lambda_rec 1.0 \
  --lambda_emb 0.3 \
  --lambda_smooth 0.01 \
  --device cuda
```

音频版续训：

```bash
cd /workspace/camplusplus

python my_methods/scripts/train_ca_afc_from_audio.py \
  --train_manifest my_methods/exp/cnceleb_fixedrate_pair_manifest.csv \
  --output_dir my_methods/exp/ca_afc_cnceleb_fixedrate_audio \
  --campplus_model_bin pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin \
  --resume \
  --device cuda
```

## 9. 评测 CA-AFC + CAM++

使用脚本：

- [run_ca_afc_codec_eval.py](/e:/Graduation_project/camplusplus/my_methods/scripts/run_ca_afc_codec_eval.py)

```bash
cd /workspace/camplusplus

python my_methods/scripts/run_ca_afc_codec_eval.py \
  --test_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/test/wav.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/trials/trials.lst \
  --frontend_ckpt my_methods/exp/ca_afc_cnceleb_fixedrate/best_frontend.pt \
  --campplus_model_bin pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin \
  --codec_conditions clean,opus@16k,aac@16k,amrwb@15.85k \
  --embedding_cache_root my_methods/exp/ca_afc_cnceleb_fixedrate_eval/embedding_cache \
  --limit 2000 \
  --target_limit 100 \
  --nontarget_limit 1900 \
  --report_csv my_methods/exp/ca_afc_cnceleb_fixedrate_eval/report.csv \
  --report_json my_methods/exp/ca_afc_cnceleb_fixedrate_eval/report.json \
  --table_md my_methods/exp/ca_afc_cnceleb_fixedrate_eval/report.md \
  --plot_dir my_methods/exp/ca_afc_cnceleb_fixedrate_eval/plots \
  --device cuda
```

## 10. 长时任务建议

容器内长训练不要直接裸跑，建议用 `tmux`：

```bash
apt-get update && apt-get install -y tmux
tmux new -s caafc
```

在 `tmux` 里启动训练并落日志：

```bash
cd /workspace/camplusplus

python my_methods/scripts/train_ca_afc.py \
  --train_feature_manifest my_methods/exp/cnceleb_fixedrate_feature_manifest.csv \
  --output_dir my_methods/exp/ca_afc_cnceleb_fixedrate \
  --campplus_model_bin pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin \
  --max_frames 300 \
  --batch_size 64 \
  --num_workers 4 \
  --pretrain_epochs 15 \
  --finetune_epochs 20 \
  --optimizer adamw \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --scheduler none \
  --grad_clip_norm 5.0 \
  --log_interval 20 \
  --lambda_rec 1.0 \
  --lambda_emb 0.3 \
  --lambda_smooth 0.01 \
  --emb_term_target_ratio 0.10 \
  --emb_lambda_scale_max 20.0 \
  --grad_probe_interval 100 \
  --finetune_codec_weights clean=0.4,aac=1.0,opus=1.5,amrwb=1.8 \
  --device cuda 2>&1 | tee my_methods/exp/ca_afc_cnceleb_fixedrate/train.log
```

退出但不停止训练：

```text
Ctrl+b 然后按 d
```

重新连回：

```bash
tmux attach -t caafc
```

## 11. 最关键的注意事项

1. 不要直接复用宿主机生成的旧 manifest 作为容器正式输入，最好在容器内重新生成
2. 容器内所有路径统一使用 `/workspace/camplusplus/...`
3. 旧的宿主机绝对路径 `/home/dgx/...` 与容器路径不一致，容易导致读取失败
4. 预提特征训练更适合正式实验；直接从音频训练更适合调试或简化流程
