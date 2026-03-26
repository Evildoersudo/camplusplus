# CA-AFC 服务器部署与运行流程

本文档按你当前服务器目录约定编写：

- 代码根目录：
  - `~/lkj/camplusplus`
- 数据下载目录：
  - `~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/download_data`

也就是说，这一版不是“代码和数据分离到 `/root/autodl-tmp`”的方案，而是**数据直接放在项目目录内部**。

本文档说明如何从“上传并解压数据集”开始，一步一步完成：

1. 准备原始数据目录
2. 生成 CN-Celeb clean / test / trials / mixed codec 数据
3. 构建 CA-AFC 的 pair manifest
4. 预提取离线特征
5. 训练 CA-AFC
6. 评测 `CA-AFC + CAM++`

## 1. 目录约定

本文档统一使用以下目录：

- 项目根目录：
  - `~/lkj/camplusplus`
- 下载数据目录：
  - `~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/download_data`
- 数据工作目录：
  - `~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database`
- 原始解压目录：
  - `~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/raw_data`
- 预训练模型目录：
  - `~/lkj/camplusplus/pretrained`
- `my_methods` 实验输出目录：
  - `~/lkj/camplusplus/my_methods/exp`

进入项目目录：

```bash
cd ~/lkj/camplusplus
```

## 2. 环境准备

### 2.1 Python 环境

```bash
cd ~/lkj/camplusplus

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install soundfile
```

### 2.2 GPU 版 PyTorch

如果服务器是 CUDA 12.1，可执行：

```bash
pip uninstall -y torch torchaudio
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 ffmpeg

固定码率 codec 数据生成和评测依赖 `ffmpeg`：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
ffmpeg -version
```

## 3. 数据应该放在哪里

### 3.1 压缩包放置位置

把原始压缩包上传到：

- `~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/download_data`

例如：

- `CN-Celeb_flac.tar.gz`
- `musan.tar.gz`
- `RIRS_NOISES.zip`

### 3.2 解压后的目标位置

把原始数据解压到：

- `~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/raw_data`

先创建目录：

```bash
mkdir -p ~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/download_data
mkdir -p ~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/raw_data
```

### 3.3 解压示例

```bash
cd ~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/download_data

tar -xzf CN-Celeb_flac.tar.gz -C ~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/raw_data
tar -xzf musan.tar.gz -C ~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/raw_data
unzip RIRS_NOISES.zip -d ~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/raw_data
```

解压后建议形成这样的结构：

```text
~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/raw_data/
├─ CN-Celeb_flac/
│  ├─ data/
│  └─ eval/
├─ musan/
└─ RIRS_NOISES/
```

## 4. 预训练 CAM++ 放在哪里

建议放在：

- `~/lkj/camplusplus/pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin`

如果目录不存在，先创建：

```bash
mkdir -p ~/lkj/camplusplus/pretrained/speech_campplus_sv_zh-cn_16k-common
```

## 5. 为什么不要直接复用本地 manifest

你本地生成过的这些文件通常包含 Windows 绝对路径：

- `my_methods/note/cnceleb_fixedrate_pair_manifest.csv`
- `my_methods/note/cnceleb_fixedrate_feature_manifest.csv`
- 各类 `wav.scp`
- 部分 `train.csv`

因此，部署到服务器时更稳的做法是：

- 不直接拷贝这些 manifest 作为正式输入
- 在服务器上重新生成

这样生成出来的路径天然就是服务器路径，不会再受 `E:\...` 影响。

## 6. 第一步：生成 CN-Celeb 工作目录和 mixed codec 训练数据

使用脚本：

- [prepare_cnceleb_mixed_data.py](E:/Speaker_recognition/Graduation_Project/camplusplus/egs/3dspeaker/sv-cam++/local/prepare_cnceleb_mixed_data.py)

在服务器上执行：

```bash
cd ~/lkj/camplusplus

python egs/3dspeaker/sv-cam++/local/prepare_cnceleb_mixed_data.py \
  --download_dir ~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/download_data \
  --data_root ~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data \
  --workspace_name CN_celeb_database \
  --raw_root ~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/raw_data \
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

如果你要第一次全量重做 degraded 音频，可以加：

```bash
--overwrite
```

生成后关键目录通常是：

- clean train：
  - `~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/clean_train`
- test：
  - `~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/test`
- trials：
  - `~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/trials`
- fixedrate mixed train：
  - `~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_fixedrate_mixed/train`
- fixedrate mixed 音频：
  - `~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_fixedrate_mixed_audio`

## 7. 第二步：构建 CA-AFC 训练配对清单

使用脚本：

- [build_pair_manifest.py](E:/Speaker_recognition/Graduation_Project/camplusplus/my_methods/tools/build_pair_manifest.py)

执行命令：

```bash
cd ~/lkj/camplusplus

python my_methods/tools/build_pair_manifest.py \
  --clean_wav_scp ~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/clean_train/wav.scp \
  --codec_wav_scp ~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_fixedrate_mixed/train/wav.scp \
  --utt2spk ~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/clean_train/utt2spk \
  --codec_assignment_csv ~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_fixedrate_mixed/train/codec_assignment.csv \
  --output_csv ~/lkj/camplusplus/my_methods/exp/cnceleb_fixedrate_pair_manifest.csv
```

输出：

- `~/lkj/camplusplus/my_methods/exp/cnceleb_fixedrate_pair_manifest.csv`

## 8. 第三步：预提取离线特征

使用脚本：

- [precompute_pair_features.py](E:/Speaker_recognition/Graduation_Project/camplusplus/my_methods/tools/precompute_pair_features.py)

执行命令：

```bash
cd ~/lkj/camplusplus

python my_methods/tools/precompute_pair_features.py \
  --pair_manifest ~/lkj/camplusplus/my_methods/exp/cnceleb_fixedrate_pair_manifest.csv \
  --output_root ~/lkj/camplusplus/my_methods/exp/ca_afc_features_cnceleb_fixedrate \
  --output_manifest ~/lkj/camplusplus/my_methods/exp/cnceleb_fixedrate_feature_manifest.csv \
  --sample_rate 16000
```

输出：

- 特征目录：
  - `~/lkj/camplusplus/my_methods/exp/ca_afc_features_cnceleb_fixedrate`
- 特征清单：
  - `~/lkj/camplusplus/my_methods/exp/cnceleb_fixedrate_feature_manifest.csv`

## 9. 第四步：先做 smoke test

先用小样本确认：

- 数据读取正常
- 训练链路正常
- checkpoint 正常保存
- loss 和 ETA 日志正常

执行命令：

```bash
cd ~/lkj/camplusplus

python my_methods/scripts/train_ca_afc.py \
  --train_feature_manifest ~/lkj/camplusplus/my_methods/exp/cnceleb_fixedrate_feature_manifest.csv \
  --output_dir ~/lkj/camplusplus/my_methods/exp/ca_afc_cnceleb_fixedrate_smoke \
  --campplus_model_bin ~/lkj/camplusplus/pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
  --max_frames 300 \
  --batch_size 16 \
  --num_workers 0 \
  --pretrain_epochs 1 \
  --finetune_epochs 1 \
  --max_train_samples 4096 \
  --max_valid_samples 512 \
  --log_interval 20 \
  --lambda_rec 1.0 \
  --lambda_emb 0.3 \
  --lambda_smooth 0.01 \
  --device cuda
```

输出目录：

- `~/lkj/camplusplus/my_methods/exp/ca_afc_cnceleb_fixedrate_smoke`

## 10. 第五步：正式训练 CA-AFC

smoke test 没问题后，再跑正式训练：

```bash
cd ~/lkj/camplusplus

python my_methods/scripts/train_ca_afc.py \
  --train_feature_manifest ~/lkj/camplusplus/my_methods/exp/cnceleb_fixedrate_feature_manifest.csv \
  --output_dir ~/lkj/camplusplus/my_methods/exp/ca_afc_cnceleb_fixedrate \
  --campplus_model_bin ~/lkj/camplusplus/pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
  --max_frames 300 \
  --batch_size 16 \
  --num_workers 0 \
  --pretrain_epochs 5 \
  --finetune_epochs 10 \
  --log_interval 100 \
  --lambda_rec 1.0 \
  --lambda_emb 0.3 \
  --lambda_smooth 0.01 \
  --device cuda
```

输出目录：

- `~/lkj/camplusplus/my_methods/exp/ca_afc_cnceleb_fixedrate`

断点续训：

```bash
cd ~/lkj/camplusplus

python my_methods/scripts/train_ca_afc.py \
  --train_feature_manifest ~/lkj/camplusplus/my_methods/exp/cnceleb_fixedrate_feature_manifest.csv \
  --output_dir ~/lkj/camplusplus/my_methods/exp/ca_afc_cnceleb_fixedrate \
  --campplus_model_bin ~/lkj/camplusplus/pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
  --resume \
  --device cuda
```

## 11. 第六步：评测 CA-AFC + CAM++

使用脚本：

- [run_ca_afc_codec_eval.py](E:/Speaker_recognition/Graduation_Project/camplusplus/my_methods/scripts/run_ca_afc_codec_eval.py)

执行命令：

```bash
cd ~/lkj/camplusplus

python my_methods/scripts/run_ca_afc_codec_eval.py \
  --test_wav_scp ~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/test/wav.scp \
  --trials_file ~/lkj/camplusplus/egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/trials/trials.lst \
  --frontend_ckpt ~/lkj/camplusplus/my_methods/exp/ca_afc_cnceleb_fixedrate/best_frontend.pt \
  --campplus_model_bin ~/lkj/camplusplus/pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
  --codec_conditions clean,opus@16k,aac@16k,amrwb@15.85k \
  --embedding_cache_root ~/lkj/camplusplus/my_methods/exp/ca_afc_cnceleb_fixedrate_eval/embedding_cache \
  --limit 2000 \
  --target_limit 100 \
  --nontarget_limit 1900 \
  --report_csv ~/lkj/camplusplus/my_methods/exp/ca_afc_cnceleb_fixedrate_eval/report.csv \
  --report_json ~/lkj/camplusplus/my_methods/exp/ca_afc_cnceleb_fixedrate_eval/report.json \
  --table_md ~/lkj/camplusplus/my_methods/exp/ca_afc_cnceleb_fixedrate_eval/report.md \
  --plot_dir ~/lkj/camplusplus/my_methods/exp/ca_afc_cnceleb_fixedrate_eval/plots \
  --device cuda
```

输出目录：

- `~/lkj/camplusplus/my_methods/exp/ca_afc_cnceleb_fixedrate_eval`

关键产物：

- `report.csv`
- `report.json`
- `report.md`
- `plots/ca_afc_eer.png`
- `plots/ca_afc_min_dcf.png`

## 12. 最终建议目录结构

```text
~/lkj/camplusplus/
├─ pretrained/
│  └─ speech_campplus_sv_zh-cn_16k-common/
│     └─ campplus_cn_common.bin
├─ my_methods/
│  └─ exp/
│     ├─ cnceleb_fixedrate_pair_manifest.csv
│     ├─ cnceleb_fixedrate_feature_manifest.csv
│     ├─ ca_afc_features_cnceleb_fixedrate/
│     ├─ ca_afc_cnceleb_fixedrate_smoke/
│     ├─ ca_afc_cnceleb_fixedrate/
│     └─ ca_afc_cnceleb_fixedrate_eval/
└─ egs/
   └─ 3dspeaker/
      └─ sv-cam++/
         └─ data/
            ├─ download_data/
            ├─ raw_data/
            └─ CN_celeb_database/
```

## 13. 最关键的注意事项

部署到服务器时，最重要的是：

1. 不要直接复用本地生成的 pair manifest 和 feature manifest
2. 到服务器后重新生成：
   - `wav.scp`
   - `pair manifest`
   - `feature manifest`
   - `.pt` 离线特征
3. 训练和评测时尽量都使用服务器本地重新生成的文件

这样可以彻底避开 Windows 绝对路径导致的问题。
