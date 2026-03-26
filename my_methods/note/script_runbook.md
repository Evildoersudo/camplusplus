# My Methods Script Runbook

## 目录组织

服务器从解压数据集到训练、评测的完整流程见：

- [server_deployment_workflow.md](E:/Speaker_recognition/Graduation_Project/camplusplus/my_methods/note/server_deployment_workflow.md)

当前 `my_methods` 已按功能拆分，后续新增脚本也按这个结构放置：

- `my_methods/models/`
  - 放网络结构定义。
- `my_methods/data/`
  - 放特征提取、数据集、损失函数、模型加载等公共数据逻辑。
- `my_methods/tools/`
  - 放数据准备、清单构建、离线预处理工具。
- `my_methods/scripts/`
  - 放训练、评测、推理等直接执行的入口脚本。
- `my_methods/note/`
  - 放技术路线、脚本操作文档、实验说明。

## 文件说明

### `my_methods/models/ca_afc_frontend.py`

- 作用：定义 CA-AFC 前端网络。
- 关键类：
- `SpectralEncoder`：编码 codec FBank 主特征分支。
- `AuxEncoder`：编码 `pitch / delta-pitch / voicing` 辅助特征分支。
- `AttentiveFusion`：做双分支动态加权融合。
- `CAAFCFrontend`：输出增强后的 80 维 FBank、频带权重和残差补偿。

### `my_methods/data/ca_afc_data.py`

- 作用：提供特征提取、配对数据集、损失函数和冻结 CAM++ 加载工具。
- 关键函数：
- `compute_fbank`：提取 80 维 Kaldi 风格 FBank。
- `compute_aux_features`：提取 `pitch + delta-pitch + voiced-flag`。
- `PairFeatureDataset`：读取 clean/codec 配对清单并生成训练样本。
- `weighted_reconstruction_loss`：加权重建损失。
- `cosine_embedding_consistency`：冻结 CAM++ 的 embedding 一致性损失。
- `load_frozen_campplus`：加载并冻结预训练 CAM++。

### `my_methods/data/frontend_features.py`

- 作用：提供经典前端特征处理函数。
- 关键函数：
- `load_audio_mono`：读取单通道音频并按目标采样率重采样。
- `compute_fbank_feature`：提取 FBank。
- `compute_mfcc_feature`：提取 MFCC。
- `apply_cmvn`：对特征做 utterance-level CMVN。

### `my_methods/tools/build_pair_manifest.py`

- 作用：把 clean `wav.scp` 和 codec `wav.scp` 组织成 clean/codec 配对训练清单。
- 输出：供 CA-AFC 训练使用的 pair manifest CSV。

### `my_methods/tools/precompute_pair_features.py`

- 作用：把 pair manifest 中的 `clean_wav / codec_wav` 预提取成 `.pt` 特征文件。
- 每个 `.pt` 文件包含：
- `clean_feat`
- `codec_feat`
- `aux_feat`
- `length`
- 输出：供 CA-AFC 训练使用的 offline feature manifest CSV。

### `my_methods/tools/visualize_frontend_features.py`

- 作用：读取单条音频，提取 FBank、MFCC、CMVN，并保存 `.npy` 和图片。
- 输出：
- `waveform.png`
- `fbank.npy`
- `fbank.png`
- `mfcc.npy`
- `mfcc.png`
- `fbank_cmvn.npy`
- `fbank_cmvn.png`
- `mfcc_cmvn.npy`
- `mfcc_cmvn.png`
- `summary.json`

### `my_methods/scripts/train_ca_afc.py`

- 作用：按“两阶段训练”训练 CA-AFC。
- 阶段 1：只做特征重建预训练。
- 阶段 2：加入冻结 CAM++ 的 embedding 一致性损失。
- 输出：
- `best_frontend.pt`
- `checkpoints/*.pt`
- `train_summary.json`

### `my_methods/scripts/run_ca_afc_codec_eval.py`

- 作用：把 CA-AFC 前端接到冻结 CAM++ 前面，在固定码率 codec 条件下评测 EER/minDCF。
- 输出：
- `report.csv`
- `report.json`
- `report.md`
- `plot_dir/ca_afc_eer.png`
- `plot_dir/ca_afc_min_dcf.png`

## 推荐流程

1. 先准备 clean/codec 配对 manifest。
2. 训练 CA-AFC 前端。
3. 用固定码率 codec 协议评测 `CA-AFC + CAM++`。

## 1. 构建配对清单

```bash
python my_methods/tools/build_pair_manifest.py \
  --clean_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/clean_train/wav.scp \
  --codec_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_fixedrate_mixed/train/wav.scp \
  --utt2spk egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/clean_train/utt2spk \
  --codec_assignment_csv egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_fixedrate_mixed/train/codec_assignment.csv \
  --output_csv my_methods/note/cnceleb_fixedrate_pair_manifest.csv
```

## 经典前端特征处理与可视化

```powershell
python my_methods\tools\visualize_frontend_features.py `
  --wav_scp egs\3dspeaker\sv-cam++\data\CN_celeb_database\cnceleb\test\wav.scp `
  --utt_id test-id00891-speech-01-001 `
  --output_dir my_methods\exp\frontend_feature_demo `
  --sample_rate 16000 `
  --num_mel_bins 80 `
  --num_ceps 13 `
  --frame_length_ms 25 `
  --frame_shift_ms 10 `
  --variance_norm
```

说明：

- 会读取一条音频，输出波形图、FBank 图、MFCC 图，以及做过 CMVN 后的特征图。
- 同时保存对应的 `.npy` 文件，方便后续做分析、对比和论文插图。
- 如果你已经知道真实音频路径，也可以直接传 `--input_wav <path>`。
- 对 CN-Celeb 这类 Kaldi 风格 recipe，推荐用 `--wav_scp + --utt_id`，不要手动猜音频目录。

## 2. 训练 CA-AFC

推荐先做离线特征预提取，再训练。这样训练阶段不再重复读取音频、提 FBank、提 pitch。

### 2.1 预提取 pair features

```bash
python my_methods/tools/precompute_pair_features.py \
  --pair_manifest my_methods/note/cnceleb_fixedrate_pair_manifest.csv \
  --output_root my_methods/exp/ca_afc_features_cnceleb_fixedrate \
  --output_manifest my_methods/note/cnceleb_fixedrate_feature_manifest.csv \
  --sample_rate 16000
```

### 2.2 用离线特征训练 CA-AFC

```bash
python my_methods/scripts/train_ca_afc.py \
  --train_feature_manifest my_methods/note/cnceleb_fixedrate_feature_manifest.csv \
  --output_dir my_methods/exp/ca_afc_cnceleb_fixedrate \
  --campplus_model_bin pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
  --max_frames 300 \
  --batch_size 16 \
  --num_workers 0 \
  --pretrain_epochs 5 \
  --finetune_epochs 10 \
  --lambda_rec 1.0 \
  --lambda_emb 0.3 \
  --lambda_smooth 0.01 \
  --device cuda
```

新增训练观察与 smoke test 参数：

- `--log_interval 100`
  - 每 100 个 batch 打印一次当前 batch loss、累计耗时和当前 epoch 预计剩余时间。
- `--max_train_samples 0`
  - 训练集样本上限，`0` 表示使用全部训练样本。
- `--max_valid_samples 0`
  - 验证集样本上限，`0` 表示使用全部验证样本。

推荐先做 smoke test：

```powershell
python my_methods\scripts\train_ca_afc.py `
  --train_feature_manifest my_methods\note\cnceleb_fixedrate_feature_manifest.csv `
  --output_dir my_methods\exp\ca_afc_cnceleb_fixedrate_smoke `
  --campplus_model_bin pretrained\speech_campplus_sv_zh-cn_16k-common\campplus_cn_common.bin `
  --max_frames 300 `
  --batch_size 16 `
  --num_workers 0 `
  --pretrain_epochs 1 `
  --finetune_epochs 1 `
  --max_train_samples 4096 `
  --max_valid_samples 512 `
  --log_interval 20 `
  --lambda_rec 1.0 `
  --lambda_emb 0.3 `
  --lambda_smooth 0.01 `
  --device cuda
```

参数解释：

- `--train_feature_manifest`
  - 离线特征清单 CSV。每一行对应一个 `.pt` 文件，里面已经保存好 `clean_feat / codec_feat / aux_feat`。
- `--train_manifest`
  - 原始音频配对清单 CSV。只有在不使用 `--train_feature_manifest` 时才会读取音频并现场提特征。
- `--output_dir`
  - 训练输出目录，保存 `best_frontend.pt`、`checkpoints/*.pt`、`train_summary.json`。
- `--campplus_model_bin`
  - 预训练 CAM++ 权重文件。在第二阶段中作为冻结后端提供 embedding 一致性监督。
- `--max_frames 300`
  - 每个训练样本裁剪或补齐到 300 帧。用于统一 batch 的时间长度。
- `--batch_size 16`
  - 每次参数更新使用 16 对 clean/codec 样本。
- `--num_workers 0`
  - 离线特征训练时通常保持 0 或较小值即可，因为主要开销已经不在音频解码和 pitch 提取。
- `--pretrain_epochs 5`
  - 第一阶段训练轮数。此阶段只优化重建损失和时间平滑损失。
- `--finetune_epochs 10`
  - 第二阶段训练轮数。此阶段继续优化重建损失，同时加入冻结 CAM++ 的 embedding 一致性损失。
- `--lambda_rec 1.0`
  - 特征重建损失权重，是整个训练过程中的主监督项。
- `--lambda_emb 0.3`
  - embedding 一致性损失权重，只在第二阶段生效。
- `--lambda_smooth 0.01`
  - 时间平滑正则项权重，用于抑制残差输出帧间抖动。
- `--device cuda`
  - 指定训练设备。通常用 `cuda`，没有 GPU 时可改成 `cpu`。

断点续训：

```bash
python my_methods/scripts/train_ca_afc.py \
  --train_feature_manifest my_methods/note/cnceleb_fixedrate_feature_manifest.csv \
  --output_dir my_methods/exp/ca_afc_cnceleb_fixedrate \
  --campplus_model_bin pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
  --resume \
  --device cuda
```

## 3. 评测 CA-AFC + CAM++

```bash
python my_methods/scripts/run_ca_afc_codec_eval.py \
  --test_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/test/wav.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/trials/trials.lst \
  --frontend_ckpt my_methods/exp/ca_afc_cnceleb_fixedrate/best_frontend.pt \
  --campplus_model_bin pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
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

## 维护约定

- 后续新增 Python 脚本时，必须放到对应功能子目录下。
- 后续新增脚本时，必须在本文件补充：
- 文件作用。
- 关键函数或关键类的职责。
- 推荐执行命令。
- 如果脚本会生成实验结果文件，也要写清楚输出路径和产物类型。
