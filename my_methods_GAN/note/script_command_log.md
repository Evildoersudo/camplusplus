### 1.1启动并进入容器

```bash
cd ~/lkj/camplusplus
docker compose up -d
docker compose exec camplusplus bash
sudo chown -R dgx:dgx /home/dgx/lkj/camplusplus
```

激活虚拟环境
```bash
eval "$(micromamba shell hook --shell=bash)"
micromamba activate autodl-torch200-py38
```

### 1.2 容器内验证

先确认 `torch/torchaudio`：

```bash
python -c "import torch, torchaudio; print(torch.__version__); print(torchaudio.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

再确认 `ffmpeg` 编码器：

```bash
ffmpeg -hide_banner -encoders | grep -E 'libopus|aac|libvo_amrwbenc|pcm_alaw|pcm_mulaw'
```
## 推荐 Rec-only baseline 实验命令（按改进建议）

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest ... \
  --valid_manifest ... \
  --output_dir ... \
  --phase1_epochs 8 \
  --phase2_epochs 0 \
  --phase3_epochs 0 \
  --batch_size 24 \
  --segment_seconds 2.0 \
  --complex_weight 0.0 \
  --warmup_steps_g 200 \
  --no_valid_sv_metric \
  --lr_g_max 3e-4 \
  --lr_g_min 1e-5
```

说明：
- `--complex_weight 0.0` 关闭 complex loss，便于 loss 快速下降。
- `--warmup_steps_g 200` 缩短 warmup。
- `--no_valid_sv_metric` 关闭验证说话人指标，专注 rec-only。
# 脚本调用命令记录（统一维护）

本文件用于统一记录 my_methods_GAN 下脚本调用命令。

维护规则：

1. 新增脚本时必须补充调用命令。
2. 修改脚本参数或默认值后，必须同步更新对应命令。
3. 本文件为后续运行与复现实验的唯一命令索引。

## 变更记录

- 2026-04-03: 新增并登记脚本
  - my_methods_GAN/scripts/build_sv_codec_manifest.py
  - my_methods_GAN/scripts/train_sv_codec_restore_gan.py
  - my_methods_GAN/scripts/eval_sv_codec_restore_gan.py
- 2026-04-03: 新增并登记脚本（CN-Celeb 真配对数据流水线）
  - my_methods_GAN/scripts/prepare_cnceleb_truepair_data.py
  - my_methods_GAN/scripts/split_sv_manifest_by_speaker.py
  - my_methods_GAN/scripts/make_cnceleb_eval_scp.py
- 2026-04-03: 修改脚本并确认命令不变
  - my_methods_GAN/scripts/build_sv_codec_manifest.py
- 2026-04-03: 修改脚本并更新命令（AdamW + 线性热身余弦退火）
  - my_methods_GAN/scripts/train_sv_codec_restore_gan.py
- 2026-04-04: 修改脚本并更新命令（manifest 分层抽样）
  - my_methods_GAN/scripts/train_sv_codec_restore_gan.py
  - my_methods_GAN/scripts/split_sv_manifest_by_speaker.py
- 2026-04-04: 修改脚本并更新命令（断点续训 resume）
  - my_methods_GAN/scripts/train_sv_codec_restore_gan.py
- 2026-04-04: 修改脚本并更新命令（续训阶段边界修复）
  - my_methods_GAN/scripts/train_sv_codec_restore_gan.py
- 2026-04-04: 修改脚本并更新命令（WavLM 改为 HuggingFace from_pretrained）
  - my_methods_GAN/sv_codec_restore_gan/models/wavlm_wrapper.py
  - my_methods_GAN/scripts/train_sv_codec_restore_gan.py
  - my_methods_GAN/sv_codec_restore_gan/train/engine.py
- 2026-04-04: 修改脚本并更新命令（按 improved_2：phase-local LR + 双验证指标）
  - my_methods_GAN/sv_codec_restore_gan/train/engine.py
  - my_methods_GAN/scripts/train_sv_codec_restore_gan.py
- 2026-04-04: 新增实验命令（Experiment A: Rec-only baseline）
  - my_methods_GAN/scripts/train_sv_codec_restore_gan.py
- 2026-04-04: 新增脚本并登记命令（绘制 train.log 的 avg_train 曲线）
  - my_methods_GAN/scripts/plot_train_avg_curve.py
- 2026-04-05: 修改脚本并更新命令（评测 restored 分块推理 + CUDA OOM 自动缩块重试）
  - my_methods_GAN/scripts/eval_sv_codec_restore_gan.py
- 2026-04-05: 修改脚本并更新命令（评测 embedding 本地缓存：clean/coded/restored 分别缓存）
  - my_methods_GAN/scripts/eval_sv_codec_restore_gan.py
- 2026-04-05: 修改脚本并更新命令（按说话人导出 embedding npy）
  - my_methods_GAN/scripts/eval_sv_codec_restore_gan.py
- 2026-04-05: 修改脚本并更新命令（restored 分块批量推理加速）
  - my_methods_GAN/scripts/eval_sv_codec_restore_gan.py
- 2026-04-06: 新增脚本并登记命令（单脚本评测 CAMP++ on CN-Celeb 全量 eval）
  - my_methods_GAN/scripts/eval_campplus_cnceleb.py
- 2026-04-07: 新增脚本并登记命令（下载并加载 microsoft/wavlm-base-plus-sv 到本地目录）
  - my_methods_GAN/scripts/load_wavlm_sv_hf.py
- 2026-04-07: 新增脚本并登记命令（WavLM-SV 小规模 CN-Celeb 评测：EER/minDCF/accuracy）
  - my_methods_GAN/scripts/eval_wavlm_sv_cnceleb.py
- 2026-04-07: 修改脚本并更新命令（实验B：CAM++ teacher 可微 log-Mel 微调生成器）
  - my_methods_GAN/scripts/train_sv_codec_restore_gan.py
  - my_methods_GAN/sv_codec_restore_gan/train/engine.py
- 2026-04-07: 修改脚本并更新命令（训练日志打印 CAM++ 传回梯度统计）
  - my_methods_GAN/scripts/train_sv_codec_restore_gan.py
  - my_methods_GAN/sv_codec_restore_gan/train/engine.py
- 2026-04-07: 修改脚本（训练每步固定打印 rec_total/spk_raw/spk_weighted/generator_grad_norm）
  - my_methods_GAN/sv_codec_restore_gan/train/engine.py
- 2026-04-08: 修改脚本并更新命令（GAN改进：AM-Softmax说话人损失 + multi-codec训练）
  - my_methods_GAN/sv_codec_restore_gan/models/speaker_losses.py
  - my_methods_GAN/sv_codec_restore_gan/train/engine.py
  - my_methods_GAN/scripts/train_sv_codec_restore_gan.py
  - my_methods_GAN/sv_codec_restore_gan/data/dataset.py
  - my_methods_GAN/sv_codec_restore_gan/data/manifest.py
  - my_methods_GAN/scripts/build_sv_codec_manifest.py
- 2026-04-08: 修复脚本（plot_train_avg_curve 正则解析错误）
  - my_methods_GAN/scripts/plot_train_avg_curve.py
- 2026-04-08: 修改脚本并更新命令（plot_train_avg_curve 支持多指标子图）
  - my_methods_GAN/scripts/plot_train_avg_curve.py
- 2026-04-08: 修改脚本并更新命令（GAN改善第二版：对齐裁剪 + CAMP++ deep feature loss + MRSTFT稳态化）
  - my_methods_GAN/sv_codec_restore_gan/data/dataset.py
  - my_methods_GAN/sv_codec_restore_gan/models/campplus_wrapper.py
  - my_methods_GAN/sv_codec_restore_gan/models/losses.py
  - my_methods_GAN/sv_codec_restore_gan/train/engine.py
  - my_methods_GAN/scripts/train_sv_codec_restore_gan.py
- 2026-04-09: 修改脚本并更新命令（评测加速：plain/restored提取分离 + 条件拆分运行 + trial向量化打分）
  - my_methods_GAN/scripts/eval_sv_codec_restore_gan.py
- 2026-04-09: 修改脚本并更新命令（评测加速：plain分支 DataLoader 多进程预取 + 同帧长分桶批量 CAM++ 前向）
  - my_methods_GAN/scripts/eval_sv_codec_restore_gan.py
- 2026-04-09: 修改脚本并更新命令（评测支持按 trials 正负样本对分层抽样 + 仅抽样 utt 提取 embedding）
  - my_methods_GAN/scripts/eval_sv_codec_restore_gan.py
- 2026-04-10: 新增脚本并登记命令（按 epoch 汇总并绘制 spk_raw/spk_feat_raw 趋势）
  - my_methods_GAN/scripts/plot_epoch_spk_trend.py
- 2026-04-11: 修改脚本并更新命令（实验C phase3：GAN轻量微调预设 + phase3-only warm-start校验）
  - my_methods_GAN/scripts/train_sv_codec_restore_gan.py
  - my_methods_GAN/sv_codec_restore_gan/train/engine.py
- 2026-04-11: 修改脚本（实验C训练日志打印 GAN 相关项：gan_d_raw + 生成器侧加权项 gan_adv_weighted/gan_fm_weighted）
  - my_methods_GAN/sv_codec_restore_gan/train/engine.py
- 2026-04-11: 修改判别器与训练脚本（按判别器修改.md：STFT-MRD + 强化FM归约 + 实验C更强D策略 + GAN原始值/d_lr日志）
  - my_methods_GAN/sv_codec_restore_gan/models/discriminators.py
  - my_methods_GAN/sv_codec_restore_gan/models/losses.py
  - my_methods_GAN/scripts/train_sv_codec_restore_gan.py
  - my_methods_GAN/sv_codec_restore_gan/train/engine.py
- 2026-04-18: 新增脚本并登记命令（多 codec 的 coded-only 批量评测：EER/minDCF）
  - my_methods_GAN/scripts/eval_sv_coded_only_multi_codec.py
- 2026-04-19: 修改脚本并更新命令（plot_epoch_spk_trend 新增报告模式：三子图 + mean±std 阴影）
  - my_methods_GAN/scripts/plot_epoch_spk_trend.py
- 2026-04-20: 修改脚本并更新命令（Experiment A 客观质量评测加速：低频GC + STOI/PESQ多进程 + 推理chunk批量前向）
  - my_methods_GAN/scripts/eval_objective_quality_experiment_a.py
- 2026-04-20: 新增脚本并登记命令（Experiment A 客观语音质量评测：coded vs restored）
  - my_methods_GAN/scripts/eval_objective_quality_experiment_a.py
- 2026-04-20: 修改脚本并更新命令（Experiment A 评测新增进度打印、速度与 ETA）
  - my_methods_GAN/scripts/eval_objective_quality_experiment_a.py
- 2026-04-20: 修改脚本（修复分块 STFT 指标在短语音上的 padding 报错，自动跳过短块并回退到补零整句计算）
  - my_methods_GAN/scripts/eval_objective_quality_experiment_a.py
- 2026-04-22: 新增脚本并登记命令（绘制 clean 与 coded AMR-WB 时频图对比，大字号）
  - my_methods_GAN/scripts/plot_clean_coded_spectrogram.py
- 2026-04-22: 新增脚本并登记命令（coded 输入模型输出 restored，并绘制 clean/coded/restored 三联时频图）
  - my_methods_GAN/scripts/plot_clean_coded_restored_spectrogram.py

## 0. 环境准备

在项目根目录执行：

```bash
cd /home/dgx/lkj/camplusplus
```

## 1. 构建 clean/coded 配对 manifest

脚本：my_methods_GAN/scripts/build_sv_codec_manifest.py

```bash
python my_methods_GAN/scripts/build_sv_codec_manifest.py \
  --clean_root /path/to/clean_root \
  --coded_root /path/to/coded_root \
  --output_csv my_methods_GAN/exp/sv_codec_restore/manifest_pair.csv
```

multi-codec manifest（每条样本写入多个 codec 路径，训练时随机采样一个）：

```bash
python my_methods_GAN/scripts/build_sv_codec_manifest.py \
  --clean_root my_methods_GAN/data/cnceleb_truepair/clean_train_wav \
  --coded_root my_methods_GAN/data/cnceleb_truepair/coded_train_opus_16k \
              my_methods_GAN/data/cnceleb_truepair/coded_train_aac_32k \
              my_methods_GAN/data/cnceleb_truepair/coded_train_amrwb_23k \
  --output_csv my_methods_GAN/exp/sv_codec_restore/pair_manifest_multicodec.csv
```

## 2. 三阶段训练 SV-CodecRestoreGAN

脚本：my_methods_GAN/scripts/train_sv_codec_restore_gan.py

推荐配置 A（先跑通，单卡稳健默认）：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_main \
  --phase1_epochs 10 \
  --phase2_epochs 10 \
  --phase3_epochs 10 \
  --batch_size 24 \
  --num_workers 8 \
  --segment_seconds 2.0 \
  --train_sample_fraction 1.0 \
  --valid_sample_fraction 1.0 \
  --train_stratified_sample \
  --valid_stratified_sample \
  --emb_dim 48 \
  --num_blocks 5 \
  --hidden_units 100 \
  --attn_heads 4 \
  --lr_g_max 5e-4 \
  --lr_g_min 5e-6 \
  --lr_d_max 1e-4 \
  --lr_d_min 1e-6 \
  --warmup_steps_g 1000 \
  --warmup_steps_d 1000 \
  --weight_decay 1e-4 \
  --grad_clip 5.0 \
  --wavlm_root my_methods_GAN/pretrained/WavLM \
  --wavlm_ckpt my_methods_GAN/pretrained/WavLM/WavLM-Base+.pt \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --device cuda
```

推荐配置 A-1（仅用 1/4 训练数据，在线分层抽样）：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_q25.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_q25.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_main_q25 \
  --phase1_epochs 5 \
  --phase2_epochs 5 \
  --phase3_epochs 5 \
  --batch_size 24 \
  --num_workers 8 \
  --segment_seconds 2.0 \
  --train_sample_fraction 0.25 \
  --valid_sample_fraction 0.25 \
  --train_stratified_sample \
  --valid_stratified_sample \
  --emb_dim 48 \
  --num_blocks 5 \
  --hidden_units 100 \
  --attn_heads 4 \
  --lr_g_max 5e-4 \
  --lr_g_min 5e-6 \
  --lr_d_max 1e-4 \
  --lr_d_min 1e-6 \
  --warmup_steps_g 1000 \
  --warmup_steps_d 1000 \
  --weight_decay 1e-4 \
  --grad_clip 5.0 \
  --wavlm_root my_methods_GAN/pretrained/WavLM \
  --wavlm_ckpt my_methods_GAN/pretrained/WavLM/WavLM-Base+.pt \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --device cuda
```

断点续训（自动加载 output_dir/checkpoints 下最新 epoch_*.pt）：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_q25.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_q25.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_main_q25 \
  --phase1_epochs 5 \
  --phase2_epochs 5 \
  --phase3_epochs 5 \
  --batch_size 24 \
  --num_workers 8 \
  --segment_seconds 2.0 \
  --train_sample_fraction 0.25 \
  --valid_sample_fraction 0.25 \
  --train_stratified_sample \
  --valid_stratified_sample \
  --emb_dim 48 \
  --num_blocks 5 \
  --hidden_units 100 \
  --attn_heads 4 \
  --lr_g_max 5e-4 \
  --lr_g_min 5e-6 \
  --lr_d_max 1e-4 \
  --lr_d_min 1e-6 \
  --warmup_steps_g 1000 \
  --warmup_steps_d 1000 \
  --weight_decay 1e-4 \
  --grad_clip 5.0 \
  --wavlm_root my_methods_GAN/pretrained/WavLM \
  --wavlm_ckpt my_methods_GAN/pretrained/WavLM/WavLM-Base+.pt \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --resume \
  --device cuda
```

说明：默认续训会沿用 checkpoint 保存时的 phase 配置（修复阶段跳变问题）。

说明（improved_2 对齐）：

- 默认关闭 CAMP++ 训练损失（`--no_use_campplus_train_loss`），避免非可微 FBank 路径参与主训练目标。
- 默认开启验证说话人指标（`--valid_sv_metric`），训练日志同时输出 `valid_rec` 与 `sv_cos`。
- 学习率调度已改为 phase-local warmup-cosine，每个 phase 内独立计步。

Experiment A（Rec-only baseline，去掉 GAN/WavLM/CAM++）：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_pair_manifest_train_spksplit_opus16k.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_pair_manifest_valid_spksplit_opus16k.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore_vox/run_expA_rec_opus16_vox1_002_cws  \
  --cws_mode nonuniform \
  --cws_band_edges 0,64,160,257 \
  --phase1_epochs 10 \
  --phase2_epochs 0 \
  --phase3_epochs 0 \
  --batch_size 7 \
  --num_workers 16 \
  --segment_seconds 4.0 \
  --train_sample_fraction 1.0 \
  --valid_sample_fraction 1.0 \
  --train_stratified_sample \
  --valid_stratified_sample \
  --emb_dim 48 \
  --num_blocks 5 \
  --hidden_units 100 \
  --attn_heads 4 \
  --lr_g_max 3e-4 \
  --lr_g_min 1e-4 \
  --warmup_steps_g 200 \
  --si_sdr_weight 2 \
  --mrstft_weight 1 \
  --complex_weight 1 \
  --weight_decay 1e-4 \
  --grad_clip 30.0 \
  --si_sdr_weight 3 \
  --mrstft_weight 1 \
  --complex_weight 1 \
  --valid_sv_metric \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_en_voxceleb_16k/campplus_voxceleb.bin \
  --device cuda \
  --log_interval 1 
```
--wavlm_root my_methods_GAN/pretrained/WavLM \
--wavlm_ckpt my_methods_GAN/pretrained/WavLM/WavLM-Base+.pt \
--resume \

Experiment B（冻结 CAM++ teacher，使用可微 log-Mel 前端微调生成器）：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_q25.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_q25.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_expB_campplus_teacher_q25_second \
  --init_generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_q25/best_generator.pt \
  --phase1_epochs 0 \
  --phase2_epochs 8 \
  --phase3_epochs 0 \
  --batch_size 24 \
  --num_workers 8 \
  --segment_seconds 3.0 \
  --phase2_segment_seconds 3.0 \
  --train_sample_fraction 1.0 \
  --valid_sample_fraction 1.0 \
  --train_stratified_sample \
  --valid_stratified_sample \
  --lr_g_max 5e-5 \
  --lr_g_min 1e-5 \
  --warmup_steps_g 200 \
  --si_sdr_weight 2.0 \
  --mrstft_weight 0.5 \
  --complex_weight 0.1 \
  --use_campplus_train_loss \
  --campplus_frontend diff_mel \
  --spk_loss_weight 10 \
  --use_spk_amsoftmax \
  --spk_cls_loss_weight 1 \
  --spk_am_margin 0.2 \
  --spk_am_scale 30 \
  --phase3_no_gan \
  --phase3_no_wavlm \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --device cuda
```

Experiment B2（按 GAN 改善第二版：关闭 AM-Softmax，启用 deep feature loss，4s 对齐裁剪）：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/voxceleb2_results/vox2_frac0025_train_pair_opus16k.csv \
  --valid_manifest my_methods_GAN/exp/voxceleb2_results/vox2_frac0025_valid_pair_opus16k.csv \
  --output_dir my_methods_GAN/exp/voxceleb2_results/run_expB_opus16_clean_vox2_0025 \
  --init_generator_ckpt my_methods_GAN/exp/voxceleb2_results/run_expA_rec_opus16_clean_vox2_0025/best_generator.pt \
  --clean_passthrough_ratio 0.2 \
  --phase1_epochs 0 \
  --phase2_epochs 20 \
  --phase3_epochs 0 \
  --batch_size 7 \
  --num_workers 16 \
  --segment_seconds 4.0 \
  --phase2_segment_seconds 4.0 \
  --lr_g_max 5e-5 \
  --lr_g_min 1e-5 \
  --warmup_steps_g 2000 \
  --si_sdr_weight 1.0 \
  --mrstft_weight 0.5 \
  --complex_weight 0.0 \
  --use_campplus_train_loss \
  --use_campplus_feat_loss \
  --campplus_feat_layers block2,out_nonlinear \
  --campplus_feat_loss_weight 3 \
  --campplus_frontend diff_mel \
  --spk_loss_weight 9 \
  --no_use_spk_amsoftmax \
  --phase3_no_gan \
  --phase3_no_wavlm \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_en_voxceleb_16k/campplus_voxceleb.bin \
  --device cuda
```

**断点续训**
``` bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/voxceleb2_results/vox2_frac004_train_pair_opus16k.csv \
  --valid_manifest my_methods_GAN/exp/voxceleb2_results/vox2_frac004_valid_pair_opus16k.csv \
  --output_dir my_methods_GAN/exp/voxceleb2_results/sv_codec_restore_vox2/run_expB_opus16_vox2_008 \
  --resume \
  --resume_use_current_phase_config \
  --resume_ckpt my_methods_GAN/exp/voxceleb2_results/sv_codec_restore_vox2/run_expB_opus16_vox2_008/epoch_009.pt \
  --emb_dim 48 \
  --num_blocks 5 \
  --hidden_units 100 \
  --attn_heads 4 \
  --phase1_epochs 0 \
  --phase2_epochs 40 \
  --phase3_epochs 0 \
  --batch_size 7 \
  --num_workers 16 \
  --audio_preflight_items 2 \
  --segment_seconds 4.0 \
  --phase2_segment_seconds 4.0 \
  --lr_g_max 5e-5 \
  --lr_g_min 1e-5 \
  --warmup_steps_g 0 \
  --si_sdr_weight 1.0 \
  --mrstft_weight 0.5 \
  --complex_weight 0.0 \
  --use_campplus_train_loss \
  --use_campplus_feat_loss \
  --campplus_feat_layers block2,out_nonlinear \
  --campplus_feat_loss_weight 3 \
  --campplus_frontend diff_mel \
  --spk_loss_weight 9 \
  --no_use_spk_amsoftmax \
  --phase3_no_gan \
  --phase3_no_wavlm \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_en_voxceleb_16k/campplus_voxceleb.bin \
  --device cuda
```

```
  --init_generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_opus_amr_q25/best_generator_sv.pt \
```
Experiment C（按 实验phase3：从实验B最优checkpoint进入 phase3 轻量 GAN 微调）：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_q25.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_q25.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_expC_phase3_from_B \
  --init_generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/best_generator_sv.pt \
  --batch_size 20 \
  --num_workers 8 \
  --segment_seconds 4.0 \
  --phase3_segment_seconds 4.0 \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --experiment_c_preset \
  --experiment_c_epochs 10 \
  --device cuda
```

说明（实验C预设生效后会自动覆盖）：

- `phase1_epochs=0, phase2_epochs=0, phase3_epochs=experiment_c_epochs`
- `phase3_use_gan=True, use_mbd=False, phase3_use_wavlm=False`
- `lr_g_max/min=2e-5/5e-6, lr_d_max/min=2e-5/5e-6, warmup_steps_g/d=50`
- `si_sdr_weight=1.0, mrstft_weight=0.5, complex_weight=0.0`
- `use_campplus_train_loss=True, campplus_frontend=diff_mel`
- `spk_loss_weight=1.0, adv_loss_weight=0.05, fm_loss_weight=0.05`
- `use_campplus_feat_loss=False, use_spk_amsoftmax=False`
- phase3-only 训练若未提供 `--init_generator_ckpt` 且未 `--resume`，会报错阻止从头训。

新增参数说明：

- `--campplus_frontend {kaldi,diff_mel}`：训练/验证时 CAM++ 特征前端，`diff_mel` 为可微 log-Mel。
- `--init_generator_ckpt`：仅加载生成器权重做 warm-start（不加载优化器和判别器）。
- 当 `--init_generator_ckpt` 指向旧实验 checkpoint 且网络宽度参数不一致时，会自动读取 checkpoint 内保存的 `emb_dim/num_blocks/hidden_units/attn_heads` 并覆盖当前配置，避免权重 shape mismatch。
- `--debug_campplus_grad`：开启后打印由 CAM++ speaker loss 传回 restored waveform 的梯度统计。
- `--debug_campplus_grad_interval`：梯度统计打印间隔（step），默认 `50`。
- `--use_spk_amsoftmax`：开启 AM-Softmax 说话人分类损失（基于 CAMP++ embedding）。
- `--spk_cls_loss_weight`：AM-Softmax loss 权重。
- `--spk_am_margin`：AM-Softmax margin。
- `--spk_am_scale`：AM-Softmax scale。
- `--adv_loss_weight` 默认值已调整为 `0.5`（与 GAN 改进建议一致）。
- `--use_campplus_feat_loss`：开启 CAMP++ 中间层 deep feature L1 loss。
- `--campplus_feat_layers`：指定 deep feature loss 使用的 CAMP++ 层名（逗号分隔）。
- `--campplus_feat_loss_weight`：deep feature loss 权重，建议从 `0.5` 起步。
- 数据裁剪已改为 clean/coded 同起点对齐裁剪，避免 pair 错位。
- MRSTFT 的 spectral convergence 已改为 per-sample 统计后再 batch 平均，降低 batch 能量主导带来的波动。

实验 B 开启 CAM++ 梯度打印（便于确认梯度回传）：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_q25.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_q25.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_expB_campplus_teacher_q25 \
  --init_generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_q25/best_generator.pt \
  --phase1_epochs 0 \
  --phase2_epochs 8 \
  --phase3_epochs 0 \
  --batch_size 24 \
  --num_workers 8 \
  --segment_seconds 4.0 \
  --phase2_segment_seconds 4.0 \
  --lr_g_max 5e-5 \
  --lr_g_min 1e-5 \
  --warmup_steps_g 200 \
  --si_sdr_weight 1.0 \
  --mrstft_weight 1.0 \
  --complex_weight 0.1 \
  --use_campplus_train_loss \
  --campplus_frontend diff_mel \
  --spk_loss_weight 10 \
  --debug_campplus_grad \
  --debug_campplus_grad_interval 10 \
  --phase3_no_gan \
  --phase3_no_wavlm \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --device cuda
```

若你确实要用当前命令中的 phase 参数覆盖 checkpoint，请显式追加：

```bash
--resume_use_current_phase_config
```

断点续训（指定 checkpoint）：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_q25.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_q25.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_main_q25 \
  --phase1_epochs 0 \
  --phase2_epochs 5 \
  --phase3_epochs 5 \
  --batch_size 24 \
  --num_workers 8 \
  --segment_seconds 2.0 \
  --train_sample_fraction 0.25 \
  --valid_sample_fraction 0.25 \
  --train_stratified_sample \
  --valid_stratified_sample \
  --emb_dim 48 \
  --num_blocks 5 \
  --hidden_units 100 \
  --attn_heads 4 \
  --lr_g_max 5e-4 \
  --lr_g_min 5e-6 \
  --lr_d_max 1e-4 \
  --lr_d_min 1e-6 \
  --warmup_steps_g 1000 \
  --warmup_steps_d 1000 \
  --weight_decay 1e-4 \
  --grad_clip 5.0 \
  --wavlm_root my_methods_GAN/pretrained/WavLM \
  --wavlm_ckpt my_methods_GAN/pretrained/WavLM/WavLM-Base+.pt \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --resume \
  --resume_ckpt my_methods_GAN/exp/sv_codec_restore/run_main_q25/checkpoints/epoch_005.pt \
  --device cuda
```

推荐配置 B（大数据更保守）：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_main_large \
  --phase1_epochs 10 \
  --phase2_epochs 10 \
  --phase3_epochs 10 \
  --batch_size 4 \
  --num_workers 4 \
  --segment_seconds 2.0 \
  --emb_dim 48 \
  --num_blocks 5 \
  --hidden_units 100 \
  --attn_heads 4 \
  --lr_g_max 5e-4 \
  --lr_g_min 1e-5 \
  --lr_d_max 1e-4 \
  --lr_d_min 2e-6 \
  --warmup_steps_g 3000 \
  --warmup_steps_d 3000 \
  --weight_decay 1e-4 \
  --grad_clip 5.0 \
  --wavlm_root /home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/WavLM \
  --wavlm_ckpt /home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/WavLM/WavLM-Base+.pt \
  --campplus_ckpt /home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --device cuda
```

WavLM（HuggingFace）加载参数说明：

```bash
--wavlm_model_id microsoft/wavlm-base-plus-sv
--wavlm_cache_dir /path/to/hf_cache   # 可选
```

本地 WavLM 脚本加载（当前默认）要求：

```bash
my_methods_GAN/pretrained/WavLM/WavLM-Base+.pt
```

若在线 HuggingFace 连接受限，可将 `--wavlm_model_id` 指向本地 HF 格式目录（仍通过 from_pretrained 加载）：

```bash
--wavlm_model_id /workspace/camplusplus/my_methods_GAN/pretrained/WavLM/wavlm-base-plus-sv
```

WavLM 冒烟验证（已实测可进入训练并完成 1 epoch）：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_q25.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_q25.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_wavlm_hf_smoke \
  --phase1_epochs 0 \
  --phase2_epochs 0 \
  --phase3_epochs 1 \
  --batch_size 1 \
  --num_workers 0 \
  --segment_seconds 2.0 \
  --phase3_segment_seconds 2.0 \
  --train_sample_fraction 0.0004 \
  --valid_sample_fraction 0.0004 \
  --no_train_stratified_sample \
  --no_valid_stratified_sample \
  --phase3_use_wavlm \
  --phase3_no_gan \
  --wavlm_model_id /workspace/camplusplus/my_methods_GAN/pretrained/WavLM/wavlm-base-plus-sv \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --device cpu
```

## 3. 评测 clean/coded/restored 的 EER/minDCF

脚本：my_methods_GAN/scripts/eval_sv_codec_restore_gan.py

```bash
python my_methods_GAN/scripts/eval_sv_codec_restore_gan.py \
  --clean_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_clean.scp \
  --coded_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_coded_amrwb1585.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac/eval/lists/trials.lst \
  --trial_sample_fraction 1.0 \
  --conditions clean,coded,restored \
  --generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_opus_amrwb/checkpoints/epoch_010.pt \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/eval_results/opus16k+amrwb1585_B_amrwb.json \
  --restore_chunk_seconds 8 \
  --restore_overlap_seconds 0.1 \
  --restore_chunk_batch_size 128 \
  --restore_auto_shrink \
  --restore_min_chunk_seconds 1.0 \
  --restore_chunk_shrink_factor 0.7 \
  --use_cache \
  --cache_incremental \
  --cache_save_every 100 \
  --plain_loader_batch_size 64 \
  --plain_num_workers 8 \
  --plain_camp_batch_size 64 \
  --cache_dir my_methods_GAN/exp/sv_codec_restore/run_main/amrwb/emb_cache \
  --dump_speaker_npy_dir my_methods_GAN/exp/sv_codec_restore/run_main/amrwb/speaker_amrwb_emb_npy \
  --speaker_id_sep / \
  --speaker_id_field 0 \
  --device cuda
```

参数说明：

- `--restore_chunk_seconds`：restored 路径分块长度（秒），默认 8。
- `--restore_overlap_seconds`：相邻块重叠长度（秒），默认 0.1。
- `--restore_chunk_batch_size`：restored 分块批量前向 batch size，默认 16；显存允许可尝试 32 提速。
- `--restore_auto_shrink`：遇到 CUDA OOM 自动缩小 chunk 并重试（默认开启）。
- `--restore_min_chunk_seconds`：自动缩块最小下限（秒），默认 1.0。
- `--restore_chunk_shrink_factor`：每次 OOM 后的缩放比例，默认 0.7。
- `--use_cache`：启用本地 embedding 缓存（默认开启）。
- `--cache_incremental`：按间隔增量写盘并支持中断续跑（默认开启）。
- `--cache_save_every`：每 N 条 utt 写一次增量缓存分片，默认 500。
- `--trial_sample_fraction`：按标签分层抽样比例，默认 1.0（全量）；例如 `0.2` 表示在正/负样本中分别抽取约 20%。
- `--trial_sample_total`：分层抽样后总 trial 数，`>0` 时优先于 `--trial_sample_fraction`。
- `--trial_sample_pos_count`：显式指定抽样正样本对数量；`>0` 时启用按类计数抽样。
- `--trial_sample_neg_count`：显式指定抽样负样本对数量；`>0` 时启用按类计数抽样。
- `--trial_sample_seed`：分层抽样随机种子，默认 42。
- `--plain_loader_batch_size`：clean/coded 分支 DataLoader 取样批大小（默认 64）。
- `--plain_num_workers`：clean/coded 分支 DataLoader worker 数（默认 4）。
- `--plain_camp_batch_size`：clean/coded 分支 CAM++ 前向批大小（默认 32，按相同帧长分桶后批处理）。
- `--cache_dir`：缓存目录，默认 `<output_json目录>/emb_cache`。
- `--overwrite_cache`：强制重算并覆盖已有缓存。
- `--conditions`：按条件评测，可选 `clean`、`coded`、`restored`，逗号分隔；用于拆分评测避免重复耗时。
- `--generator_ckpt`：仅当 `--conditions` 包含 `restored` 时必填。
- `--dump_speaker_npy_dir`：按 clean/coded/restored 分目录导出每个说话人的 embedding `.npy`。
- `--speaker_id_sep`：从 utt 提取说话人 ID 的分隔符，默认 `/`。
- `--speaker_id_field`：分隔后取第几个字段作为说话人 ID，默认 `0`。

按条件拆分运行示例（推荐先跑 clean/coded，再单独调 restored）：

```bash
python my_methods_GAN/scripts/eval_sv_codec_restore_gan.py \
  --clean_wav_scp my_methods_GAN/exp/LibriSpeech_results/eval_clean.scp \
  --coded_wav_scp my_methods_GAN/exp/LibriSpeech_results/eval_coded_opus16k.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/raw_data/vox1/test_lists/veri_test2.txt \
  --conditions clean,coded \
  --backend_type campplus \
  --backend_ckpt my_methods_GAN/pretrained/speech_campplus_sv_en_voxceleb_16k/campplus_voxceleb.bin \
  --output_json my_methods_GAN/exp/LibriSpeech_results/run_main/eval_results/opus16_coded_campplus.json \
  --use_cache \
  --cache_incremental \
  --cache_save_every 100 \
  --plain_loader_batch_size 64 \
  --plain_num_workers 4 \
  --plain_camp_batch_size 32 \
  --cache_dir my_methods_GAN/exp/LibriSpeech_results/run_main/opus/emb_cache \
  --device cuda
```

```bash
python my_methods_GAN/scripts/eval_sv_codec_restore_gan.py \
  --clean_wav_scp my_methods_GAN/exp/LibriSpeech_results/eval_clean.scp \
  --coded_wav_scp my_methods_GAN/exp/LibriSpeech_results/eval_coded_opus16k.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/raw_data/vox1/test_lists/veri_test2.txt \
  --conditions restored \
  --generator_ckpt my_methods_GAN/exp/LibriSpeech_results/run_expB_opus_16_libri_aug_100_temp/best_generator_sv.pt \
  --backend_type campplus \
  --backend_ckpt my_methods_GAN/pretrained/speech_campplus_sv_en_voxceleb_16k/campplus_voxceleb.bin \
  --output_json my_methods_GAN/exp/LibriSpeech_results/run_main/eval_results/opus16k_libri100_aug.json \
  --restore_chunk_seconds 8 \
  --restore_overlap_seconds 0.1 \
  --restore_chunk_batch_size 32 \
  --use_cache \
  --cache_incremental \
  --cache_save_every 200 \
  --cache_dir my_methods_GAN/exp/LibriSpeech_results/run_main/opus/emb_cache \
  --device cuda
```
```
  --trial_sample_pos_count 2000 \
  --trial_sample_neg_count 2000 \
  --trial_sample_seed 42 \
```
## 3.1 单脚本评测 CAMP++（CN-Celeb 全量测试集）

脚本：my_methods_GAN/scripts/eval_campplus_cnceleb.py

```bash
python my_methods_GAN/scripts/eval_campplus_cnceleb.py \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --cnceleb_root egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/campplus_cnceleb_eval.json \
  --device cuda
```

先做映射体检（不提 embedding、不跑全量打分）：

```bash
python my_methods_GAN/scripts/eval_campplus_cnceleb.py \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --cnceleb_root egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/campplus_cnceleb_eval.json \
  --dry_run \
  --device cuda
```

说明：

- 该脚本单独完成全流程：读取 trials、自动索引 `eval` 音频、提取 CAMP++ embedding、计算 EER/minDCF。
- 默认使用 `<cnceleb_root>/eval/lists/trials.lst` 和 `<cnceleb_root>/eval`，无需额外生成 scp。
- 若路径不同，可通过 `--trials_file`、`--eval_audio_root` 显式覆盖。
- `--dry_run`：仅检查 trials 到音频映射覆盖率（快速排查扩展名/路径问题）。

## 3.2 绘制 train.log 的 avg_train 曲线（按步长抽样）

脚本：my_methods_GAN/scripts/plot_train_avg_curve.py

每 40 个 step 取一个点：

```bash
python my_methods_GAN/scripts/plot_train_avg_curve.py \
  --log_file my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_q25_third/train.log \
  --sample_every 40 \
  --output_png my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_q25_third/avg_train_curve_s40.png
```

## 3.3 下载并加载 WavLM-SV（HuggingFace）

脚本：my_methods_GAN/scripts/load_wavlm_sv_hf.py

下载并保存到 `my_methods_GAN/pretrained/WavLM_SV`，同时运行相似度 demo：

```bash
python my_methods_GAN/scripts/load_wavlm_sv_hf.py \
  --model_id microsoft/wavlm-base-plus-sv \
  --output_dir my_methods_GAN/pretrained/WavLM_SV \
  --device cuda
```

仅下载保存模型文件，不跑 demo：

```bash
python my_methods_GAN/scripts/load_wavlm_sv_hf.py \
  --model_id microsoft/wavlm-base-plus-sv \
  --output_dir my_methods_GAN/pretrained/WavLM_SV \
  --skip_demo
```

## 3.4 WavLM-SV 小规模 CN-Celeb 评测（EER/minDCF/accuracy）

脚本：my_methods_GAN/scripts/eval_wavlm_sv_cnceleb.py

快速小评测（例如 5000 对 trials）：

```bash
python my_methods_GAN/scripts/eval_wavlm_sv_cnceleb.py \
  --model_dir my_methods_GAN/pretrained/WavLM_SV \
  --cnceleb_root egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac \
  --max_trials 5000 \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/wavlm_sv_cnceleb_small_eval.json \
  --device cuda
```

说明：

- 输出包含 `eer_percent`、`min_dcf`、`acc_at_eer_threshold`、`best_accuracy`。
- 默认 trials 路径：`<cnceleb_root>/eval/lists/trials.lst`。
- 默认音频索引路径：`<cnceleb_root>/eval`（支持 `wav/flac/mp3/m4a/ogg/opus`）。

每 60 个 step 取一个点：

```bash
python my_methods_GAN/scripts/plot_train_avg_curve.py \
  --log_file my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_q25_fast/train.log \
  --sample_every 60 \
  --output_png my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_q25_fast/avg_train_curve_s60.png
```

## 3.5 按 epoch 汇总 speaker loss 趋势（均值/标准差）

脚本：my_methods_GAN/scripts/plot_epoch_spk_trend.py

导出 phase2 的 `spk_raw/spk_feat_raw` 每个 epoch 统计（当前环境无 matplotlib 时至少会导出 CSV）：

```bash
python my_methods_GAN/scripts/plot_epoch_spk_trend.py \
  --log_file my_methods_GAN/exp/voxceleb2_results/run_expB_opus16_clean_vox2_0025/train.log \
  --phase phase2 \
  --metrics spk_raw,spk_feat_raw,si_sdr,mrstft \
  --output_csv my_methods_GAN/exp/voxceleb2_results/run_expB_opus16_clean_vox2_0025/epoch_phase2spk_trend.csv \
  --output_png my_methods_GAN/exp/voxceleb2_results/run_expB_opus16_clean_vox2_0025/epoch_phase2_spk_trend.png
```

说明：

- 输出 CSV 字段包含：`*_mean`、`*_std`、`*_min`、`*_max`。
- 若环境安装了 `matplotlib`，会同时生成 PNG 趋势图（均值 + mean±std 阴影）。

中期报告模式（从 `train_summary.json` + `epoch_phase2_spk_trend.csv` 读取 3 条曲线）：

```bash
python my_methods_GAN/scripts/plot_epoch_spk_trend.py \
  --plot_report_trend \
  --summary_json my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/train_summary.json \
  --trend_csv my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/epoch_phase2_spk_trend.csv \
  --report_output_csv my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/epoch_phase2_validsv_spk_trend.csv \
  --output_png my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/epoch_phase2_validsv_spk_trend.png \
  --title "ExpB2 Phase2 Validation SV and Speaker-Loss Trend"
```

说明：

- 报告模式输出 3 个子图：`valid_sv_cos`、`spk_raw_mean`、`spk_feat_raw_mean`。
- 每个子图都绘制 `mean±std` 阴影；`spk_*` 的 std 来自 `epoch_phase2_spk_trend.csv`。
- 若 `train_summary.json` 无 `valid_sv_cos` 的 std 字段，会自动回退为 `0.0` 并打印 warning。

可选参数补充：

- `--epoch_range`：只绘制指定 epoch（例如 `1-10` 或 `1,3,5-8`）。
- `--valid_sv_source`：`summary_json` 或 `train_log`，用于控制 `valid_sv_cos` 来源。
- `--valid_sv_log_file`：当 `--valid_sv_source train_log` 时指定日志路径（默认尝试 `trend_csv` 同目录下的 `train.log`）。

示例 1（只画 epoch 1-10，`valid_sv_cos` 来自 train_summary.json）：

```bash
python my_methods_GAN/scripts/plot_epoch_spk_trend.py \
  --plot_report_trend \
  --phase phase2 \
  --valid_sv_source summary_json \
  --summary_json my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/train_summary.json \
  --trend_csv my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/epoch_phase2_spk_trend.csv \
  --epoch_range 1-10 \
  --report_output_csv my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/epoch_phase2_validsv_spk_trend_ep1_10.csv \
  --output_png my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/epoch_phase2_validsv_spk_trend_ep1_10.png
```

示例 2（断点续训场景，`valid_sv_cos` 改从 train.log 读取）：

```bash
python my_methods_GAN/scripts/plot_epoch_spk_trend.py \
  --plot_report_trend \
  --phase phase2 \
  --valid_sv_source train_log \
  --valid_sv_log_file my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_opus_amrwb/train.log \
  --trend_csv my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_opus_amrwb/epoch_phase2_spk_trend.csv \
  --epoch_range 1-20 \
  --report_output_csv my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_opus_amrwb/epoch_phase2_validsv_spk_trend_fromlog_ep1_10.csv \
  --output_png my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_opus_amrwb/epoch_phase2_validsv_spk_trend_fromlog_ep1_10.png
```

## 3.6 多 codec coded-only 批量评测（EER/minDCF）

脚本：my_methods_GAN/scripts/eval_sv_coded_only_multi_codec.py

一次命令评测 Opus、AMR-WB、AAC、G.711（coded-only）：

```bash
python my_methods_GAN/scripts/eval_sv_coded_only_multi_codec.py \
  --codec_scp opus16k=my_methods_GAN/exp/sv_codec_restore/eval_coded_opus16k.scp,amrwb1585=my_methods_GAN/exp/sv_codec_restore/eval_coded_amrwb1585.scp,aac32k=my_methods_GAN/exp/sv_codec_restore/eval_coded_aac32k.scp,g711mulaw=my_methods_GAN/exp/sv_codec_restore/eval_coded_g711mulaw.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac/eval/lists/trials.lst \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/eval_results/coded_only_4codec.json \
  --use_cache \
  --cache_incremental \
  --cache_save_every 200 \
  --cache_dir my_methods_GAN/exp/sv_codec_restore/run_main/coded_only_4codec/emb_cache \
  --plain_loader_batch_size 64 \
  --plain_num_workers 8 \
  --plain_camp_batch_size 64 \
  --device cuda
```

不使用缓存（完全重算）：

```bash
python my_methods_GAN/scripts/eval_sv_coded_only_multi_codec.py \
  --codec_scp opus16k=my_methods_GAN/exp/sv_codec_restore/eval_coded_opus16k.scp,amrwb1585=my_methods_GAN/exp/sv_codec_restore/eval_coded_amrwb1585.scp,aac32k=my_methods_GAN/exp/sv_codec_restore/eval_coded_aac32k.scp,g711mulaw=my_methods_GAN/exp/sv_codec_restore/eval_coded_g711mulaw.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac/eval/lists/trials.lst \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/eval_results/coded_only_4codec_nocache.json \
  --no_use_cache \
  --plain_loader_batch_size 64 \
  --plain_num_workers 8 \
  --plain_camp_batch_size 64 \
  --device cuda
```

说明：

- `--codec_scp` 使用 `codec_tag=scp_path` 形式，多个条目用逗号分隔。
- 输出 JSON 每项包含 `condition`、`num_trials`、`eer_percent`、`min_dcf`、`missing_utts`。
- 若你尚未生成 `eval_coded_aac32k.scp` 或 `eval_coded_g711mulaw.scp`，请先用 `make_cnceleb_eval_scp.py` 生成对应 scp。

## 3.7 一键流水线：从 eval_clean 到四 codec coded-only 评测

适用场景：还没有完整的 AAC/G.711 eval coded 数据与 scp。

### 第一步：由 eval_clean 生成四种 codec 的 eval coded wav

```bash
python my_methods_GAN/scripts/transcode_clean_wav_to_codec.py \
  --clean_root egs/3dspeaker/sv-cam++/data/raw_data/vox1/test/wav \
  --coded_root my_methods_GAN/data/LibriSpeech/test_data/eval_coded_opus_16k \
  --codec opus \
  --bitrate 16k \
  --sample_rate 16000 \
  --workers 24

python my_methods_GAN/scripts/transcode_clean_wav_to_codec.py \
  --clean_root /root/autodl-tmp/SC_data/data/eval_clean \
  --coded_root /root/autodl-tmp/SC_data/data/eval_coded_amrwb_885 \
  --codec amrwb \
  --bitrate 8.85k \
  --sample_rate 16000 \
  --workers 16

python my_methods_GAN/scripts/transcode_clean_wav_to_codec.py \
  --clean_root my_methods_GAN/data/cnceleb_truepair/eval_clean \
  --coded_root my_methods_GAN/data/cnceleb_truepair/eval_coded_aac_32k \
  --codec aac \
  --bitrate 32k \
  --sample_rate 16000 \
  --workers 16

python my_methods_GAN/scripts/transcode_clean_wav_to_codec.py \
  --clean_root my_methods_GAN/data/cnceleb_truepair/eval_clean \
  --coded_root my_methods_GAN/data/cnceleb_truepair/eval_coded_g711mulaw \
  --codec g711_mulaw \
  --sample_rate 16000 \
  --workers 16
```

### 第二步：生成四种 codec 的 eval scp（复用已有 clean scp）

```bash
python my_methods_GAN/scripts/make_cnceleb_eval_scp.py \
  --trials_file egs/3dspeaker/sv-cam++/data/raw_data/vox1/test_lists/veri_test2.txt \
  --eval_clean_dir egs/3dspeaker/sv-cam++/data/raw_data/vox1/test/wav \
  --eval_coded_dir my_methods_GAN/data/LibriSpeech/test_data/eval_coded_opus_16k \
  --coded_scp_out my_methods_GAN/exp/LibriSpeech_results/eval_coded_opus16k.scp \
  --clean_scp_out my_methods_GAN/exp/LibriSpeech_results/eval_clean.scp \
  --strict

python my_methods_GAN/scripts/make_cnceleb_eval_scp.py \
  --trials_file /root/autodl-tmp/raw_data/CN-Celeb_flac/eval/lists/trials.lst \
  --eval_clean_dir /root/autodl-tmp/SC_data/data/eval_clean \
  --eval_coded_dir /root/autodl-tmp/SC_data/data/eval_coded_amrwb_885 \
  --coded_scp_out my_methods_GAN/exp/sv_codec_restore/eval_coded_amrwb885.scp \
  --skip_clean_scp \
  --strict

python my_methods_GAN/scripts/make_cnceleb_eval_scp.py \
  --trials_file egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac/eval/lists/trials.lst \
  --eval_clean_dir my_methods_GAN/data/cnceleb_truepair/eval_clean \
  --eval_coded_dir my_methods_GAN/data/cnceleb_truepair/eval_coded_aac_32k \
  --coded_scp_out my_methods_GAN/exp/sv_codec_restore/eval_coded_aac32k.scp \
  --skip_clean_scp \
  --strict

python my_methods_GAN/scripts/make_cnceleb_eval_scp.py \
  --trials_file egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac/eval/lists/trials.lst \
  --eval_clean_dir my_methods_GAN/data/cnceleb_truepair/eval_clean \
  --eval_coded_dir my_methods_GAN/data/cnceleb_truepair/eval_coded_g711mulaw \
  --coded_scp_out my_methods_GAN/exp/sv_codec_restore/eval_coded_g711mulaw.scp \
  --skip_clean_scp \
  --strict
```

### 第三步：批量计算四 codec 的 coded-only EER/minDCF

```bash
python my_methods_GAN/scripts/eval_sv_coded_only_multi_codec.py \
  --codec_scp opus16k=my_methods_GAN/exp/sv_codec_restore/eval_coded_opus16k.scp,amrwb1585=my_methods_GAN/exp/sv_codec_restore/eval_coded_amrwb1585.scp,aac32k=my_methods_GAN/exp/sv_codec_restore/eval_coded_aac32k.scp,g711mulaw=my_methods_GAN/exp/sv_codec_restore/eval_coded_g711mulaw.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac/eval/lists/trials.lst \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/eval_results/coded_only_4codec.json \
  --use_cache \
  --cache_incremental \
  --cache_save_every 200 \
  --cache_dir my_methods_GAN/exp/sv_codec_restore/run_main/coded_only_4codec/emb_cache \
  --plain_loader_batch_size 64 \
  --plain_num_workers 8 \
  --plain_camp_batch_size 64 \
  --device cuda
```

## 4. 模块入口说明（非直接脚本）

以下文件是训练/评测脚本调用的核心模块：

- my_methods_GAN/sv_codec_restore_gan/data/manifest.py
- my_methods_GAN/sv_codec_restore_gan/data/dataset.py
- my_methods_GAN/sv_codec_restore_gan/models/generator.py
- my_methods_GAN/sv_codec_restore_gan/models/discriminators.py
- my_methods_GAN/sv_codec_restore_gan/models/losses.py
- my_methods_GAN/sv_codec_restore_gan/models/wavlm_wrapper.py
- my_methods_GAN/sv_codec_restore_gan/models/campplus_wrapper.py
- my_methods_GAN/sv_codec_restore_gan/train/engine.py

## 5. 从原始 CN-Celeb 生成 true-pair 数据

脚本：my_methods_GAN/scripts/prepare_cnceleb_truepair_data.py

```bash
python my_methods_GAN/scripts/prepare_cnceleb_truepair_data.py \
  --raw_root /root/rivermind-data/raw_data/vox1/train/wav \
  --output_root /root/rivermind-data/experiment_data_voxceleb/train_data/opus \
  --codec opus \
  --bitrate 16k \
  --sample_rate 16000 \
  --workers 16 \
  --overwrite
```

该命令会生成：

- my_methods_GAN/data/cnceleb_truepair/clean_train_wav
- my_methods_GAN/data/cnceleb_truepair/coded_train_opus_16k
- my_methods_GAN/data/cnceleb_truepair/eval_clean
- my_methods_GAN/data/cnceleb_truepair/eval_coded_opus_16k

## 6. 由 true-pair 目录生成训练 pair manifest

脚本：my_methods_GAN/scripts/build_sv_codec_manifest.py

```bash
python my_methods_GAN/scripts/build_sv_codec_manifest.py \
  --clean_root my_methods_GAN/data/cnceleb_truepair/clean_train_wav \
  --coded_root my_methods_GAN/data/cnceleb_truepair/coded_train_opus_16k \
  --output_csv my_methods_GAN/exp/sv_codec_restore/pair_manifest_all.csv
```

## 7. 按说话人切分 train/valid manifest

脚本：my_methods_GAN/scripts/split_sv_manifest_by_speaker.py

```bash
python my_methods_GAN/scripts/split_sv_manifest_by_speaker.py \
  --input_manifest my_methods_GAN/exp/sv_codec_restore/pair_manifest_all.csv \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest.csv \
  --valid_ratio 0.1 \
  --seed 42
```

按 speaker 切分并离线抽样 1/4（分层）：

```bash
python my_methods_GAN/scripts/split_sv_manifest_by_speaker.py \
  --input_manifest my_methods_GAN/exp/LibriSpeech_results/LibriSpeech_100_clean_manifest_all.csv \
  --train_manifest my_methods_GAN/exp/LibriSpeech_results/train_manifest_opus_100.csv \
  --valid_manifest my_methods_GAN/exp/LibriSpeech_results/valid_manifest_opus_100.csv \
  --valid_ratio 0.1 \
  --train_fraction 1 \
  --valid_fraction 1 \
  --stratified \
  --seed 42
```

### 7.1 LibriSpeech Kaldi 风格声学增强

脚本：my_methods_GAN/scripts/augment_librispeech_kaldi_style.py

这个脚本会保持 LibriSpeech 原始目录组织格式：

```text
speaker_id/chapter_id/utterance_id.wav
```

每条输入语音只输出一条目标语音，比例为：

```text
clean 50%
reverb-only 12.5%
noise-only 12.5%
music-only 12.5%
babble-only 12.5%
```

默认设置：

- RIR: `RIRS_NOISES/simulated_rirs/{smallroom,mediumroom}`，两类房间 1:1 随机
- MUSAN noise SNR: `0,5,10,15`
- MUSAN music SNR: `5,8,10,15`
- MUSAN speech/babble SNR: `13,15,17,20`
- babble 源数量：`3~7`
- 输出：单声道 16 kHz PCM WAV，峰值限制到 `0.95`

生成增强后的 LibriSpeech 目标语音：

```bash
python my_methods_GAN/scripts/augment_librispeech_kaldi_style.py \
  --input_root my_methods_GAN/data/LibriSpeech/train-clean-100 \
  --output_root my_methods_GAN/data/LibriSpeech/train-clean-100-kaldi-aug \
  --rir_root egs/3dspeaker/sv-cam++/data/raw_data/RIRS_NOISES \
  --musan_root egs/3dspeaker/sv-cam++/data/raw_data/musan \
  --metadata_csv my_methods_GAN/exp/LibriSpeech_results/LibriSpeech_100_kaldi_aug_metadata.csv \
  --workers 16 \
  --seed 42
```

如果之前已经生成过该目录，建议加 `--overwrite` 或先删除旧目录，避免旧版本残留文件被跳过：

```bash
python my_methods_GAN/scripts/augment_librispeech_kaldi_style.py \
  --input_root my_methods_GAN/data/LibriSpeech/train-clean-100 \
  --output_root my_methods_GAN/data/LibriSpeech/train-clean-100-kaldi-aug \
  --rir_root egs/3dspeaker/sv-cam++/data/raw_data/RIRS_NOISES \
  --musan_root egs/3dspeaker/sv-cam++/data/raw_data/musan \
  --metadata_csv my_methods_GAN/exp/LibriSpeech_results/LibriSpeech_100_kaldi_aug_metadata.csv \
  --workers 16 \
  --seed 42 \
  --overwrite
```

先 dry-run 检查数量和资源路径：

```bash
python my_methods_GAN/scripts/augment_librispeech_kaldi_style.py \
  --input_root my_methods_GAN/data/LibriSpeech/train-clean-100 \
  --output_root my_methods_GAN/data/LibriSpeech/train-clean-100-kaldi-aug \
  --rir_root egs/3dspeaker/sv-cam++/data/raw_data/RIRS_NOISES \
  --musan_root egs/3dspeaker/sv-cam++/data/raw_data/musan \
  --workers 16 \
  --seed 42 \
  --dry_run
```

如果想使用稍微保守的 noise SNR：

```bash
python my_methods_GAN/scripts/augment_librispeech_kaldi_style.py \
  --input_root my_methods_GAN/data/LibriSpeech/train-clean-100 \
  --output_root my_methods_GAN/data/LibriSpeech/train-clean-100-kaldi-aug-conservative \
  --rir_root egs/3dspeaker/sv-cam++/data/raw_data/RIRS_NOISES \
  --musan_root egs/3dspeaker/sv-cam++/data/raw_data/musan \
  --noise_snrs 5,10,15,20 \
  --music_snrs 5,10,15,20 \
  --workers 16 \
  --seed 42
```

注意这里的训练目标是增强后的语音 `x_aug`。后续 codec 转码应该对
`train-clean-100-kaldi-aug` 做：

```text
x_clean -> acoustic augmentation -> x_aug -> codec -> x_coded
```

训练 pair 为：

```text
input = x_coded
target = x_aug
```

### 7.2 基于增强后的 LibriSpeech 构建 manifest 并转码

可先从增强后的目标语音构建 clean-only manifest，再按 speaker 切分，最后为该子集补 coded 语音：

```bash
python my_methods_GAN/scripts/build_clean_manifest.py \
  --clean_root my_methods_GAN/data/LibriSpeech/train-clean-100-kaldi-aug \
  --output_csv my_methods_GAN/exp/LibriSpeech_results/LibriSpeech_100_kaldi_aug_manifest_all.csv

python my_methods_GAN/scripts/split_sv_manifest_by_speaker.py \
  --input_manifest my_methods_GAN/exp/LibriSpeech_results/LibriSpeech_100_kaldi_aug_manifest_all.csv \
  --train_manifest my_methods_GAN/exp/LibriSpeech_results/train_manifest_opus_100_kaldi_aug.csv \
  --valid_manifest my_methods_GAN/exp/LibriSpeech_results/valid_manifest_opus_100_kaldi_aug.csv \
  --valid_ratio 0.1 \
  --train_fraction 1 \
  --valid_fraction 1 \
  --stratified \
  --seed 42
```

同时转码 train/valid 子集。LibriSpeech 与 VoxCeleb1 说话人集合无关，因此这里不需要
`veri_test2.txt`，也不需要剔除 VoxCeleb1 测试说话人。

先生成 Opus 16k coded wav：

```bash
python my_methods_GAN/scripts/prepare_vox1_truepair_train_data.py \
  --raw_root egs/3dspeaker/sv-cam++/data/raw_data/vox2_wav/wav \
  --output_root my_methods_GAN/data/vox2/opus16_003 \
  --codec opus \
  --bitrate 16k \
  --sample_rate 16000 \
  --clean_write_mode hardlink \
  --workers 16 \
  --include_manifest my_methods_GAN/exp/voxceleb2_results/train_manifest_vox2_frac003_balanced.csv \
  --valid_include_manifest my_methods_GAN/exp/voxceleb2_results/valid_manifest_vox2_frac003_balanced.csv
```

再生成 Opus 8k coded wav：

```bash
python my_methods_GAN/scripts/prepare_vox1_truepair_train_data.py \
  --raw_root my_methods_GAN/data/LibriSpeech/train-clean-100-kaldi-aug \
  --output_root my_methods_GAN/data/LibriSpeech/opus8-100-kaldi-aug \
  --codec opus \
  --bitrate 8k \
  --sample_rate 16000 \
  --clean_write_mode hardlink \
  --workers 16 \
  --include_manifest my_methods_GAN/exp/LibriSpeech_results/train_manifest_opus_100_kaldi_aug.csv \
  --valid_include_manifest my_methods_GAN/exp/LibriSpeech_results/valid_manifest_opus_100_kaldi_aug.csv
```

`--clean_write_mode hardlink` 会让 `clean_train_wav` 直接硬链接到源 WAV，避免
clean 语音再次经过 ffmpeg 转成 clean。若源目录和输出目录不在同一文件系统，脚本会自动
退回到复制；需要旧的重采样/单声道标准化行为时，使用默认 `--clean_write_mode normalize`。

最后把 Opus 16k 和 Opus 8k 合并成训练/验证 pair manifest：

```bash
python my_methods_GAN/scripts/build_sv_codec_manifest.py \
  --clean_root my_methods_GAN/data/vox2/opus16_003/clean_train_wav \
  --coded_root \
    my_methods_GAN/data/vox2/opus16_003/coded_train_opus_16k \
  --include_manifest my_methods_GAN/exp/voxceleb2_results/train_manifest_vox2_frac003_balanced.csv \
  --output_csv my_methods_GAN/exp/voxceleb2_results/train_pair_manifest_vox2_frac003_balanced.csv

python my_methods_GAN/scripts/build_sv_codec_manifest.py \
  --clean_root my_methods_GAN/data/vox2/opus16_003/clean_train_wav \
  --coded_root \
    my_methods_GAN/data/vox2/opus16_003/coded_train_opus_16k \
  --include_manifest my_methods_GAN/exp/voxceleb2_results/valid_manifest_vox2_frac003_balanced.csv \
  --output_csv my_methods_GAN/exp/voxceleb2_results/valid_pair_manifest_vox2_frac003_balanced.csv
```

如果以后处理 VoxCeleb1，并且需要按官方 verification trials 剔除测试说话人，再额外加上：

```bash
  --exclude_test_speakers \
  --test_trials /root/rivermind-data/experiment_data_voxceleb/test_list/veri_test2.txt
```

`raw_root` 被删除时，脚本会自动从 `output_root/clean_train_wav` 进入 coded-only 续跑模式。已有且大小正常的 coded WAV 会跳过，0 字节残缺文件会自动重做。

## 8. 根据 trials 生成 eval clean/coded scp

脚本：my_methods_GAN/scripts/make_cnceleb_eval_scp.py

```bash
python my_methods_GAN/scripts/make_cnceleb_eval_scp.py \
  --trials_file egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac/eval/lists/trials.lst \
  --eval_clean_dir my_methods_GAN/data/cnceleb_truepair/eval_clean \
  --eval_coded_dir my_methods_GAN/data/cnceleb_truepair/eval_coded_amrwb_1585 \
  --clean_scp_out my_methods_GAN/exp/sv_codec_restore/eval_clean.scp \
  --coded_scp_out my_methods_GAN/exp/sv_codec_restore/eval_coded_amrwb1585.scp \
  --skip_clean_scp \
  --strict
```

## 9. 基于现有 clean_train_wav 生成 AMR-WB/G.711 全量数据（与 OPUS 对应 clean 一致）

脚本：my_methods_GAN/scripts/transcode_clean_wav_to_codec.py

先检查编码器：

```bash
ffmpeg -hide_banner -encoders | grep -E 'libvo_amrwbenc|pcm_mulaw|pcm_alaw'
```

示例：对现有 clean 目录全量生成 AMR-WB（15.85k，最接近 16k）

```bash
python my_methods_GAN/scripts/transcode_clean_wav_to_codec.py \
  --clean_root my_methods_GAN/data/cnceleb_truepair/clean_train_wav \
  --coded_root my_methods_GAN/data/cnceleb_truepair/coded_train_amrwb_1585 \
  --codec amrwb \
  --bitrate 15.85k \
  --sample_rate 16000 \
  --workers 16
```

建议先生成 OPUS 全量（确保多 codec 与同一 clean 集合对齐）：

```bash
python my_methods_GAN/scripts/transcode_clean_wav_to_codec.py \
  --clean_root my_methods_GAN/data/cnceleb_truepair/clean_train_wav \
  --coded_root my_methods_GAN/data/cnceleb_truepair/coded_train_opus_16k_full \
  --codec opus \
  --bitrate 16k \
  --sample_rate 16000 \
  --workers 16
```

示例：对同一 clean 全量生成 G.711 mu-law

```bash
python my_methods_GAN/scripts/transcode_clean_wav_to_codec.py \
  --clean_root my_methods_GAN/data/cnceleb_truepair/clean_train_wav \
  --coded_root my_methods_GAN/data/cnceleb_truepair/coded_train_g711mulaw \
  --codec g711_mulaw \
  --sample_rate 16000 \
  --workers 16
```

可选：G.711 A-law

```bash
python my_methods_GAN/scripts/transcode_clean_wav_to_codec.py \
  --clean_root my_methods_GAN/data/cnceleb_truepair/clean_train_wav \
  --coded_root my_methods_GAN/data/cnceleb_truepair/coded_train_g711alaw \
  --codec g711_alaw \
  --sample_rate 16000 \
  --workers 16
```

## 10. 构建 multi-codec manifest（OPUS + AMR-WB + G.711，全量）并切分

```bash
python my_methods_GAN/scripts/build_sv_codec_manifest.py \
  --clean_root my_methods_GAN/data/cnceleb_truepair/clean_train_wav \
  --coded_root my_methods_GAN/data/cnceleb_truepair/coded_train_opus_16k_full \
              my_methods_GAN/data/cnceleb_truepair/coded_train_amrwb_1585 \
              my_methods_GAN/data/cnceleb_truepair/coded_train_g711mulaw \
  --output_csv my_methods_GAN/exp/sv_codec_restore/pair_manifest_opus_amr_g711_full.csv
```

```bash
python my_methods_GAN/scripts/split_sv_manifest_by_speaker.py \
  --input_manifest my_methods_GAN/exp/sv_codec_restore/pair_manifest_opus_amr_g711_full.csv \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_opus_amr_g711_full.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_opus_amr_g711_full.csv \
  --valid_ratio 0.1 \
  --seed 42
```

## 11. 继续实验 A / B（仅替换 manifest）

实验 A（rec-only）：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_opus_amr_g711_full.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_opus_amr_g711_full.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_opus_amr_g711_full \
  --phase1_epochs 25 --phase2_epochs 0 --phase3_epochs 0 \
  --batch_size 8 --num_workers 8
```

实验 B（teacher，warm-start from A）：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_opus_amr_g711_full.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_opus_amr_g711_full.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_expB_campplus_opus_amr_g711_full \
  --init_generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_opus_amr_g711_full/best_generator.pt \
  --phase1_epochs 0 --phase2_epochs 8 --phase3_epochs 0 \
  --use_campplus_train_loss \
  --spk_loss_weight 3.0 \
  --batch_size 8 --num_workers 8
```

## 12. 当前执行策略：仅 AMR-WB 训练（保留 G.711 逻辑，暂不启用）

说明：
- 当前只使用 `OPUS + AMR-WB` 进行训练。
- `G.711` 相关转码与命令保留在上文，后续需要时可直接启用。

构建仅 OPUS + AMR-WB 的 manifest：

```bash
python my_methods_GAN/scripts/build_sv_codec_manifest.py \
  --clean_root my_methods_GAN/data/cnceleb_truepair/clean_train_wav \
  --coded_root my_methods_GAN/data/cnceleb_truepair/coded_train_opus_16k \
              my_methods_GAN/data/cnceleb_truepair/coded_train_amrwb_1585 \
  --output_csv my_methods_GAN/exp/sv_codec_restore/pair_manifest_opus_amrwb_full.csv
```

```bash
python my_methods_GAN/scripts/split_sv_manifest_by_speaker.py \
  --input_manifest my_methods_GAN/exp/sv_codec_restore/pair_manifest_opus_amrwb_full.csv \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_opus_amrwb_full.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_opus_amrwb_full.csv \
  --valid_ratio 0.1 \
  --seed 42
```

实验 A（rec-only，OPUS+AMR-WB）：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_opus_amrwb_full.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_opus_amrwb_full.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_opus_amrwb_full \
  --phase1_epochs 25 --phase2_epochs 0 --phase3_epochs 0 \
  --batch_size 8 --num_workers 8
```

实验 B（teacher，warm-start from A，OPUS+AMR-WB）：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_opus_amrwb_full.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_opus_amrwb_full.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_expB_campplus_opus_amrwb_full \
  --init_generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_opus_amrwb_full/best_generator.pt \
  --phase1_epochs 0 --phase2_epochs 8 --phase3_epochs 0 \
  --use_campplus_train_loss \
  --spk_loss_weight 3.0 \
  --batch_size 8 --num_workers 8
```

## 13. 数据读取模式更新：展开样本（multi-codec -> multi-row）

- 修改文件：`my_methods_GAN/sv_codec_restore_gan/data/dataset.py`
- 当前训练读取方式：
  - 若 manifest 某行 `codec_wav` 含多个路径（如 `opus|amrwb`），会在加载时展开为多条样本。
  - 即同一个 `clean_wav` 会对应多条单 codec 样本（`clean+opus`、`clean+amrwb` 分别参与训练）。
  - 不再是“同一行随机选一个 codec”模式。

## 14. Experiment A 客观语音质量评测（coded vs restored）

脚本：`my_methods_GAN/scripts/eval_objective_quality_experiment_a.py`

说明：
- 输入可选两种模式：`manifest_csv`（需包含 `clean_wav` / `codec_wav` 列）或 `clean_wav_scp + coded_wav_scp`，并加载 Experiment A 的 `best_generator.pt`。
- 同时输出 `coded->clean` 与 `restored->clean` 的客观指标均值。
- 输出 JSON 含 `overall` 与 `by_codec` 两级统计，便于直接写报告。
- 默认指标（无需额外安装）：`si_sdr_db,snr_db,l1,mse,mrstft,complex_l1`
- 可选指标：`stoi`（需 `pystoi`）、`pesq_wb`（需 `pesq`）
- 中期正式结论建议使用 `eval` 集（`eval_clean.scp` + `eval_coded_*.scp`），不建议用 `valid` 集做最终报告。
- 分块参数（用于防 OOM）：
  - `--infer_chunk_seconds`：生成器推理分块长度（秒）。例如 `4` 表示每次只前向 4 秒语音。
  - `--infer_hop_seconds`：生成器推理分块步长（秒）。`4/4` 表示无重叠；如 `4/2` 表示 50% 重叠融合。
  - `--metric_chunk_seconds`：`mrstft/complex_l1` 等 STFT 指标的分块长度（秒）。
  - `--metric_hop_seconds`：STFT 指标分块步长（秒）。
  - 建议：先用 `4/4`；若仍 OOM 改为 `2/2`；若追求更平滑可用 `4/2`（但更慢）。
- 本次新增加速参数：
  - `--infer_chunk_batch_size`：推理分块批量前向大小（默认 8，显存足够可适当增大）。
  - `--mp_metric_workers`：STOI/PESQ 多进程 worker 数（`>0` 开启并行，`0` 关闭）。
  - `--mp_metric_prefetch`：STOI/PESQ 异步队列深度（默认 32，建议 16~64）。
  - `--mp_metric_start_method`：多进程启动方式（推荐 `spawn`，CUDA 场景更稳）。
  - `--mp_metric_max_tasks_per_child`：每个 worker 处理 N 个任务后重建（默认 200，降低原生库长跑崩溃概率）。
  - `--gc_collect_every`：低频 `gc.collect()` 间隔（默认 `0` 关闭，建议仅在内存压力大时启用）。
- 新增“两阶段缓存评测”参数：
  - `--restored_cache_dir`：restored wav 缓存目录。
  - `--write_restored_cache`：把生成的 restored wav 写入缓存。
  - `--read_restored_cache`：评测时优先读取缓存 restored wav。
  - `--read_restored_cache_only`：仅从缓存读取 restored（缺失则跳过，不再推理）。
  - `--restore_only`：只做 coded->restored 缓存，不计算指标。

推荐：两阶段缓存流程（先缓存，再评测）

第一阶段：仅生成并缓存 restored wav（全量）

```bash
python my_methods_GAN/scripts/eval_objective_quality_experiment_a.py \
  --clean_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_clean.scp \
  --coded_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_coded_opus16k.scp \
  --codec_name opus \
  --generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_q25_third/checkpoints/epoch_008.pt \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/eval_results/expA_restore_cache_opus.json \
  --device cuda \
  --restore_only \
  --restored_cache_dir my_methods_GAN/exp/sv_codec_restore/run_main/restored_cache/opus \
  --write_restored_cache \
  --infer_chunk_seconds 4 \
  --infer_hop_seconds 4 \
  --infer_chunk_batch_size 32 \
  --verbose_every 200
```

第二阶段：仅读缓存 restored，计算客观指标

```bash
python my_methods_GAN/scripts/eval_objective_quality_experiment_a.py \
  --clean_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_clean.scp \
  --coded_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_coded_opus16k.scp \
  --codec_name opus \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/eval_results/expA_objective_quality_eval_only_opus_cached.json \
  --metrics si_sdr_db,snr_db,l1,mse,mrstft,complex_l1,stoi,pesq_wb \
  --disable_pesq_wb \
  --device cuda \
  --restored_cache_dir my_methods_GAN/exp/sv_codec_restore/run_main/restored_cache/opus \
  --read_restored_cache \
  --read_restored_cache_only \
  --metric_chunk_seconds 4 \
  --metric_hop_seconds 4 \
  --mp_metric_workers 2 \
  --mp_metric_prefetch 16 \
  --mp_metric_start_method spawn \
  --mp_metric_max_tasks_per_child 100 \
  --verbose_every 200
```

eval 全量评测（Opus）：

```bash
python my_methods_GAN/scripts/eval_objective_quality_experiment_a.py \
  --clean_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_clean.scp \
  --coded_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_coded_opus16k.scp \
  --codec_name opus \
  --generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_q25_third/checkpoints/epoch_008.pt \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/eval_results/expA_objective_quality_eval_only_opus.json \
  --metrics si_sdr_db,snr_db,l1,mse,mrstft,complex_l1,stoi,pesq_wb \
  --device cuda \
  --infer_chunk_seconds 16 \
  --infer_hop_seconds 16 \
  --infer_chunk_batch_size 8 \
  --metric_chunk_seconds 16 \
  --metric_hop_seconds 16 \
  --mp_metric_workers 2 \
  --mp_metric_prefetch 16 \
  --mp_metric_start_method spawn \
  --mp_metric_max_tasks_per_child 100 \
  --verbose_every 200 \
  --clear_cuda_every 200 \
  --gc_collect_every 0
```

eval 全量评测（AMR-WB）：

```bash
python my_methods_GAN/scripts/eval_objective_quality_experiment_a.py \
  --clean_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_clean.scp \
  --coded_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_coded_amrwb1585.scp \
  --codec_name amrwb \
  --generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_opus_amr_q25/best_generator.pt \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/eval_results/expA_objective_quality_eval_amrwb.json \
  --verbose_every 200 \
  --device cuda
```

快速冒烟测试（先验证链路）：

```bash
python my_methods_GAN/scripts/eval_objective_quality_experiment_a.py \
  --clean_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_clean.scp \
  --coded_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_coded_opus16k.scp \
  --codec_name opus \
  --generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_opus_amr_q25/best_generator.pt \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/eval_results/expA_objective_quality_eval_opus_smoke.json \
  --max_utts 5 \
  --device cpu
```

可选：若 `scp` 中路径为容器内绝对路径（如 `/workspace/camplusplus/...`）与当前环境不一致，可使用路径映射：

```bash
python my_methods_GAN/scripts/eval_objective_quality_experiment_a.py \
  --clean_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_clean.scp \
  --coded_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_coded_opus16k.scp \
  --codec_name opus \
  --generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_opus_amr_q25/best_generator.pt \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/eval_results/expA_objective_quality_eval_opus.json \
  --path_remap /workspace/camplusplus=/home/dgx/lkj/camplusplus \
  --device cuda
```

## 15. 绘制 clean 与 coded（AMR-WB）时频图对比

脚本：my_methods_GAN/scripts/plot_clean_coded_spectrogram.py

基础用法（默认就是大字号）：

```bash
python my_methods_GAN/scripts/plot_clean_coded_spectrogram.py \
  --clean_wav my_methods_GAN/data/cnceleb_truepair/clean_train_wav/id00000/singing-01-001.wav \
  --coded_wav my_methods_GAN/data/cnceleb_truepair/coded_train_opus_16k/id00000/singing-01-001.wav \
  --codec_name Opus \
  --output_png my_methods_GAN/exp/sv_codec_restore/run_main/eval_results/spec_clean_vs_Opuss.png
```

更大字号（汇报图推荐）：

```bash
python my_methods_GAN/scripts/plot_clean_coded_spectrogram.py \
  --clean_wav my_methods_GAN/data/cnceleb_truepair/clean_train_wav/id00000/singing-01-001.wav \
  --coded_wav my_methods_GAN/data/cnceleb_truepair/coded_train_amrwb_1585/id00000/singing-01-001.wav \
  --codec_name AMR-WB \
  --output_png my_methods_GAN/exp/sv_codec_restore/run_main/eval_results/spec_clean_vs_amrwb_bigfont.png \
  --font_size 26 \
  --title_size 32 \
  --subplot_title_size 28 \
  --label_size 26 \
  --tick_size 22 \
  --colorbar_size 22 \
  --fig_width 20 \
  --fig_height 9 \
  --dpi 260
```

## 16. coded 输入模型并导出 restored，再绘制 clean/coded/restored 时频图

脚本：my_methods_GAN/scripts/plot_clean_coded_restored_spectrogram.py

示例（AMR-WB，导出 restored wav + 三联时频图）：

```bash
python my_methods_GAN/scripts/plot_clean_coded_restored_spectrogram.py \
  --clean_wav my_methods_GAN/data/cnceleb_truepair/clean_train_wav/id00000/singing-01-001.wav \
  --coded_wav my_methods_GAN/data/cnceleb_truepair/coded_train_amrwb_1585/id00000/singing-01-001.wav \
  --generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_opus_amrwb/checkpoints/epoch_020.pt \
  --output_restored_wav my_methods_GAN/exp/sv_codec_restore/run_main/eval_results/restored_amrwb_singing-01-001.wav \
  --output_png my_methods_GAN/exp/sv_codec_restore/run_main/eval_results/spec_clean_coded_restored_amrwb.png \
  --coded_title "Coded (AMR-WB)" \
  --restored_title "Restored (Generator)" \
  --font_size 24 \
  --title_size 30 \
  --subplot_title_size 26 \
  --label_size 24 \
  --tick_size 19 \
  --colorbar_size 19 \
  --fig_width 27 \
  --fig_height 8 \
  --dpi 240
```

若右侧 dB 色条仍偏左，可继续右移：

```bash
--colorbar_left 0.935 --colorbar_width 0.016
```

## 17. VoxCeleb2 保留 M4A，按 speaker/video 均衡抽取一半

为了避免把 VoxCeleb2 全量转换成约 257G 的 WAV，clean/reference 直接保留原始 AAC
`.m4a`。训练 Dataset 会调用 FFmpeg，将 M4A 和 Opus 在线解码成 16 kHz 单声道
float Tensor。

源目录应为：

```text
/root/autodl-tmp/vox2/dev/aac/id*/video/*.m4a
```

先扫描全量 M4A，生成 clean-only manifest。这个过程只写 CSV，不复制或解码音频：

```bash
cd /root/camplusplus

python my_methods_GAN/scripts/build_clean_manifest.py \
  --clean_root /root/autodl-tmp/vox2/dev/aac \
  --extensions .m4a \
  --output_csv my_methods_GAN/exp/voxceleb2_results/vox2_clean_manifest_all_m4a.csv
```

按每位 speaker 抽取约 50%，并在该 speaker 的不同 video/session 之间轮流取样：

```bash
python my_methods_GAN/scripts/sample_vox2_manifest.py \
  --input_manifest my_methods_GAN/exp/voxceleb2_results/vox2_clean_manifest_all_m4a.csv \
  --output_manifest my_methods_GAN/exp/voxceleb2_results/vox2_clean_manifest_half_m4a.csv \
  --fraction 0.5 \
  --min_keep_per_speaker 1 \
  --seed 42
```

这里的“一半”只体现在 manifest 中，不会删除原始 M4A，也不会额外占用一份音频空间。
脚本会报告总行数、保留 speaker 数量、每位 speaker 的保留行数以及覆盖的 video 数量。

如果磁盘空间要求必须物理删除另外一半 M4A，先执行 dry-run。脚本会确认 half manifest
中的每个文件都实际存在，并统计预计删除的文件数与空间：

```bash
python my_methods_GAN/scripts/prune_audio_by_manifest.py \
  --audio_root /root/autodl-tmp/vox2/dev/aac \
  --keep_manifest my_methods_GAN/exp/voxceleb2_results/vox2_clean_manifest_half_m4a.csv \
  --extensions .m4a
```

确认 `Files to keep`、`Files to delete` 和空间统计正确后，才执行永久删除：

```bash
python my_methods_GAN/scripts/prune_audio_by_manifest.py \
  --audio_root /root/autodl-tmp/vox2/dev/aac \
  --keep_manifest my_methods_GAN/exp/voxceleb2_results/vox2_clean_manifest_half_m4a.csv \
  --extensions .m4a \
  --delete \
  --remove_empty_dirs
```

默认会在 half manifest 旁生成
`vox2_clean_manifest_half_m4a.csv.deleted_files.txt`，记录所有被删除路径。删除不可恢复，
因此必须保留好 half manifest；脚本中断后可以用相同命令继续，已删除文件不会被重复处理。

## 18. 从半量 M4A 再切训练子集，只为选中语音生成 Opus

### 18.1 从半量集取约原始全量的 1/25

半量 manifest 已经约等于原始全量的 `0.5`。如果最终实验子集希望约为原始全量的
`1/25 = 0.04`，则需要从半量 manifest 再取：

```text
0.04 / 0.5 = 0.08
```

下面命令从符合条件的 speaker 中选取约 8% 行，每位选中 speaker 最多保留 50 条，
然后执行 speaker-level train/valid 切分：

```bash
python my_methods_GAN/scripts/split_sv_manifest_by_speaker.py \
  --input_manifest my_methods_GAN/exp/voxceleb2_results/vox2_clean_manifest_half_m4a.csv \
  --train_manifest my_methods_GAN/exp/voxceleb2_results/vox2_frac004_train_clean_m4a.csv \
  --valid_manifest my_methods_GAN/exp/voxceleb2_results/vox2_frac004_valid_clean_m4a.csv \
  --valid_ratio 0.1 \
  --target_total_fraction 0.08 \
  --min_rows_per_speaker 50 \
  --max_rows_per_speaker 50 \
  --seed 42
```

如果要使用完整的“半量数据集”训练，不再缩小 speaker 集合，则去掉
`--target_total_fraction`、`--min_rows_per_speaker` 和 `--max_rows_per_speaker`：

```bash
python my_methods_GAN/scripts/split_sv_manifest_by_speaker.py \
  --input_manifest my_methods_GAN/exp/voxceleb2_results/vox2_clean_manifest_half_m4a.csv \
  --train_manifest my_methods_GAN/exp/voxceleb2_results/vox2_half_train_clean_m4a.csv \
  --valid_manifest my_methods_GAN/exp/voxceleb2_results/vox2_half_valid_clean_m4a.csv \
  --valid_ratio 0.1 \
  --seed 42
```

### 18.2 dry-run：确认只会转码最终选中的语音

Opus 使用 `.opus` 容器；不要将 Opus 文件命名为 `.m4a`。下面以约 1/25 子集为例：

```bash
python my_methods_GAN/scripts/convert_vox2_manifest_to_opus.py \
  --input_manifests \
    my_methods_GAN/exp/voxceleb2_results/vox2_frac004_train_clean_m4a.csv \
    my_methods_GAN/exp/voxceleb2_results/vox2_frac004_valid_clean_m4a.csv \
  --pair_manifests_out \
    my_methods_GAN/exp/voxceleb2_results/vox2_frac004_train_pair_opus16k.csv \
    my_methods_GAN/exp/voxceleb2_results/vox2_frac004_valid_pair_opus16k.csv \
  --output_root /root/autodl-tmp/vox2_opus_16k \
  --bitrate 16k \
  --sample_rate 16000 \
  --channels 1 \
  --workers 16 \
  --strict \
  --dry_run
```

### 18.3 正式生成 Opus 和训练 pair manifest

确认 dry-run 的 `Selected utterances` 和 `Unique Opus outputs` 正确后，去掉
`--dry_run`：

```bash
python my_methods_GAN/scripts/convert_vox2_manifest_to_opus.py \
  --input_manifests \
    my_methods_GAN/exp/voxceleb2_results/vox2_frac0025_train_clean_m4a.csv \
    my_methods_GAN/exp/voxceleb2_results/vox2_frac0025_valid_clean_m4a.csv \
  --pair_manifests_out \
    my_methods_GAN/exp/voxceleb2_results/vox2_frac0025_train_pair_opus16k.csv \
    my_methods_GAN/exp/voxceleb2_results/vox2_frac0025_valid_pair_opus16k.csv \
  --output_root /root/autodl-tmp/vox2_opus_16k \
  --bitrate 16k \
  --sample_rate 16000 \
  --channels 1 \
  --workers 16 \
  --strict
```

输出 pair manifest 格式为：

```text
utt_id,spk_id,clean_wav,codec_wav,codec_type
id00001-video-00001,id00001,/.../00001.m4a,/.../00001.opus,opus
```

训练时直接使用这两个 pair manifest：

```bash
--train_manifest my_methods_GAN/exp/voxceleb2_results/vox2_frac004_train_pair_opus16k.csv
--valid_manifest my_methods_GAN/exp/voxceleb2_results/vox2_frac004_valid_pair_opus16k.csv
--num_workers 8
--audio_preflight_items 2
```

Dataset 对 `.m4a/.opus` 使用 FFmpeg 在线解码并统一为 16 kHz、单声道。在线解码会增加
CPU 负担，建议先使用 `--num_workers 8`；如果 GPU 经常等待数据，再逐步提高 workers。
`train_sv_codec_restore_gan.py` 默认会在训练开始前从 train/valid 各抽取 2 对音频，
沿实际 Dataset 路径执行在线解码预检，并输出：

```text
[audio] train online-decode preflight passed: items=2 extensions=.m4a,.opus ...
[audio] valid online-decode preflight passed: items=2 extensions=.m4a,.opus ...
```

该预检通过后，正式训练批次也会由 DataLoader workers 在线解码。只有排查启动耗时时才建议
使用 `--audio_preflight_items 0` 关闭预检。

转码进程中断后直接重新执行原命令即可：默认跳过已有非空 Opus；只有需要重做已有文件时
才添加 `--overwrite`。转码脚本中的 `--strict` 默认在完成后均匀抽查 100 个输出，而不是
对数万条文件逐一执行 ffprobe；如需检查全部输出可额外设置 `--strict_samples 0`。

## 19. Step1：clean passthrough 对照训练

目的：在不修改生成器网络结构、不加入门控机制的情况下，先验证 `clean→clean` 数据是否能缓解
clean 输入经过模型后的说话人识别指标退化。

本阶段仍然使用原始生成器结构：

```text
x_out = x_in + R(x)
```

新增基础训练参数：

```text
--clean_passthrough_ratio 0.1
```

等价于：

```text
--train_clean_passthrough_ratio 0.1
--valid_clean_passthrough_ratio 0.0
```

含义：训练集中最终约 10% 样本为 `clean→clean`，90% 样本仍为 codec 修复样本。验证集默认不混入
clean passthrough，以免 codec restoration 验证指标被 clean 样本稀释。

注意：直接使用 `--clean_passthrough_ratio 0.2` 且 clean/code 同权训练，容易把模型推向
identity 映射，导致 restored 相对 coded 指标变差；同时 clean 样本上的 SI-SDR 可能产生很大的
裁剪前梯度。因此当前推荐使用“保护式 clean passthrough”：

```text
--clean_rec_loss_weight 0.1      clean 重建项降权，只做轻量 identity 约束
--clean_si_sdr_weight 0.0        clean 样本不使用 SI-SDR，避免梯度尖峰
--clean_mrstft_weight 0.5        clean 样本保留轻量频谱约束
--clean_complex_weight 0.0       clean 样本不使用 complex STFT L1
--codec_only_sv_loss             CAMP++ speaker/feature loss 只在 codec 样本上计算
```

训练开始时日志会打印样本组成，例如：

```text
[data] sample_types train=[clean=...,codec=...] valid=[codec=...] clean_passthrough_ratio train=0.1 valid=0.0
[config] clean_passthrough: train_ratio=0.1 ... clean_rec_weight=0.1 clean_si_sdr_weight=0.0 ... codec_only_sv_loss=True
```

新的训练日志会同时打印裁剪前和裁剪后梯度范数：

```text
generator_grad_norm_pre_clip=...
generator_grad_norm_post_clip=...
```

`pre_clip` 大表示 loss 梯度尖锐，`post_clip` 才是实际进入 optimizer step 前的梯度范数。
如果 `post_clip` 长期等于 `--grad_clip`，说明几乎每一步都被裁剪，阈值过低或 loss 权重过大。
在 VoxCeleb2 4s、CAM++ teacher 微调中，裁剪前全局梯度范数常见在 15～35 附近，因此建议
`--grad_clip 30.0`，让裁剪只处理尖峰，而不是每一步都把梯度压到 5。

### 19.1 VoxCeleb2 Opus16k Step1 训练命令

下面命令基于已有 Opus16k pair manifest，加入 10% clean passthrough 样本，并对 clean loss
降权，避免影响 codec 修复能力：

```bash
cd /root/camplusplus

python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/voxceleb2_results/vox2_frac004_train_pair_opus16k.csv \
  --valid_manifest my_methods_GAN/exp/voxceleb2_results/vox2_frac004_valid_pair_opus16k.csv \
  --output_dir my_methods_GAN/exp/voxceleb2_results/sv_codec_restore_vox2/run_step1_clean_passthrough_safe_opus16 \
  --clean_passthrough_ratio 0.1 \
  --clean_rec_loss_weight 0.1 \
  --clean_si_sdr_weight 0.0 \
  --clean_mrstft_weight 0.5 \
  --clean_complex_weight 0.0 \
  --codec_only_sv_loss \
  --emb_dim 48 \
  --num_blocks 5 \
  --hidden_units 100 \
  --attn_heads 4 \
  --phase1_epochs 0 \
  --phase2_epochs 40 \
  --phase3_epochs 0 \
  --batch_size 4 \
  --num_workers 16 \
  --audio_preflight_items 2 \
  --segment_seconds 4.0 \
  --phase2_segment_seconds 4.0 \
  --lr_g_max 5e-5 \
  --lr_g_min 1e-5 \
  --warmup_steps_g 0 \
  --grad_clip 30.0 \
  --si_sdr_weight 1.0 \
  --mrstft_weight 0.5 \
  --complex_weight 0.0 \
  --use_campplus_train_loss \
  --use_campplus_feat_loss \
  --campplus_feat_layers block2,out_nonlinear \
  --campplus_feat_loss_weight 3 \
  --campplus_frontend diff_mel \
  --spk_loss_weight 9 \
  --no_use_spk_amsoftmax \
  --phase3_no_gan \
  --phase3_no_wavlm \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_en_voxceleb_16k/campplus_voxceleb.bin \
  --device cuda
```

如果要从已有 checkpoint 继续做 Step1 对照训练，则增加：

```bash
  --resume \
  --resume_use_current_phase_config \
  --resume_ckpt my_methods_GAN/exp/voxceleb2_results/sv_codec_restore_vox2/run_expB_opus16_vox2_008/checkpoints/epoch_009.pt
```

注意：

```text
1. 当前 Step1 不会修改 generator.py，也不会启用 gate；
2. clean passthrough 样本在 Dataset 内部生成，不需要额外生成新的 manifest；
3. clean 样本的 coded 输入直接等于 clean 语音，不会再经过 Opus 或 FFmpeg 转码；
4. clean 样本只做轻量 identity 约束，不再与 codec 样本同权竞争；
5. 如果 4 秒、batch_size=4 仍然 OOM，则优先降到 --batch_size 3 或 2。
```

### 19.2 当前 VoxCeleb2 frac0025 安全版续训命令

针对 `run_expA_rec_opus16_clean_vox2_0025` / `run_expB_opus16_clean_vox2_0025` 中观察到的
裁剪前梯度过大、restored 相对 coded 变差问题，不建议继续使用同权 `--clean_passthrough_ratio 0.2`。
建议新开一个输出目录，用 10% clean passthrough + clean 降权：

```bash
cd /root/camplusplus

python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/voxceleb2_results/vox2_frac0025_train_pair_opus16k.csv \
  --valid_manifest my_methods_GAN/exp/voxceleb2_results/vox2_frac0025_valid_pair_opus16k.csv \
  --output_dir my_methods_GAN/exp/voxceleb2_results/run_expB_opus16_clean_vox2_0025_safe_cleanpt \
  --init_generator_ckpt my_methods_GAN/exp/voxceleb2_results/run_expB_opus16_vox2_003/best_generator_sv.pt \
  --clean_passthrough_ratio 0.2 \
  --clean_rec_loss_weight 1 \
  --clean_si_sdr_weight 0.0 \
  --clean_mrstft_weight 0.5 \
  --clean_complex_weight 0.0 \
  --codec_only_sv_loss \
  --phase1_epochs 0 \
  --phase2_epochs 20 \
  --phase3_epochs 0 \
  --batch_size 7 \
  --num_workers 16 \
  --audio_preflight_items 2 \
  --segment_seconds 4.0 \
  --phase2_segment_seconds 4.0 \
  --lr_g_max 5e-5 \
  --lr_g_min 1e-5 \
  --warmup_steps_g 1000 \
  --grad_clip 30.0 \
  --si_sdr_weight 1.0 \
  --mrstft_weight 0.5 \
  --complex_weight 0.0 \
  --use_campplus_train_loss \
  --use_campplus_feat_loss \
  --campplus_feat_layers block2,out_nonlinear \
  --campplus_feat_loss_weight 3 \
  --campplus_frontend diff_mel \
  --spk_loss_weight 9 \
  --no_use_spk_amsoftmax \
  --phase3_no_gan \
  --phase3_no_wavlm \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_en_voxceleb_16k/campplus_voxceleb.bin \
  --device cuda
```

如果仍然发现 restored 相对 coded 变差，下一轮优先继续减小 clean 影响，而不是加大 clean 比例：

```text
--clean_passthrough_ratio 0.05
--clean_rec_loss_weight 0.05
```

如果日志里仍然出现：

```text
generator_grad_norm_post_clip=3.000e+01
```

并且几乎每一步都是 30，说明 `--grad_clip 30.0` 仍然太低或 loss 过强。下一轮优先降低：

```text
--spk_loss_weight 6
--campplus_feat_loss_weight 2
```

## 20. 非均匀 CWS 子带分割实验

目标：只验证 CWS 频带划分是否能提升 coded 语音修复能力，不加入 gate、artifact encoder 或 mag confidence。

新增训练参数：

```text
--cws_mode uniform                 默认，保持原来的均匀 CWS
--cws_mode nonuniform              启用非均匀 CWS
--cws_band_edges 0,64,160,257      3 个子带：[0:64], [64:160], [160:257]
```

说明：

```text
n_fft=512 时，STFT 频点数 F = 257
bin resolution = 16000 / 512 = 31.25 Hz
0:64      约 0~2 kHz
64:160    约 2~5 kHz
160:257   约 5~8 kHz
```

默认不加参数时仍然是旧的均匀 CWS。启用非均匀 CWS 时，模型参数形状不变，旧 checkpoint 可以加载，但每个 CWS channel 对应的频带语义会变化，所以建议只做小学习率 finetune，不要直接拿旧 checkpoint 评测。

### 20.1 VoxCeleb1 非均匀 CWS phase1 小学习率 finetune

示例命令：

```bash
cd /root/camplusplus

python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_pair_manifest_train_q25.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_pair_manifest_valid_q25.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore_vox/run_expA_nonuniform_cws_opus_q25 \
  --init_generator_ckpt my_methods_GAN/exp/sv_codec_restore_vox/run_expA_rec_only_opus_q25/best_generator.pt \
  --cws_mode nonuniform \
  --cws_band_edges 0,64,160,257 \
  --emb_dim 48 \
  --num_blocks 5 \
  --hidden_units 100 \
  --attn_heads 4 \
  --phase1_epochs 10 \
  --phase2_epochs 0 \
  --phase3_epochs 0 \
  --batch_size 7 \
  --num_workers 16 \
  --log_interval 50 \
  --audio_preflight_items 2 \
  --segment_seconds 4.0 \
  --lr_g_max 3e-5 \
  --lr_g_min 5e-6 \
  --warmup_steps_g 200 \
  --si_sdr_weight 3 \
  --mrstft_weight 1 \
  --complex_weight 1 \
  --grad_clip 30.0 \
  --valid_sv_metric \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_en_voxceleb_16k/campplus_voxceleb.bin \
  --device cuda
```

### 20.2 VoxCeleb1 非均匀 CWS phase2 speaker-aware finetune

如果 phase1 的 valid_rec / sv_cos 没有明显变差，再从 phase1 best checkpoint 进入 phase2：

```bash
cd /root/camplusplus

python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_pair_manifest_train_q25.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_pair_manifest_valid_q25.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore_vox/run_expB_nonuniform_cws_opus_q25 \
  --init_generator_ckpt my_methods_GAN/exp/sv_codec_restore_vox/run_expA_nonuniform_cws_opus_q25/best_generator.pt \
  --cws_mode nonuniform \
  --cws_band_edges 0,64,160,257 \
  --emb_dim 48 \
  --num_blocks 5 \
  --hidden_units 100 \
  --attn_heads 4 \
  --phase1_epochs 0 \
  --phase2_epochs 5 \
  --phase3_epochs 0 \
  --batch_size 7 \
  --num_workers 16 \
  --log_interval 50 \
  --audio_preflight_items 2 \
  --segment_seconds 4.0 \
  --phase2_segment_seconds 4.0 \
  --lr_g_max 1e-5 \
  --lr_g_min 2e-6 \
  --warmup_steps_g 200 \
  --si_sdr_weight 1.0 \
  --mrstft_weight 0.5 \
  --complex_weight 0.0 \
  --use_campplus_train_loss \
  --use_campplus_feat_loss \
  --campplus_feat_layers block2,out_nonlinear \
  --campplus_feat_loss_weight 2 \
  --campplus_frontend diff_mel \
  --spk_loss_weight 6 \
  --no_use_spk_amsoftmax \
  --grad_clip 30.0 \
  --valid_sv_metric \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_en_voxceleb_16k/campplus_voxceleb.bin \
  --device cuda
```

### 20.3 VoxCeleb1 训练数据处理位置

当前文档里与 VoxCeleb1 训练数据处理最相关的位置：

```text
第 5 节：从原始 CN-Celeb 生成 true-pair 数据
```

注意：这一节标题仍写 CN-Celeb，但命令里的 `--raw_root` 实际指向 VoxCeleb1 train/wav：

```text
/root/rivermind-data/raw_data/vox1/train/wav
```

另外，第 7.2 附近有 `prepare_vox1_truepair_train_data.py` 的补充说明，并说明如果处理 VoxCeleb1 且需要剔除官方 verification trials 中的测试说话人，需要额外加：

```text
--exclude_test_speakers
--test_trials /root/rivermind-data/experiment_data_voxceleb/test_list/veri_test2.txt
```

## 21. VoxCeleb1 先按 speaker 切分 manifest，再按表单构建训练集

推荐流程：先只扫描 VoxCeleb1 train/wav 生成 clean-only 表单，再按 speaker-level 切分
train/valid，最后让 `prepare_vox1_truepair_train_data.py` 只为表单内语音生成 clean/coded true-pair。
这样可以避免直接对全量目录盲目转码，也方便后续复现实验子集。

### 21.1 扫描 VoxCeleb1 train/wav 生成 clean-only manifest

```bash
cd /root/camplusplus

python my_methods_GAN/scripts/build_clean_manifest.py \
  --clean_root /root/autodl-tmp/raw_data/vox1/train/wav \
  --extensions .wav \
  --output_csv my_methods_GAN/exp/sv_codec_restore_vox/vox1_train_clean_manifest_all.csv
```

输出 CSV 字段包括：

```text
utt_id,spk_id,video_id,rel_path,clean_wav
```

### 21.2 按 speaker-level 切分 train/valid manifest

VoxCeleb1 当前 train/wav 已经与 test/wav 说话人不重叠；这里做的是训练内部的
speaker-level train/valid 切分。

```bash
python my_methods_GAN/scripts/split_sv_manifest_by_speaker.py \
  --input_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_train_clean_manifest_all.csv \
  --train_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_train_clean_manifest_spksplit.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_valid_clean_manifest_spksplit.csv \
  --valid_ratio 0.1 \
  --train_fraction 0.2 \
  --valid_fraction 0.2 \
  --stratified \
  --seed 42
```

如果只想先取一个更小的 speaker 子集，例如最多 300 位 speaker、每人最多 80 条语音，可以用：

```bash
python my_methods_GAN/scripts/split_sv_manifest_by_speaker.py \
  --input_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_train_clean_manifest_all.csv \
  --train_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_train_clean_manifest_spk300_u80.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_valid_clean_manifest_spk300_u80.csv \
  --valid_ratio 0.1 \
  --max_speakers 300 \
  --min_rows_per_speaker 20 \
  --max_rows_per_speaker 80 \
  --seed 42
```

### 21.3 按 train/valid 表单生成 clean/coded true-pair 数据

这里使用 `--clean_write_mode hardlink`，避免 clean 语音再次经过 ffmpeg 转成 clean；coded
仍然会正常经过 Opus 编解码并输出 WAV。

```bash
python my_methods_GAN/scripts/prepare_vox1_truepair_train_data.py \
  --raw_root /root/autodl-tmp/raw_data/vox1/train/wav \
  --test_wav_root /root/autodl-tmp/raw_data/vox1/test/wav \
  --test_trials /root/autodl-tmp/raw_data/vox1/data/vox1_trials/veri_test2.txt \
  --output_root /root/autodl-tmp/SC_data/data/voxceleb1_spksplit \
  --codec opus \
  --bitrate 16k \
  --sample_rate 16000 \
  --clean_write_mode hardlink \
  --workers 16 \
  --include_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_train_clean_manifest_spksplit.csv \
  --valid_include_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_valid_clean_manifest_spksplit.csv \
  --train_pair_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_pair_manifest_train_spksplit_opus16k.csv \
  --valid_pair_manifest my_methods_GAN/exp/sv_codec_restore_vox/vox1_pair_manifest_valid_spksplit_opus16k.csv
```

如果要先检查统计而不写文件，加：

```text
--dry_run
```

如果中断后继续，通常不要加 `--overwrite`；脚本会跳过已存在且大小正常的 coded WAV。
如果确认要重做 clean 链接和 coded 文件，再加 `--overwrite`。
