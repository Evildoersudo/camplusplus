### 1.1启动并进入容器

```bash
cd ~/lkj/camplusplus
docker compose up -d
docker compose exec camplusplus bash
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
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_q25.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_q25.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_q25_second \
  --phase1_epochs 25 \
  --phase2_epochs 0 \
  --phase3_epochs 0 \
  --batch_size 28 \
  --num_workers 8 \
  --segment_seconds 3.0 \
  --train_sample_fraction 1.0 \
  --valid_sample_fraction 1.0 \
  --train_stratified_sample \
  --valid_stratified_sample \
  --emb_dim 48 \
  --num_blocks 5 \
  --hidden_units 100 \
  --attn_heads 4 \
  --lr_g_max 3e-4 \
  --lr_g_min 1e-5 \
  --warmup_steps_g 200 \
  --weight_decay 1e-4 \
  --grad_clip 5.0 \
  --si_sdr_weight 3 \
  --mrstft_weight 1 \
  --complex_weight 1 \
  --no_valid_sv_metric \
  --wavlm_root my_methods_GAN/pretrained/WavLM \
  --wavlm_ckpt my_methods_GAN/pretrained/WavLM/WavLM-Base+.pt \
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
  --coded_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_coded_opus16k.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac/eval/lists/trials.lst \
  --generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_q25_second/best_generator.pt \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/eval_results_second.json \
  --restore_chunk_seconds 8 \
  --restore_overlap_seconds 0.5 \
  --restore_chunk_batch_size 16 \
  --restore_auto_shrink \
  --restore_min_chunk_seconds 1.0 \
  --restore_chunk_shrink_factor 0.7 \
  --use_cache \
  --cache_incremental \
  --cache_save_every 500 \
  --cache_dir my_methods_GAN/exp/sv_codec_restore/run_main/emb_cache \
  --dump_speaker_npy_dir my_methods_GAN/exp/sv_codec_restore/run_main/speaker_emb_npy \
  --speaker_id_sep / \
  --speaker_id_field 0 \
  --device cuda
```

参数说明：

- `--restore_chunk_seconds`：restored 路径分块长度（秒），默认 8。
- `--restore_overlap_seconds`：相邻块重叠长度（秒），默认 0.5。
- `--restore_chunk_batch_size`：restored 分块批量前向 batch size，默认 8；显存允许可尝试 12/16 提速。
- `--restore_auto_shrink`：遇到 CUDA OOM 自动缩小 chunk 并重试（默认开启）。
- `--restore_min_chunk_seconds`：自动缩块最小下限（秒），默认 1.0。
- `--restore_chunk_shrink_factor`：每次 OOM 后的缩放比例，默认 0.7。
- `--use_cache`：启用本地 embedding 缓存（默认开启）。
- `--cache_incremental`：按间隔增量写盘并支持中断续跑（默认开启）。
- `--cache_save_every`：每 N 条 utt 写一次增量缓存分片，默认 500。
- `--cache_dir`：缓存目录，默认 `<output_json目录>/emb_cache`。
- `--overwrite_cache`：强制重算并覆盖已有缓存。
- `--dump_speaker_npy_dir`：按 clean/coded/restored 分目录导出每个说话人的 embedding `.npy`。
- `--speaker_id_sep`：从 utt 提取说话人 ID 的分隔符，默认 `/`。
- `--speaker_id_field`：分隔后取第几个字段作为说话人 ID，默认 `0`。

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
  --log_file my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_q25_second/train.log \
  --sample_every 40 \
  --output_png my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_q25_second/avg_train_curve_s40.png
```

每 60 个 step 取一个点：

```bash
python my_methods_GAN/scripts/plot_train_avg_curve.py \
  --log_file my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_q25_fast/train.log \
  --sample_every 60 \
  --output_png my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_q25_fast/avg_train_curve_s60.png
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
  --raw_root egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac \
  --output_root my_methods_GAN/data/cnceleb_truepair \
  --codec opus \
  --bitrate 16k \
  --sample_rate 16000 \
  --workers 8 \
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
  --input_manifest my_methods_GAN/exp/sv_codec_restore/pair_manifest_all.csv \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_q25.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_q25.csv \
  --valid_ratio 0.1 \
  --train_fraction 0.25 \
  --valid_fraction 0.25 \
  --stratified \
  --seed 42
```

## 8. 根据 trials 生成 eval clean/coded scp

脚本：my_methods_GAN/scripts/make_cnceleb_eval_scp.py

```bash
python my_methods_GAN/scripts/make_cnceleb_eval_scp.py \
  --trials_file egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac/eval/lists/trials.lst \
  --eval_clean_dir my_methods_GAN/data/cnceleb_truepair/eval_clean \
  --eval_coded_dir my_methods_GAN/data/cnceleb_truepair/eval_coded_opus_16k \
  --clean_scp_out my_methods_GAN/exp/sv_codec_restore/eval_clean.scp \
  --coded_scp_out my_methods_GAN/exp/sv_codec_restore/eval_coded_opus16k.scp \
  --strict
```
