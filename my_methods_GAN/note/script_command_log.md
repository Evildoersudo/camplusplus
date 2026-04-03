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
  --batch_size 4 \
  --num_workers 2 \
  --segment_seconds 2.0 \
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
  --wavlm_root /home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/WavLM \
  --wavlm_ckpt /home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/WavLM/WavLM-Base.pt \
  --campplus_ckpt /home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin \
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
  --wavlm_ckpt /home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/WavLM/WavLM-Base.pt \
  --campplus_ckpt /home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin \
  --device cuda
```

## 3. 评测 clean/coded/restored 的 EER/minDCF

脚本：my_methods_GAN/scripts/eval_sv_codec_restore_gan.py

```bash
python my_methods_GAN/scripts/eval_sv_codec_restore_gan.py \
  --clean_wav_scp /path/to/test_clean_wav.scp \
  --coded_wav_scp /path/to/test_coded_wav.scp \
  --trials_file /path/to/test_trials.txt \
  --generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_main/best_generator.pt \
  --campplus_ckpt /home/dgx/lkj/camplusplus/my_methods_GAN/pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/eval_results.json \
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
