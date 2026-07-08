# 容器内 Linux 命令总手册（唯一命令版本）

本文档统一维护可执行命令，默认在容器内执行。

默认约定：

- 宿主机项目目录：`~/lkj/camplusplus`
- 容器内项目目录：`/workspace/camplusplus`
- Shell：`bash`

## 1. 进入容器

```bash
cd ~/lkj/camplusplus
docker compose up -d
docker compose exec camplusplus bash
cd /workspace/camplusplus
```

## 2. 环境检查

```bash
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__); print(torch.cuda.is_available())"
ffmpeg -hide_banner -encoders | grep -E 'libopus|aac|libvo_amrwbenc|pcm_alaw|pcm_mulaw'
```

## 3. 准备 CN-Celeb mixed 数据

```bash
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

## 4. 构建 pair manifest

```bash
python my_methods/tools/build_pair_manifest.py \
  --clean_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/clean_train/wav.scp \
  --codec_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_mixed/train/wav.scp \
  --utt2spk egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/clean_train/utt2spk \
  --codec_assignment_csv egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_mixed/train/codec_assignment.csv \
  --output_csv my_methods/exp/cnceleb_fixedrate_pair_manifest.csv
```

## 5. 预提离线特征

```bash
python my_methods/tools/precompute_pair_features.py \
  --pair_manifest my_methods/exp/cnceleb_fixedrate_pair_manifest.csv \
  --output_root my_methods/exp/ca_afc_features_cnceleb_fixedrate \
  --output_manifest my_methods/exp/cnceleb_fixedrate_feature_manifest.csv \
  --sample_rate 16000 \
  --num_workers 4
```

## 6. CA-AFC smoke 训练

```bash
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
  --log_interval 20 \
  --lambda_rec 1.0 \
  --lambda_emb 0.3 \
  --lambda_smooth 0.01 \
  --device cuda
```

## 7. CA-AFC 正式训练

```bash
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

## 8. 断点续训

```bash
python my_methods/scripts/train_ca_afc.py \
  --train_feature_manifest my_methods/exp/cnceleb_fixedrate_feature_manifest.csv \
  --output_dir my_methods/exp/ca_afc_cnceleb_fixedrate \
  --campplus_model_bin pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin \
  --resume \
  --device cuda
```

## 9. CA-AFC + CAM++ 评测

```bash
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

## 10. 约定

1. 所有命令默认在容器内 `/workspace/camplusplus` 执行。
2. 本文档是唯一命令来源，其他文档只放链接不再复制命令。
3. 若命令更新，优先修改本文档并同步相关索引页。
