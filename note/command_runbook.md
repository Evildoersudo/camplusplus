# 命令执行记录

## 说明

从当前阶段开始，所有需要你手动执行的脚本命令都统一记录在这份文档里，方便后续调试、复现和迁移到服务器环境。

记录原则：

- 只保留可以直接执行的命令
- 按实验主题分组
- 尽量使用绝对路径或清晰的相对路径

## 环境与前置检查

### 使用现有环境（不新建虚拟环境）

```bash
conda activate Graduation_Project
```

### 必需工具检查（仅首次）

```bash
python -c "import torch, torchaudio, matplotlib; print('python deps ok')"
ffmpeg -version
```

### 数据路径快速自检

```bash
for path in \
  egs/3dspeaker/sv-cam++/data/vctk_16k/wav16k/p225/p225_001.wav \
  egs/3dspeaker/sv-cam++/data/vctk_opus4k/wav16k/p225/p225_001.wav
do
  printf "%s => %s\n" "$path" "$(test -e "$path" && echo true || echo false)"
done
```

## FBank 对比实验

### 目的

观察不同编解码器在低码率条件下对高频信息的截断效应。

### 依赖安装

如果当前环境还没有 `matplotlib`，先执行：

```bash
pip install matplotlib
```

说明：

- `compare_fbank_codec.py` 已支持多后端解码回退（ffmpeg/sox_io/soundfile）
- 如果仍报错，优先检查输入文件是否存在、是否被占用、是否损坏

### 生成 clean 与 degraded 的 FBank 对比图

下面这条命令以同一条语音的 clean 版本和 Opus 4k 版本为例，生成一张对比图：

```bash
python egs/3dspeaker/sv-cam++/local/compare_fbank_codec.py \
  --clean_wav egs/3dspeaker/sv-cam++/data/vctk_16k/wav16k/p225/p225_001.wav \
  --degraded_wav egs/3dspeaker/sv-cam++/data/vctk_opus4k/wav16k/p225/p225_001.wav \
  --out_png note/compare_codec/fbank_compare_opus4k_p225_001.png \
  --sample_rate 16000 \
  --n_mels 80 \
  --max_frames 400
```

### 当前仓库状态说明（2026-03-18）

当前已存在并可直接使用的数据目录：

- `egs/3dspeaker/sv-cam++/data/vctk_16k`
- `egs/3dspeaker/sv-cam++/data/vctk_opus4k`

当前未生成（路径默认不存在）的目录：

- `egs/3dspeaker/sv-cam++/data/vctk_amrwb_885`
- `egs/3dspeaker/sv-cam++/data/vctk_g711_alaw`

如果直接使用未生成目录执行对比，会报文件读取失败。

### 先生成单条 AMR-WB 8.85k 样例（用于可视化）

```bash
mkdir -p egs/3dspeaker/sv-cam++/data/vctk_amrwb_885/wav16k/p225
ffmpeg -y -i egs/3dspeaker/sv-cam++/data/vctk_16k/wav16k/p225/p225_001.wav -ar 16000 -ac 1 -c:a libvo_amrwbenc -b:a 8.85k egs/3dspeaker/sv-cam++/data/vctk_amrwb_885/wav16k/p225/p225_001.amr
ffmpeg -y -i egs/3dspeaker/sv-cam++/data/vctk_amrwb_885/wav16k/p225/p225_001.amr -ar 16000 -ac 1 egs/3dspeaker/sv-cam++/data/vctk_amrwb_885/wav16k/p225/p225_001.wav
```

### 生成 clean 与 AMR-WB 8.85k 的 FBank 对比图

```bash
python egs/3dspeaker/sv-cam++/local/compare_fbank_codec.py \
  --clean_wav egs/3dspeaker/sv-cam++/data/vctk_16k/wav16k/p225/p225_001.wav \
  --degraded_wav egs/3dspeaker/sv-cam++/data/vctk_amrwb_885/wav16k/p225/p225_001.wav \
  --out_png note/compare_codec/fbank_compare_amrwb885_p225_001.png \
  --sample_rate 16000 \
  --n_mels 80 \
  --max_frames 400
```

### 先生成单条 G.711 A-law 样例（用于可视化）

```bash
mkdir -p egs/3dspeaker/sv-cam++/data/vctk_g711_alaw/wav16k/p225
ffmpeg -y -i egs/3dspeaker/sv-cam++/data/vctk_16k/wav16k/p225/p225_001.wav -ar 8000 -ac 1 -c:a pcm_alaw egs/3dspeaker/sv-cam++/data/vctk_g711_alaw/wav16k/p225/p225_001_alaw_8k.wav
ffmpeg -y -i egs/3dspeaker/sv-cam++/data/vctk_g711_alaw/wav16k/p225/p225_001_alaw_8k.wav -ar 16000 -ac 1 egs/3dspeaker/sv-cam++/data/vctk_g711_alaw/wav16k/p225/p225_001.wav
```

### 生成 clean 与 G.711 A-law 的 FBank 对比图

```bash
python egs/3dspeaker/sv-cam++/local/compare_fbank_codec.py \
  --clean_wav egs/3dspeaker/sv-cam++/data/vctk_16k/wav16k/p225/p225_001.wav \
  --degraded_wav egs/3dspeaker/sv-cam++/data/vctk_g711_alaw/wav16k/p225/p225_001.wav \
  --out_png note/compare_codec/fbank_compare_g711_alaw_p225_001.png \
  --sample_rate 16000 \
  --n_mels 80 \
  --max_frames 400
```

## Python 脚本命令文档维护规则

从现在开始，每次新增 Python 脚本或修改已有 Python 脚本时，都要在本文件同步更新以下内容：

- 脚本路径
- 脚本用途
- 一条快速验证命令（smoke test）
- 一条完整实验命令
- 关键注意事项或已知失败条件

## CN-Celeb 编解码扫描（CAM++）

### 脚本

- `egs/3dspeaker/sv-cam++/local/run_cnceleb_codec_sweep.py`

### 用途

- 构建 Opus / G.711 / AMR-WB / AAC 的退化测试条件
- 在固定 CAM++ 后端下评估各 codec/码率条件
- 输出 EER、minDCF、相对 clean 退化倍数与分组排名表

### 快速验证命令（smoke test）

```bash
python egs/3dspeaker/sv-cam++/local/run_cnceleb_codec_sweep.py \
  --test_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/test/wav.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/trials/trials.lst \
  --model_bin pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
  --embedding_cache_root egs/3dspeaker/sv-cam++/exp/codec_bitrate_test/embedding_cache \
  --stratified_sampling \
  --trial_sample_mode random \
  --trial_sample_seed 42 \
  --limit 2000 \
  --target_limit 100 \
  --nontarget_limit 1900 \
  --num_workers 4 \
  --report_csv egs/3dspeaker/sv-cam++/exp/codec_bitrate_test/smoke_report.csv \
  --report_json egs/3dspeaker/sv-cam++/exp/codec_bitrate_test/smoke_report.json \
  --ranking_csv egs/3dspeaker/sv-cam++/exp/codec_bitrate_test/smoke_ranking.csv \
  --ranking_md egs/3dspeaker/sv-cam++/exp/codec_bitrate_test/smoke_ranking.md
```

### 完整实验命令

```bash
python egs/3dspeaker/sv-cam++/local/run_cnceleb_codec_sweep.py \
  --test_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/test/wav.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/trials/trials.lst \
  --model_bin pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
  --embedding_cache_root egs/3dspeaker/sv-cam++/exp/codec_bitrate_test/embedding_cache \
  --stratified_sampling \
  --trial_sample_mode random \
  --trial_sample_seed 42 \
  --limit 2000 \
  --target_limit 100 \
  --nontarget_limit 1900 \
  --num_workers 8 \
  --report_csv egs/3dspeaker/sv-cam++/exp/codec_bitrate_test/cnceleb_codec_report.csv \
  --report_json egs/3dspeaker/sv-cam++/exp/codec_bitrate_test/cnceleb_codec_report.json \
  --ranking_csv egs/3dspeaker/sv-cam++/exp/codec_bitrate_test/cnceleb_codec_grouped_ranking.csv \
  --ranking_md egs/3dspeaker/sv-cam++/exp/codec_bitrate_test/cnceleb_codec_grouped_ranking.md
```

### 注意事项

- 脚本已改为“在线编解码 + embedding 缓存”流程，不再落地保存整套 degraded wav。
- `--embedding_cache_root` 目录下会按条件缓存 `.npy` embedding，后续重复运行会自动复用。
- `--overwrite` / `--overwrite_embeddings` 为可选项。启用后会强制重算并覆盖缓存 embedding。
- 脚本内部已对 CN-Celeb 的 trials 键做归一化（如 `idxxxx-enroll`、`test/xxx.wav`）。
- 当子集过小（例如 `--limit` 太小）且标签只包含单一类别时，无法计算 EER/minDCF。
- 新增 `--trial_sample_mode`：
  - `head`：取 trials 文件前缀（默认）
  - `random`：随机均匀抽样
- 新增 `--trial_sample_seed`：仅在 `random` 模式下生效；seed 相同可复现实验。
- 若希望“每次测试固定样本数”，优先设置 `--limit N`；配合 `--trial_sample_mode random --trial_sample_seed 固定值` 可保证每次抽到同一批 trials。

## 频带贡献实验（CAM++）

### 脚本

- `egs/3dspeaker/sv-cam++/local/run_cnceleb_band_contribution.py`

### 用途

- 在 FBank 上做频带保留/置零（full、low、mid、high、low_mid）
- 评估 clean 与低码率 codec 条件下各频带的 EER/minDCF 贡献
- 直接支撑频带重加权设计的实验依据

### 快速验证命令（smoke test）

```bash
python egs/3dspeaker/sv-cam++/local/run_cnceleb_band_contribution.py \
  --test_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/test/wav.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/trials/trials.lst \
  --model_bin pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
  --embedding_cache_root egs/3dspeaker/sv-cam++/exp/band_contribution/embedding_cache_smoke \
  --plot_dir egs/3dspeaker/sv-cam++/exp/band_contribution/plots_smoke \
  --codec_conditions clean,opus@4k,amrwb@8.85k \
  --bands full,low,mid,high,low_mid \
  --low_range 1-26 \
  --mid_range 27-53 \
  --high_range 54-80 \
  --stratified_sampling \
  --trial_sample_mode random \
  --trial_sample_seed 42 \
  --limit 2000 \
  --target_limit 100 \
  --nontarget_limit 1900 \
  --report_csv egs/3dspeaker/sv-cam++/exp/band_contribution/smoke_report.csv \
  --report_json egs/3dspeaker/sv-cam++/exp/band_contribution/smoke_report.json \
  --table_md egs/3dspeaker/sv-cam++/exp/band_contribution/smoke_table.md
```

### 完整实验命令

```bash
python egs/3dspeaker/sv-cam++/local/run_cnceleb_band_contribution.py \
  --test_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/test/wav.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/trials/trials.lst \
  --model_bin pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
  --embedding_cache_root egs/3dspeaker/sv-cam++/exp/band_contribution/embedding_cache \
  --plot_dir egs/3dspeaker/sv-cam++/exp/band_contribution/plots \
  --codec_conditions clean,opus@4k,opus@8k,amrwb@8.85k,amrwb@12.65k \
  --bands full,low,mid,high,low_mid \
  --low_range 1-26 \
  --mid_range 27-53 \
  --high_range 54-80 \
  --stratified_sampling \
  --trial_sample_mode random \
  --trial_sample_seed 42 \
  --limit 2000 \
  --target_limit 100 \
  --nontarget_limit 1900 \
  --report_csv egs/3dspeaker/sv-cam++/exp/band_contribution/band_contribution_report.csv \
  --report_json egs/3dspeaker/sv-cam++/exp/band_contribution/band_contribution_report.json \
  --table_md egs/3dspeaker/sv-cam++/exp/band_contribution/band_contribution_table.md
```

### 注意事项

- 频带定义是 1-based 且闭区间，例如 `1-26`、`27-53`、`54-80`。
- 该脚本不会生成用于评测的数据集 `wav` 文件；流程是在线编解码后直接前向，最终仅缓存 embedding `.npy`。
- 频带实验使用在线编解码 + embedding 缓存，`--embedding_cache_root` 不变即可断点续跑。
- 若要强制重算某次实验缓存，请加 `--overwrite_embeddings`。
- 脚本默认自动绘图，输出 `band_contribution_eer.png` 与 `band_contribution_min_dcf.png` 到 `--plot_dir`。
- 如只想导出表格不绘图，可加 `--disable_plot`。
- 新增 `--trial_sample_mode`：`head`（默认前缀）或 `random`（随机均匀抽样）。
- 新增 `--trial_sample_seed`：仅在 `random` 模式下生效；seed 固定可复现实验。
- 若要“每次测试固定样本数”，优先设置 `--limit N`；配合 `--trial_sample_mode random --trial_sample_seed 固定值` 可保证每次抽样一致。

## FBank 时间窗长度对比实验（独立脚本）

### 目的

- 固定 CAM++ 预训练模型与其余评测参数，仅改变 FBank 窗长。
- 验证在低码率条件下，短窗（更高时间分辨率）是否更稳。
- 使用独立脚本，避免影响现有 `run_cnceleb_codec_sweep.py` 与 `run_cnceleb_band_contribution.py`。

### 推荐设置

- 使用脚本：`egs/3dspeaker/sv-cam++/local/run_cnceleb_fbank_window_eval.py`
- 固定帧移：`--fbank_shift_ms 10`
- 依次测试窗长：`--window_lengths_ms 25,20,15,10`
- 为保证可复现，固定随机抽样：`--trial_sample_mode random --trial_sample_seed 42 --limit 1000`

### 推荐命令（一次跑完多窗长）

```bash
python egs/3dspeaker/sv-cam++/local/run_cnceleb_fbank_window_eval.py \
  --test_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/test/wav.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/trials/trials.lst \
  --model_bin pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
  --embedding_cache_root egs/3dspeaker/sv-cam++/exp/fbank_window_eval/embedding_cache \
  --codec_conditions clean,opus@4k,opus@8k,amrwb@8.85k \
  --window_lengths_ms 25,20,15,10 \
  --fbank_shift_ms 10 \
  --stratified_sampling \
  --trial_sample_mode random \
  --trial_sample_seed 42 \
  --limit 2000 \
  --target_limit 100 \
  --nontarget_limit 1900 \
  --report_csv egs/3dspeaker/sv-cam++/exp/fbank_window_eval/window_eval_report.csv \
  --report_json egs/3dspeaker/sv-cam++/exp/fbank_window_eval/window_eval_report.json \
  --table_md egs/3dspeaker/sv-cam++/exp/fbank_window_eval/window_eval_table.md
```

## CN-Celeb 固定码率 codec 评测（CAM++）
### 脚本

- `egs/3dspeaker/sv-cam++/local/run_cnceleb_codec_fixedrate_eval.py`

### 用途
- 固定目标码率附近的 codec 条件，对 CN-Celeb 测试集做 clean-vs-codec 对比
- 输出 EER、minDCF、相对 clean 的退化量到 CSV、JSON、Markdown
- 自动生成更适合论文展示的柱状图

### 快速验证命令（smoke test）
```bash
python egs/3dspeaker/sv-cam++/local/run_cnceleb_codec_fixedrate_eval.py \
  --test_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/test/wav.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/trials/trials.lst \
  --model_bin pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
  --codec_conditions clean,opus@16k,aac@16k,amrwb@15.85k \
  --embedding_cache_root egs/3dspeaker/sv-cam++/exp/codec_fixedrate_eval/embedding_cache_smoke \
  --stratified_sampling \
  --trial_sample_mode random \
  --trial_sample_seed 42 \
  --limit 2000 \
  --target_limit 100 \
  --nontarget_limit 1900 \
  --report_csv egs/3dspeaker/sv-cam++/exp/codec_fixedrate_eval/smoke_report.csv \
  --report_json egs/3dspeaker/sv-cam++/exp/codec_fixedrate_eval/smoke_report.json \
  --table_md egs/3dspeaker/sv-cam++/exp/codec_fixedrate_eval/smoke_table.md \
  --plot_dir egs/3dspeaker/sv-cam++/exp/codec_fixedrate_eval/plots_smoke
```

### 完整实验命令
```bash
python egs/3dspeaker/sv-cam++/local/run_cnceleb_codec_fixedrate_eval.py \
  --test_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/test/wav.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/trials/trials.lst \
  --model_bin pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
  --codec_conditions clean,opus@16k,aac@16k,amrwb@15.85k \
  --embedding_cache_root egs/3dspeaker/sv-cam++/exp/codec_fixedrate_eval/embedding_cache \
  --stratified_sampling \
  --trial_sample_mode random \
  --trial_sample_seed 42 \
  --limit 2000 \
  --target_limit 100 \
  --nontarget_limit 1900 \
  --report_csv egs/3dspeaker/sv-cam++/exp/codec_fixedrate_eval/codec_fixedrate_report.csv \
  --report_json egs/3dspeaker/sv-cam++/exp/codec_fixedrate_eval/codec_fixedrate_report.json \
  --table_md egs/3dspeaker/sv-cam++/exp/codec_fixedrate_eval/codec_fixedrate_table.md \
  --plot_dir egs/3dspeaker/sv-cam++/exp/codec_fixedrate_eval/plots
```

### 注意事项
- 默认条件是 `clean,opus@16k,aac@16k,amrwb@15.85k`。AMR-WB 没有严格 16k 档位，所以默认用最接近的 `15.85k`
- 建议默认开启 `--stratified_sampling`，并显式指定 `--target_limit 100 --nontarget_limit 1900`
- 使用显式按类上限时，要求 `target_limit + nontarget_limit <= limit`
- `--max_utts` 仍可用，但更建议优先控制 trial 数和 target/nontarget 数，而不是先控制 utt 数
- 图像默认输出 3 张：`fixedrate_eer.png`、`fixedrate_min_dcf.png`、`fixedrate_relative_eer.png`
- 脚本采用“在线编解码 + embedding 缓存”，不会落地保存整套 degraded wav
- 如果需要强制重算 embedding 缓存，增加 `--overwrite_embeddings`
## CN-Celeb 固定码率 codec 微调与评测（CAM++）
### 脚本

- `egs/3dspeaker/sv-cam++/local/run_cnceleb_codec_fixedrate_finetune.py`
- `egs/3dspeaker/sv-cam++/conf/cam++_cnceleb_codec16k_ft.yaml`

### 用途
- 构建 `clean + opus@16k + aac@16k + amrwb@15.85k` 的混合训练集
- 从 `pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin` 初始化 CAM++ 做微调
- 训练前先跑一份预训练模型基线评测，训练结束后再跑微调后评测
- 自动生成“预训练 vs 微调后”的对比 CSV、Markdown 和柱状图

### 推荐命令
```bash
python egs/3dspeaker/sv-cam++/local/run_cnceleb_codec_fixedrate_finetune.py \
  --train_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/clean_train/wav.scp \
  --train_utt2spk egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/clean_train/utt2spk \
  --test_wav_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/test/wav.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb/trials/trials.lst \
  --noise_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/musan/wav.scp \
  --reverb_scp egs/3dspeaker/sv-cam++/data/CN_celeb_database/rirs/wav.scp \
  --train_config egs/3dspeaker/sv-cam++/conf/cam++_cnceleb_codec16k_ft.yaml \
  --init_model pretrained/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin \
  --mixed_train_dir egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_fixedrate_mixed/train \
  --mixed_audio_root egs/3dspeaker/sv-cam++/data/CN_celeb_database/cnceleb_fixedrate_mixed_audio \
  --exp_dir egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft \
  --baseline_embedding_cache_root egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/baseline_embedding_cache \
  --baseline_report_csv egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/pretrained_report.csv \
  --baseline_report_json egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/pretrained_report.json \
  --baseline_table_md egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/pretrained_report.md \
  --baseline_plot_dir egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/pretrained_plots \
  --eval_embedding_cache_root egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/embedding_cache \
  --eval_report_csv egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/report.csv \
  --eval_report_json egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/report.json \
  --eval_table_md egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/report.md \
  --eval_plot_dir egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/plots \
  --compare_csv egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/compare_pretrained_vs_finetuned.csv \
  --compare_md egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/compare_pretrained_vs_finetuned.md \
  --compare_plot_dir egs/3dspeaker/sv-cam++/exp/campp_cnceleb_codec16k_ft_eval/compare_plots \
  --clean_ratio 0.25 \
  --opus_ratio 0.25 \
  --aac_ratio 0.25 \
  --amrwb_ratio 0.25 \
  --g711_ratio 0.0 \
  --opus_bitrates 16k \
  --aac_bitrates 16k \
  --amrwb_bitrates 15.85k \
  --gpus 0 \
  --limit 2000 \
  --target_limit 100 \
  --nontarget_limit 1900
```

### 注意事项
- 这个脚本会先生成混合训练音频，再启动训练，最后自动评测；耗时明显长于单独评测脚本
- 评测阶段默认复用 `run_cnceleb_codec_fixedrate_eval.py` 的协议：`clean,opus@16k,aac@16k,amrwb@15.85k`
- 训练前会先输出预训练基线结果：`pretrained_report.csv/json/md` 和 `pretrained_plots/`
- 训练后会输出微调结果：`report.csv/json/md` 和 `plots/`
- 对比结果会额外输出：`compare_pretrained_vs_finetuned.csv`、`compare_pretrained_vs_finetuned.md`、`compare_plots/`
- 训练阶段实际加载的是 fine-tune 目录下最新 checkpoint 的 `embedding_model.ckpt`
- 如果要只重跑评测，可加 `--skip_train`
- 如果要强制重建混合训练音频，可加 `--overwrite_data`
