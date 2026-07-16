# 生成 VoxCeleb1 测试集与 `trials.lst`

> **协议更正：** VoxCeleb1 官方 `veri_test2.txt` 已经直接给出两条待比较语音，正式评测应使用其中的 37,611 个 verification trials。本文前半部分记录的 enroll/test 笛卡尔积方案仅保留作历史说明，不应再用于论文指标对比。

脚本位置：

```text
/root/camplusplus/my_methods_GAN/scripts/prepare_vox1_test_raw.py
```

脚本读取官方 `veri_test2.txt` 中出现的全部唯一语音。每位说话人按路径排序后选择第一条作为 enroll，剩余语音全部作为 test。这样选择是确定性的，多次运行结果一致。

输出结构如下：

```text
/root/autodl-tmp/SC_data/voxceleb_data_test_raw/
├── enroll/
│   ├── id10270-enroll.wav
│   └── ...
├── test/
│   ├── id10270/
│   │   ├── id10270-8jEAjG6SegY-00008.wav
│   │   └── ...
│   └── ...
└── trials.lst
```

`id/video/00001.wav` 会被命名为 `id-video-00001.wav`，避免不同视频目录中的 `00001.wav` 相互覆盖。`trials.lst` 使用全部 enroll 与全部 test 的笛卡尔积，和原 CN-Celeb 评测列表保持同一结构：

```text
id10270-enroll test/id10270/某条语音.wav 1
id10270-enroll test/id10300/某条语音.wav 0
```

## 执行方法

先做只读检查；该命令不会创建、复制或删除任何文件：

```bash
python3 /root/camplusplus/my_methods_GAN/scripts/prepare_vox1_test_raw.py --dry-run
```

当前数据应显示 40 位说话人、40 条 enroll、4,668 条 test 和 186,720 条 trial。

确认统计正确后正式执行：

```bash
python3 /root/camplusplus/my_methods_GAN/scripts/prepare_vox1_test_raw.py
```

默认使用复制，源 WAV 不会被修改。如果希望节省磁盘空间，可以改用硬链接：

```bash
python3 /root/camplusplus/my_methods_GAN/scripts/prepare_vox1_test_raw.py --link-mode hardlink
```

若输出目录已经存在，脚本会停止以防误删。确认需要完整删除并重新生成该目录时，才添加 `--overwrite`：

```bash
python3 /root/camplusplus/my_methods_GAN/scripts/prepare_vox1_test_raw.py --overwrite
```

## 默认输入与输出

```text
评测列表：/root/autodl-tmp/raw_data/vox1/data/vox1_trials/veri_test2.txt
源 WAV：   /root/autodl-tmp/raw_data/vox1/train/wav
输出目录： /root/autodl-tmp/SC_data/voxceleb_data_test_raw
```

如路径变化，可分别通过 `--trials`、`--wav-root` 和 `--output-dir` 指定新路径。运行 `python3 .../prepare_vox1_test_raw.py --help` 可查看所有参数。

## VoxCeleb1 训练集真配对转码

训练集脚本：

```text
/root/camplusplus/my_methods_GAN/scripts/prepare_vox1_truepair_train_data.py
```

该脚本直接读取 VoxCeleb 的 `id/video/*.wav`，不需要 CN-Celeb 的 `dev/dev.lst`。它会从官方 `veri_test2.txt` 提取 40 位测试说话人并将其全部排除，只处理剩余 1,211 位开发集说话人，防止训练集与官方测试集发生说话人泄漏。

先进行只读检查：

```bash
cd /root/workspace/camplusplus

python my_methods_GAN/scripts/prepare_vox1_truepair_train_data.py \
  --raw_root /root/autodl-tmp/raw_data/vox1/train/wav \
  --test_wav_root /root/autodl-tmp/raw_data/vox1/test/wav \
  --test_trials /root/autodl-tmp/raw_data/vox1/data/vox1_trials/veri_test2.txt \
  --output_root /root/autodl-tmp/raw_data/vox1_processed \
  --codec opus \
  --bitrate 16k \
  --sample_rate 16000 \
  --clean_write_mode hardlink \
  --workers 16 \
  --dry_run
```

当前数据已经拆分为 `train/wav` 和 `test/wav`。预期检查结果为：训练树 1,211 位说话人、148,642 条训练语音；测试树 40 位官方测试说话人、4,874 条测试语音；从训练树中需要额外排除的测试语音数为 0。

确认统计正确后正式转码：

```bash
cd /root/workspace/camplusplus

python my_methods_GAN/scripts/prepare_vox1_truepair_train_data.py \
  --raw_root /root/autodl-tmp/raw_data/vox1/train/wav \
  --test_wav_root /root/autodl-tmp/raw_data/vox1/test/wav \
  --test_trials /root/autodl-tmp/raw_data/vox1/data/vox1_trials/veri_test2.txt \
  --output_root /root/autodl-tmp/raw_data/vox1_processed \
  --codec opus \
  --bitrate 16k \
  --sample_rate 16000 \
  --clean_write_mode hardlink \
  --workers 16 \
  --overwrite
```

`--clean_write_mode hardlink` 表示 clean 输出不再经过 ffmpeg 重新转码，而是在
`clean_train_wav` 中创建指向原始 VoxCeleb1 WAV 的硬链接；如果源目录和输出目录不在同一
文件系统，脚本会自动退回到复制。若需要保持旧行为，可改回默认的
`--clean_write_mode normalize`。

输出目录为：

```text
/root/rivermind-data/experiment_data_voxceleb/train_data/opus/clean_train_wav/
/root/rivermind-data/experiment_data_voxceleb/train_data/opus/coded_train_opus_16k/
```

两边保持相同的 `说话人/视频/文件.wav` 相对路径，因此每条 clean 与 coded 音频严格一一对应。`--overwrite` 会重新生成已存在的目标文件；若希望断点续做并跳过已有文件，去掉该参数即可。
