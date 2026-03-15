# 实验记录表模板

## 1. 文档用途

本文档用于记录当前仓库中的实验结果，适配当前项目的实际结构与流程：

- 数据集：VCTK
- 训练入口：`egs/3dspeaker/sv-cam++/run.sh`
- 配置文件：`egs/3dspeaker/sv-cam++/conf/cam++.yaml`
- 预训练 codec baseline 脚本：`egs/3dspeaker/sv-cam++/local/run_pretrained_codec_eval.py`
- 干净测试音频目录：`egs/3dspeaker/sv-cam++/data/vctk_16k/wav16k`
- trials 文件：`egs/3dspeaker/sv-cam++/data/vctk/trials/trials`

该模板主要分为四类记录：

1. 预训练 CAM++ 的 codec 鲁棒性实验
2. 自训练 CAM++ 的训练配置记录
3. 自训练 CAM++ 的评测结果记录
4. 后续鲁棒性方法对比实验

## 2. 预训练 CAM++ codec baseline 记录表

适用于：

- `egs/3dspeaker/sv-cam++/local/run_pretrained_codec_eval.py`

说明：

- 以下已填写结果来自 `egs/3dspeaker/sv-cam++/exp/pretrained_codec_eval_*.json`
- 当前这些结果均为 `fraction = 1.0`
- 同一批 clean trials 的 `Clean EER` 都是 `2.250%`

| 实验编号 | 日期 | 模型 ID | 数据集 | Trials 文件 | Codec | Bitrate | Fraction | Clean EER (%) | Degraded EER (%) | 性能恶化倍数 | Degraded 音频目录 | JSON 报告路径 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| P-001 |  | `iic/speech_campplus_sv_zh-cn_16k-common` | `VCTK` | `egs/3dspeaker/sv-cam++/data/vctk/trials/trials` | `opus` | `16k` | `1.0` | 2.250 | 2.250 | 1.000x | `egs/3dspeaker/sv-cam++/data/vctk_opus16k/wav16k` | `egs/3dspeaker/sv-cam++/exp/pretrained_codec_eval_opus16k_full.json` | 与 clean 基本一致 |
| P-002 |  | `iic/speech_campplus_sv_zh-cn_16k-common` | `VCTK` | `egs/3dspeaker/sv-cam++/data/vctk/trials/trials` | `opus` | `8k` | `1.0` | 2.250 | 7.250 | 3.222x | `egs/3dspeaker/sv-cam++/data/vctk_opus8k/wav16k` | `egs/3dspeaker/sv-cam++/exp/pretrained_codec_eval_opus8k_full.json` | 明显退化 |
| P-003 |  | `iic/speech_campplus_sv_zh-cn_16k-common` | `VCTK` | `egs/3dspeaker/sv-cam++/data/vctk/trials/trials` | `opus` | `6k` | `1.0` | 2.250 | 9.500 | 4.222x | `egs/3dspeaker/sv-cam++/data/vctk_opus6k/wav16k` | `egs/3dspeaker/sv-cam++/exp/pretrained_codec_eval_opus6k_full.json` | 低码率退化进一步加重 |
| P-004 |  | `iic/speech_campplus_sv_zh-cn_16k-common` | `VCTK` | `egs/3dspeaker/sv-cam++/data/vctk/trials/trials` | `opus` | `4k` | `1.0` | 2.250 | 14.750 | 6.556x | `egs/3dspeaker/sv-cam++/data/vctk_opus4k/wav16k` | `egs/3dspeaker/sv-cam++/exp/pretrained_codec_eval_opus4k_full.json` | 退化最明显 |
| P-005 |  | `iic/speech_campplus_sv_zh-cn_16k-common` | `VCTK` | `egs/3dspeaker/sv-cam++/data/vctk/trials/trials` | `g711_mulaw` | `8k` | `1.0` | 2.250 | 4.250 | 1.889x | `egs/3dspeaker/sv-cam++/data/vctk_g711_mulaw/wav16k` | `egs/3dspeaker/sv-cam++/exp/pretrained_codec_eval_g711_mulaw_full.json` | 电话窄带条件下退化明显 |
| P-006 |  | `iic/speech_campplus_sv_zh-cn_16k-common` | `VCTK` | `egs/3dspeaker/sv-cam++/data/vctk/trials/trials` | `g711_alaw` | `8k` | `1.0` | 2.250 | 4.750 | 2.111x | `egs/3dspeaker/sv-cam++/data/vctk_g711_alaw/wav16k` | `egs/3dspeaker/sv-cam++/exp/pretrained_codec_eval_g711_alaw_full.json` | 略差于 mu-law |
| P-007 |  | `iic/speech_campplus_sv_zh-cn_16k-common` | `VCTK` | `egs/3dspeaker/sv-cam++/data/vctk/trials/trials` | `amrwb` | `23.85k` | `1.0` | 2.250 | 2.250 | 1.000x | `egs/3dspeaker/sv-cam++/data/vctk_amrwb_2385/wav16k` | `egs/3dspeaker/sv-cam++/exp/pretrained_codec_eval_amrwb_2385_full.json` | 当前仓库实现为 `libvo_amrwbenc + .amr` |
| P-008 |  | `iic/speech_campplus_sv_zh-cn_16k-common` | `VCTK` | `egs/3dspeaker/sv-cam++/data/vctk/trials/trials` | `amrwb` | `12.65k` | `1.0` | 2.250 | 2.250 | 1.000x | `egs/3dspeaker/sv-cam++/data/vctk_amrwb_1265/wav16k` | `egs/3dspeaker/sv-cam++/exp/pretrained_codec_eval_amrwb_1265_full.json` | 当前仓库实现为 `libvo_amrwbenc + .amr` |
| P-009 |  | `iic/speech_campplus_sv_zh-cn_16k-common` | `VCTK` | `egs/3dspeaker/sv-cam++/data/vctk/trials/trials` | `amrwb` | `8.85k` | `1.0` | 2.250 | 3.500 | 1.556x | `egs/3dspeaker/sv-cam++/data/vctk_amrwb_885/wav16k` | `egs/3dspeaker/sv-cam++/exp/pretrained_codec_eval_amrwb_885_full.json` | 低码率 AMR-WB 开始出现退化 |

## 3. 自训练 CAM++ 配置记录表

适用于：

- `egs/3dspeaker/sv-cam++/run.sh`
- `egs/3dspeaker/sv-cam++/conf/cam++.yaml`

| 实验编号 | 日期 | 实验名 | 数据集 | 配置文件 | Train CSV | Test wav.scp | Trials 文件 | 说话人数 | 每人训练条数 | 每人测试条数 | Batch Size | Epoch | 学习率 | Num Workers | GPU 设置 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| T-001 |  | `campp_vctk` | `VCTK` | `egs/3dspeaker/sv-cam++/conf/cam++.yaml` | `egs/3dspeaker/sv-cam++/data/vctk/train/train.csv` | `egs/3dspeaker/sv-cam++/data/vctk/test/wav.scp` | `egs/3dspeaker/sv-cam++/data/vctk/trials/trials` | 20 | 100 | 20 | 32 | 30 | 0.02 | 0 | `0` | 当前是适配 Windows 单卡的配置 |

## 4. 自训练 CAM++ 评测结果记录表

适用于：

- 使用 `run.sh` 完成训练与评测后的结果记录

| 实验编号 | 日期 | 实验名 | Checkpoint 路径 | 数据集 | Trials 文件 | Clean EER (%) | minDCF | Embedding 目录 | Score 文件 | 配置快照 | 备注 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| E-001 |  | `campp_vctk` | `egs/3dspeaker/sv-cam++/exp/campp_vctk/models/CKPT-EPOCH-30-00` | `VCTK` | `egs/3dspeaker/sv-cam++/data/vctk/trials/trials` | 1.500 | 0.0400 | `egs/3dspeaker/sv-cam++/exp/campp_vctk/embeddings` | `egs/3dspeaker/sv-cam++/exp/campp_vctk/scores/trials.score` | `egs/3dspeaker/sv-cam++/conf/cam++.yaml` | 来自当前 VCTK 单卡训练实验 |

## 5. 鲁棒性方法对比表

后续如果你要比较：

- 预训练 CAM++
- 自训练 CAM++
- 微调后的 CAM++
- 多条件训练或 codec augmentation 方法

建议用下表统一记录：

| 方法 | 权重来源 | 训练条件 | 测试 Codec | Bitrate | Clean EER (%) | Degraded EER (%) | 性能恶化倍数 | 相对预训练提升 | 备注 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 预训练 CAM++ | ModelScope | 无 | `opus` | `8k` | 2.250 | 7.250 | 3.222x | baseline |  |
| 自训练 CAM++ | 本地训练 | clean only | `opus` | `8k` |  |  |  |  |  |
| 微调 CAM++ | 本地训练 | clean + degraded | `opus` | `8k` |  |  |  |  |  |
| 你的方法 | 本地训练 |  | `opus` | `8k` |  |  |  |  |  |

## 6. 论文主表建议格式

建议论文主表采用如下结构：

| 模型 | Clean | Opus 16k | Opus 8k | Opus 6k | Opus 4k | G.711 mu-law | G.711 A-law | AMR-WB 23.85k | AMR-WB 12.65k | AMR-WB 8.85k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 预训练 CAM++ | 2.250 | 2.250 | 7.250 | 9.500 | 14.750 | 4.250 | 4.750 | 2.250 | 2.250 | 3.500 |
| 自训练 CAM++ | 1.500 |  |  |  |  |  |  |  |  |  |
| 微调 CAM++ |  |  |  |  |  |  |  |  |  |  |
| 你的方法 |  |  |  |  |  |  |  |  |  |  |

如果需要，也可以再做一张“性能恶化倍数表”，把原始 EER 换成 `degradation factor`。

## 7. 每次实验建议记录的信息

每次正式实验，至少要补全以下内容：

- 实际运行命令
- 配置文件路径
- 数据划分方式
- codec 类型与 bitrate
- 是否全量 trials
- 输出的 JSON、score 或 log 路径
- clean EER
- degraded EER
- 是否为正式结果还是调试结果

## 8. 当前阶段可直接使用的结论

根据当前已经完成的 baseline 结果，可以先得到以下观察：

- `Opus 16k` 对当前预训练 CAM++ 几乎没有影响
- `Opus 8k / 6k / 4k` 会随着码率降低出现明显退化
- `G.711` 会带来中等程度退化，其中 `A-law` 略差于 `mu-law`
- `AMR-WB 23.85k` 和 `12.65k` 基本接近 clean
- `AMR-WB 8.85k` 开始出现可见退化
- 当前自训练 CAM++ 在 clean VCTK 上的 EER 为 `1.500%`，优于当前预训练 baseline 的 `2.250%`

## 9. 当前仓库中的固定约定

当前项目下建议统一使用以下路径约定：

- VCTK 元数据输出目录：
  - `egs/3dspeaker/sv-cam++/data/vctk`
- 干净 16k 测试音频目录：
  - `egs/3dspeaker/sv-cam++/data/vctk_16k/wav16k`
- 预训练 codec baseline 脚本：
  - `egs/3dspeaker/sv-cam++/local/run_pretrained_codec_eval.py`
- 自训练 recipe 入口：
  - `egs/3dspeaker/sv-cam++/run.sh`
- 当前仓库中的 AMR-WB 实现：
  - 编码器：`libvo_amrwbenc`
  - 中间压缩文件：`.amr`
