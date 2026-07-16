# 语音编解码对比实验说明

## 1. 文档用途

本文档用于记录当前仓库中“预训练 CAM++ 模型 + 语音编解码退化 + EER 对比评测”的实验流程。

当前实验链路对应的脚本为：

- `egs/3dspeaker/sv-cam++/local/simulate_codec.py`
- `egs/3dspeaker/sv-cam++/local/evaluate_eer.py`
- `egs/3dspeaker/sv-cam++/local/run_pretrained_codec_eval.py`

当前默认测试数据为：

- 干净测试音频目录：`egs/3dspeaker/sv-cam++/data/vctk_16k/wav16k`
- trials 文件：`egs/3dspeaker/sv-cam++/data/vctk/trials/trials`

## 2. 当前仓库中各编解码方式的实际实现

### 2.1 Opus

- 编码器：`libopus`
- 典型场景：现代互联网实时语音通信、WebRTC 类场景
- 主要实验变量：码率，例如 `32k`、`16k`、`8k`、`4k`
- 当前仓库中的处理方式：
  1. 将原始 wav 编码为 Opus
  2. 再解码回 wav
  3. 最终统一恢复为 `16 kHz`，用于 CAM++ 推理

### 2.2 G.711 mu-law

- 编码器：`pcm_mulaw`
- 典型场景：传统电话语音、窄带 VoIP
- 特点：
  - 窄带语音
  - 一般是 `8 kHz`
  - 对频带限制较明显
- 当前仓库中的处理方式：
  1. 先转为 `8 kHz` 单声道
  2. 以 `mu-law` 编码
  3. 再解码回 wav
  4. 最终统一恢复为 `16 kHz`

### 2.3 G.711 A-law

- 编码器：`pcm_alaw`
- 典型场景：传统电话语音、部分 PSTN/VoIP 系统
- 特点：
  - 窄带语音
  - 一般是 `8 kHz`
  - 与 mu-law 属于同类电话编码，但压扩规则不同
- 当前仓库中的处理方式：
  1. 先转为 `8 kHz` 单声道
  2. 以 `A-law` 编码
  3. 再解码回 wav
  4. 最终统一恢复为 `16 kHz`

### 2.4 AMR-WB

- 编码器：`libvo_amrwbenc`
- 中间压缩文件格式：`.amr`
- 典型场景：移动宽带语音通信
- 特点：
  - 宽带语音编码
  - 常见采样率为 `16 kHz`
  - 适合模拟移动语音链路中的压缩失真
- 当前仓库中的处理方式：
  1. 使用 `libvo_amrwbenc` 编码
  2. 中间压缩文件保存为 `.amr`
  3. 再解码回 wav
  4. 最终统一保持为 `16 kHz`

这里需要特别说明：

**当前仓库中 AMR-WB 的实际实现是 `libvo_amrwbenc + .amr`，不是 `amr_wb + .awb`。**

## 3. 一键评测脚本

推荐直接使用：

- `egs/3dspeaker/sv-cam++/local/run_pretrained_codec_eval.py`

该脚本会自动完成以下步骤：

1. 读取 `trials`
2. 找出这些 `trials` 实际引用到的全部 utterance
3. 仅对这些 utterance 生成 degraded 音频
4. 评测 clean EER
5. 评测 degraded EER
6. 输出 summary
7. 可选保存 JSON 报告

## 4. 通用命令模板

```powershell
python egs\3dspeaker\sv-cam++\local\run_pretrained_codec_eval.py `
  --model_id iic/speech_campplus_sv_zh-cn_16k-common `
  --trials_file egs\3dspeaker\sv-cam++\data\vctk\trials\trials `
  --clean_wav_root egs\3dspeaker\sv-cam++\data\vctk_16k\wav16k `
  --degraded_wav_root <degraded_wav_root> `
  --codec <codec_name> `
  --bitrate <bitrate_if_needed> `
  --sample_rate 16000 `
  --fraction 1 `
  --overwrite `
  --report_file <report_json>
```

说明：

- `--fraction 1` 表示使用全部 trials
- `--fraction 0.2` 表示只使用前 20% 的 trials
- `opus` 和 `amrwb` 需要传 `--bitrate`
- `g711_mulaw` 和 `g711_alaw` 一般不需要传 `--bitrate`

## 5. 可直接复制的实验命令

### 5.1 Opus 32k

```powershell
python egs\3dspeaker\sv-cam++\local\run_pretrained_codec_eval.py `
  --model_id iic/speech_campplus_sv_zh-cn_16k-common `
  --trials_file egs\3dspeaker\sv-cam++\data\vctk\trials\trials `
  --clean_wav_root egs\3dspeaker\sv-cam++\data\vctk_16k\wav16k `
  --degraded_wav_root egs\3dspeaker\sv-cam++\data\vctk_opus32k\wav16k `
  --codec opus `
  --bitrate 32k `
  --sample_rate 16000 `
  --fraction 1 `
  --overwrite `
  --report_file egs\3dspeaker\sv-cam++\exp\pretrained_codec_eval_opus32k_full.json
```

### 5.2 Opus 16k

```powershell
python egs\3dspeaker\sv-cam++\local\run_pretrained_codec_eval.py `
  --model_id iic/speech_campplus_sv_zh-cn_16k-common `
  --trials_file egs\3dspeaker\sv-cam++\data\vctk\trials\trials `
  --clean_wav_root egs\3dspeaker\sv-cam++\data\vctk_16k\wav16k `
  --degraded_wav_root egs\3dspeaker\sv-cam++\data\vctk_opus16k\wav16k `
  --codec opus `
  --bitrate 16k `
  --sample_rate 16000 `
  --fraction 1 `
  --overwrite `
  --report_file egs\3dspeaker\sv-cam++\exp\pretrained_codec_eval_opus16k_full.json
```

### 5.3 Opus 8k

```powershell
python egs\3dspeaker\sv-cam++\local\run_pretrained_codec_eval.py `
  --model_id iic/speech_campplus_sv_zh-cn_16k-common `
  --trials_file egs\3dspeaker\sv-cam++\data\vctk\trials\trials `
  --clean_wav_root egs\3dspeaker\sv-cam++\data\vctk_16k\wav16k `
  --degraded_wav_root egs\3dspeaker\sv-cam++\data\vctk_opus8k\wav16k `
  --codec opus `
  --bitrate 8k `
  --sample_rate 16000 `
  --fraction 1 `
  --overwrite `
  --report_file egs\3dspeaker\sv-cam++\exp\pretrained_codec_eval_opus8k_full.json
```

### 5.4 Opus 4k

```powershell
python egs\3dspeaker\sv-cam++\local\run_pretrained_codec_eval.py `
  --model_id iic/speech_campplus_sv_zh-cn_16k-common `
  --trials_file egs\3dspeaker\sv-cam++\data\vctk\trials\trials `
  --clean_wav_root egs\3dspeaker\sv-cam++\data\vctk_16k\wav16k `
  --degraded_wav_root egs\3dspeaker\sv-cam++\data\vctk_opus4k\wav16k `
  --codec opus `
  --bitrate 4k `
  --sample_rate 16000 `
  --fraction 1 `
  --overwrite `
  --report_file egs\3dspeaker\sv-cam++\exp\pretrained_codec_eval_opus4k_full.json
```

### 5.5 G.711 mu-law

```powershell
python egs\3dspeaker\sv-cam++\local\run_pretrained_codec_eval.py `
  --model_id iic/speech_campplus_sv_zh-cn_16k-common `
  --trials_file egs\3dspeaker\sv-cam++\data\vctk\trials\trials `
  --clean_wav_root egs\3dspeaker\sv-cam++\data\vctk_16k\wav16k `
  --degraded_wav_root egs\3dspeaker\sv-cam++\data\vctk_g711_mulaw\wav16k `
  --codec g711_mulaw `
  --sample_rate 16000 `
  --fraction 1 `
  --overwrite `
  --report_file egs\3dspeaker\sv-cam++\exp\pretrained_codec_eval_g711_mulaw_full.json
```

### 5.6 G.711 A-law

```powershell
python egs\3dspeaker\sv-cam++\local\run_pretrained_codec_eval.py `
  --model_id iic/speech_campplus_sv_zh-cn_16k-common `
  --trials_file egs\3dspeaker\sv-cam++\data\vctk\trials\trials `
  --clean_wav_root egs\3dspeaker\sv-cam++\data\vctk_16k\wav16k `
  --degraded_wav_root egs\3dspeaker\sv-cam++\data\vctk_g711_alaw\wav16k `
  --codec g711_alaw `
  --sample_rate 16000 `
  --fraction 1 `
  --overwrite `
  --report_file egs\3dspeaker\sv-cam++\exp\pretrained_codec_eval_g711_alaw_full.json
```

### 5.7 AMR-WB 12.65k

```powershell
python egs\3dspeaker\sv-cam++\local\run_pretrained_codec_eval.py `
  --model_id iic/speech_campplus_sv_zh-cn_16k-common `
  --trials_file egs\3dspeaker\sv-cam++\data\vctk\trials\trials `
  --clean_wav_root egs\3dspeaker\sv-cam++\data\vctk_16k\wav16k `
  --degraded_wav_root egs\3dspeaker\sv-cam++\data\vctk_amrwb_1265\wav16k `
  --codec amrwb `
  --bitrate 12.65k `
  --sample_rate 16000 `
  --fraction 1 `
  --overwrite `
  --report_file egs\3dspeaker\sv-cam++\exp\pretrained_codec_eval_amrwb_1265_full.json
```

### 5.8 AMR-WB 23.85k

```powershell
python egs\3dspeaker\sv-cam++\local\run_pretrained_codec_eval.py `
  --model_id iic/speech_campplus_sv_zh-cn_16k-common `
  --trials_file egs\3dspeaker\sv-cam++\data\vctk\trials\trials `
  --clean_wav_root egs\3dspeaker\sv-cam++\data\vctk_16k\wav16k `
  --degraded_wav_root egs\3dspeaker\sv-cam++\data\vctk_amrwb_2385\wav16k `
  --codec amrwb `
  --bitrate 23.85k `
  --sample_rate 16000 `
  --fraction 1 `
  --overwrite `
  --report_file egs\3dspeaker\sv-cam++\exp\pretrained_codec_eval_amrwb_2385_full.json
```

## 6. 建议实验顺序

建议按以下顺序推进：

1. `Opus 16k`
2. `Opus 8k`
3. `Opus 4k`
4. `G.711 mu-law`
5. `G.711 A-law`
6. `AMR-WB 12.65k`
7. `AMR-WB 23.85k`

原因：

- `Opus` 代表现代互联网语音通信
- `G.711` 代表传统窄带电话链路
- `AMR-WB` 代表移动宽带语音链路

## 7. 实验记录时建议保留的字段

每次实验建议记录以下信息：

- 模型 ID
- 数据集名称
- trials 文件路径
- codec 类型
- bitrate
- fraction
- clean EER
- degraded EER
- degradation factor
- degraded 音频目录
- report json 路径
- 备注

## 8. 结果分析建议

对比结果时可以按照以下逻辑分析：

- 如果 `degraded EER` 与 `clean EER` 接近，说明模型对该 codec 条件较鲁棒
- 如果 `degraded EER` 明显高于 `clean EER`，说明该 codec 对说话人判别信息破坏较强
- 如果随着 bitrate 降低，`degradation factor` 持续增大，说明性能恶化趋势与压缩强度一致

## 9. 实验注意事项

- 调试阶段可使用 `--fraction 0.1` 或 `--fraction 0.2`
- 正式实验建议使用 `--fraction 1`
- 做 codec 扫描时，尽量固定同一份 `trials`
- 做 codec 扫描时，尽量固定同一份 `clean_wav_root`
- 每次只改变一个变量，便于后续画表和写论文
