# SV-CodecRestoreGAN 实现说明

本文档将 raw_method.md 的主线落地为可执行代码，目标是：

- 以 16 kHz coded 语音为输入
- 内部升采样到 48 kHz 做时频恢复
- 使用 CWS-TF-GridNet 风格生成器
- 在 phase2/phase3 引入 MRD+MBD GAN 约束
- 在 phase3 引入 WavLM 与 CAM++ 一致性损失
- 最终以 EER/minDCF 作为核心模型选择指标

## 1. 目录结构

- 代码根目录：my_methods_GAN
- 模型与训练主包：my_methods_GAN/sv_codec_restore_gan
- 命令入口脚本：my_methods_GAN/scripts
- 文档目录：my_methods_GAN/note

核心子目录：

- data：pair manifest 与数据集
- models：生成器、判别器、损失、WavLM/CAM++ 封装
- train：三阶段训练引擎
- utils：音频、指标、随机种子、IO

## 2. 网络与损失设计

### 2.1 生成器

- 文件：sv_codec_restore_gan/models/generator.py
- 路径：16k -> 48k -> STFT -> CWS split(3) -> GridNetBlock x 5 -> merge -> iSTFT -> skip add -> 16k
- 默认参数：emb_dim=48, num_blocks=5, hidden_units=100, attn_heads=4

### 2.2 判别器

- 文件：sv_codec_restore_gan/models/discriminators.py
- MRD：多分辨率时域判别
- MBD：多带宽时域判别

### 2.3 损失

- 文件：sv_codec_restore_gan/models/losses.py
- 重建损失：
  - L_rec = 2*L_SDR + 1.5*L_LSD + 70*L_mag + 30*(L_real + L_imag)
- GAN损失：
  - L_gan = L_adv + 0.2*L_feat
- phase3 额外任务损失：
  - L_wavlm = 1 - cos(WavLM(restored), WavLM(clean))
  - L_spk = 1 - cos(CAM++(restored), CAM++(clean))
- phase3 总损失：
  - L = 10*L_rec + L_adv + 0.2*L_feat + 1.0*L_wavlm + 3.0*L_spk

## 3. 三阶段训练

- 文件：sv_codec_restore_gan/train/engine.py

### Phase1

- 仅训练生成器
- 优化目标：10*L_rec
- 建议：先确认验证集 rec 指标稳定下降

学习率策略（当前实现）：

- 优化器：AdamW（G 与 D）
- 调度器：线性热身 + 余弦退火（step 级）
- G 在全训练步长上调度，D 在 phase2+phase3 有效步长上调度

### Phase2

- 打开 MRD+MBD
- 优化目标：10*L_rec + L_adv + 0.2*L_feat

### Phase3

- 冻结 WavLM 与 CAM++ 特征抽取分支
- 优化目标：10*L_rec + L_adv + 0.2*L_feat + 1.0*L_wavlm + 3.0*L_spk
- 推荐以 EER/minDCF 选优

### 3.1 稳健默认超参（推荐）

适用于单卡先跑通并观察收敛趋势：

- phase1_epochs=10, phase2_epochs=10, phase3_epochs=10
- batch_size=4, segment_seconds=2.0
- lr_g_max=5e-4, lr_g_min=5e-6
- lr_d_max=1e-4, lr_d_min=1e-6
- warmup_steps_g=1000, warmup_steps_d=1000
- weight_decay=1e-4, grad_clip=5.0

如果训练集规模较大（>100k 对）可用更保守版本：

- warmup_steps_g=3000, warmup_steps_d=3000
- lr_g_min=1e-5, lr_d_min=2e-6

## 4. 数据组织约定

建议采用 clean 与 coded 镜像目录：

- clean_root/spk_id/xxx.wav
- coded_root/spk_id/xxx.wav

或更深层路径，但 clean 与 coded 的相对路径必须一致。

由 build_sv_codec_manifest.py 自动对齐相对路径并输出 manifest：

- 列：utt_id, spk_id, clean_wav, codec_wav

### 4.1 CN-Celeb true-pair 生成流程（推荐）

为避免 clean 与 codec 非同源，建议直接从原始 CN-Celeb_flac 构建 true-pair 数据。

使用脚本：

- my_methods_GAN/scripts/prepare_cnceleb_truepair_data.py
- my_methods_GAN/scripts/build_sv_codec_manifest.py
- my_methods_GAN/scripts/split_sv_manifest_by_speaker.py
- my_methods_GAN/scripts/make_cnceleb_eval_scp.py

流程顺序：

1. 从 raw_data/CN-Celeb_flac 转出 clean_train_wav 与 eval_clean。
2. 对同一批 clean wav 编解码生成 coded_train_* 与 eval_coded_*。
3. 用 clean_train_wav 与 coded_train_* 生成 pair_manifest_all.csv。
4. 按 speaker 切分 train_manifest.csv 与 valid_manifest.csv。
5. 按 trials.lst 生成 eval_clean.scp 与 eval_coded.scp 用于 SV 评测。

这样可以确保训练与评测的 clean/coded 是同源一一对应对。

## 5. 评测

- 文件：scripts/eval_sv_codec_restore_gan.py
- 输入：clean_wav.scp, coded_wav.scp, trials
- 输出：clean/coded/restored 三组 EER 与 minDCF
- 目标：restored 在 EER/minDCF 上优于 coded

## 6. 现有依赖路径

- ESPnet：my_methods_GAN/espnet
- WavLM 权重：my_methods_GAN/pretrained/WavLM/WavLM-Base.pt
- WavLM 调用参考：my_methods_GAN/pretrained/WavLM/WavLM_use.py

说明：若本地 WavLM.py 不在 my_methods_GAN/pretrained/WavLM 下，请补齐后再运行 phase3。
