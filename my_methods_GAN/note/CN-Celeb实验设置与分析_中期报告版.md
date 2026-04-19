# CN-Celeb 实验设置与实验分析

本节围绕 CN-Celeb 数据集上的 SC-GridRestore 实验展开说明。实验目标是在不修改后端说话人识别模型的前提下，通过前端恢复模型减弱 codec 压缩造成的声纹表征偏移，并比较 clean、coded 与 restored 三种语音条件下的说话人验证性能。实验分析重点关注两个问题：第一，codec 压缩是否会显著降低后端说话人验证性能；第二，SC-GridRestore 生成器是否能够相对 coded 语音降低 EER 和 minDCF。

## 1. 实验数据与任务设置

实验使用 CN-Celeb 作为主要评测数据。CN-Celeb 是中文说话人识别领域常用的大规模数据集，包含大量真实场景语音，具有说话人数量多、录音条件复杂、信道差异明显等特点，适合用于检验 codec 失真条件下说话人验证系统的鲁棒性。本文实验从 CN-Celeb 中构建 clean/coded 成对数据，其中 clean 语音作为目标参考，coded 语音由 clean 语音经过指定 codec 转码得到。

训练阶段采用 clean/coded 成对样本，输入为 coded 语音 $x_c$，目标为对应 clean 语音 $x$。模型输出 restored 语音 $\hat{x}=G_\theta(x_c)$。评测阶段使用固定的 CAMP++ 说话人后端分别提取 clean、coded 和 restored 的说话人嵌入，并根据 CN-Celeb trials 计算 EER 和 minDCF。该设置保证后端模型在所有条件下完全一致，因此不同条件之间的性能差异主要反映 codec 失真和前端恢复模块的影响。

本文实验中使用的评测条件如下：

| 条件     | 输入语音                             | 说明                                |
| -------- | ------------------------------------ | ----------------------------------- |
| clean    | 原始 CN-Celeb 语音                   | 作为无 codec 失真的性能上限         |
| coded    | 经过 codec 转码后的语音              | 用于衡量 codec 失真对 SV 后端的影响 |
| restored | coded 经 SC-GridRestore 恢复后的语音 | 用于衡量前端恢复后的性能改善        |

其中，EER 越低表示系统在目标试验和非目标试验之间的区分能力越强；minDCF 越低表示在给定代价函数下系统的检测代价越小。因此，若 restored 条件下的 EER/minDCF 低于 coded 条件，则说明前端恢复模型对说话人验证任务具有正向作用。

### 1.1 CN-Celeb 数据结构细化（训练/测试）

为便于在中期报告中直接引用，下面给出当前项目实际使用的 CN-Celeb 结构统计（统计路径：`egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac`）。

| 子集                              | 用途             |    录音场景（genre） |                                 说话人个数 |           语音个数 | 正样本对 |  负样本对 |
| --------------------------------- | ---------------- | -------------------: | -----------------------------------------: | -----------------: | -------: | --------: |
| `data`（训练集）                  | 训练前端恢复模型 |                   11 |                                        997 |            126,532 |        - |         - |
| `eval/enroll`（测试注册端）       | 说话人模型注册   |                    - |                                        196 |                196 |        - |         - |
| `eval/test`（测试检索端）         | 试验端测试语音   |                   11 |                                        200 |             17,777 |        - |         - |
| `eval/lists/trials.lst`（测试对） | SV 验证打分      | 11（由 test 侧继承） | 196 个注册说话人模型（test 侧覆盖 200 人） | 3,484,292 对 trial |   17,755 | 3,466,537 |

说明：  
- `trials.lst` 中正负样本对比例约为 1:195.24，属于典型的大规模不平衡验证设置。  
- 训练集没有官方固定 trial 对，因此“正/负样本对个数”只在测试 trial 中定义。  
- 官方 README 对 eval 说话人数有“196/200”两种表述；本项目按实际文件统计为：`enroll` 196 人、`test` 200 人。  

训练集与测试集的录音场景分布如下（单位：条语音）：

| 录音场景       | 训练集 `data` | 测试集 `eval/test` |
| -------------- | ------------: | -----------------: |
| interview      |        59,556 |              6,175 |
| entertainment  |        22,137 |              3,434 |
| singing        |        12,561 |              2,017 |
| live_broadcast |         8,747 |              2,304 |
| speech         |         8,404 |              2,089 |
| drama          |         4,994 |                661 |
| play           |         4,244 |                 50 |
| recitation     |         2,747 |                105 |
| vlog           |         1,894 |                776 |
| movie          |         1,127 |                146 |
| advertisement  |           121 |                 20 |
| **合计**       |   **126,532** |         **17,777** |

若需要在报告中进一步展示“场景难度”，建议再补一张 `trials.lst` 的场景级正负样本对统计表（如 interview/singing/speech 各自的正负样本对），用于解释不同场景下 EER/minDCF 的潜在差异来源。

## 2. 数据构建与预处理

为了构建与说话人验证任务一致的训练数据，首先从 CN-Celeb 中提取 clean 训练语音，并通过脚本生成对应的 codec 版本。当前实验主要包含两类 codec 条件：Opus 与 AMR-WB。其中，Opus 用于模拟常见低延迟语音通信压缩，AMR-WB 用于模拟宽带语音通信场景中的编码失真。后续实验中也保留了 G.711 的数据生成逻辑，但中期阶段主要围绕 Opus 与 AMR-WB 展开。

训练 manifest 采用 clean/coded 配对形式。对于 multi-codec 数据，当前实现将同一 clean 语音对应的不同 codec 样本展开为多条训练样本，即同一条 clean 语音可以分别与 Opus 和 AMR-WB coded 语音组成训练 pair。这样做可以避免每次读取时随机选择 codec 带来的统计不稳定，也使每个 epoch 中不同 codec 条件都能稳定参与训练。

为避免波形级重建损失受到时间错位影响，数据读取阶段对 clean 和 coded 语音执行同起点裁剪。具体而言，先取两者共同长度，再生成统一裁剪起点，同时截取 clean 和 coded 片段；若长度不足，则对二者执行一致补零。此外，AMR-WB 转码可能带来固定时间偏移，因此当前 OPUS+AMR-WB 实验中启用了 codec 时间对齐，并对 AMR-WB coded 语音使用 95 个采样点的固定补偿。

当前主要训练数据规模如下：

| 数据配置          | train 样本数 | valid 样本数 | 说明                         |
| ----------------- | -----------: | -----------: | ---------------------------- |
| Opus q25          |        24075 |         2639 | 早期单 codec 训练与对比实验  |
| Opus + AMR-WB q25 |        48474 |         5321 | 当前 multi-codec B2 实验配置 |

注：表中样本数为去除 CSV 表头后的 manifest 行数，multi-codec 配置已经按 codec 展开为多行样本。

## 3. Codec-only 预实验设置与分析

在训练恢复模型之前，首先对不同 codec 条件下的 coded-only 语音进行预实验。该预实验不使用 SC-GridRestore 生成器，也不进行任何恢复处理，而是直接将不同 codec 转码后的语音输入同一个冻结 CAMP++ 后端，计算 CN-Celeb trials 上的 EER 和 minDCF。这样做的目的是排除前端模型影响，单独观察 codec 类型和码率对说话人验证性能的影响。

该预实验的设置如下：

| 条件       | 处理方式                | 后端模型 | 评价指标     | 目的                      |
| ---------- | ----------------------- | -------- | ------------ | ------------------------- |
| clean      | 原始语音                | CAMP++   | EER / minDCF | 作为无 codec 失真的上限   |
| coded-only | clean 经指定 codec 转码 | CAMP++   | EER / minDCF | 衡量不同 codec 的直接退化 |

在报告中，建议将 Opus、AMR-WB、AAC 和 G.711 的 coded-only 结果统一整理为一张表。该表不要求每种 codec 都训练恢复模型，只需要列出同一后端、同一 trials、同一评测脚本下的 coded-only 基线，即可支撑“不同 codec 会造成不同程度的说话人验证退化”这一实验动机。

| Codec                   | 码率/配置  | 主要失真类型                                 | EER (%) | 相对 clean 的 EER 增量 | minDCF | 相对 clean 的 minDCF 增量 |
| ----------------------- | ---------- | -------------------------------------------- | ------: | ---------------------: | -----: | ------------------------: |
| Clean                   | 无编码压缩 | 无 codec 失真                                |  7.1360 |                 0.0000 | 0.4112 |                    0.0000 |
| Opus                    | 16 kbps    | 低码率感知编码、频谱细节损失、瞬态平滑       |  9.1692 |                 2.0332 | 0.5123 |                    0.1011 |
| AMR-WB                  | 15.85 kbps | 宽带语音参数编码、固定时延、谱包络与相位扰动 | 10.4703 |                 3.3343 | 0.5149 |                    0.1037 |
| AAC                     | 32 kbps    | 变换域量化、掩蔽模型导致的高频/瞬态细节变化  |  7.9401 |                 0.8041 | 0.4428 |                    0.0316 |
| G.711 $\mu$-law / A-law | 64 kbps    | 波形压扩量化、量化噪声、非线性幅度失真       | 12.9019 |                 5.7659 | 0.6583 |                    0.2471 |

从 coded-only 全量结果可见，在同一后端与同一 trials 下，四种 codec 都会导致性能下降，但下降幅度差异显著。按 EER 从好到差排序为 AAC (7.9401) < Opus (9.1692) < AMR-WB (10.4703) < G.711 (12.9019)。相对 clean，四者的 EER 分别恶化 0.8041、2.0332、3.3343、5.7659 个百分点；minDCF 分别恶化 0.0316、0.1011、0.1037、0.2471。

这一结果直接支撑“不同 codec 对说话人识别影响不同”的结论。AAC 32 kbps 的退化最轻，说明较高码率下的感知编码对身份判别信息保留更充分；Opus 16 kbps 和 AMR-WB 15.85 kbps 退化更明显，其中 AMR-WB 的 EER 劣化高于 Opus，反映出参数化语音编码对说话人细粒度特征的破坏更强。

关于“码率越低性能越差”，本组结果在同类感知编码器中可以观察到一致趋势（AAC 32 kbps 明显优于 Opus 16 kbps），但跨 codec 不满足简单单调关系。典型例子是 G.711 虽然码率最高（64 kbps），却出现最差 EER/minDCF，说明退化程度不仅由码率决定，更受编码机理与后端特征匹配关系影响。

从机理上看，AAC/Opus 主要引入时频量化与掩蔽引起的细节损失，AMR-WB 还会叠加参数化建模与时延效应，而 G.711 的压扩量化会带来显著非线性幅度失真。这些失真类型作用于不同声学线索，因此会在 CAMP++ 嵌入空间中表现为不同方向和不同强度的分布偏移。

对后续恢复模型设计的启示是：单 codec 训练难以覆盖多样失真模式，multi-codec 训练是必要的；同时建议在后续章节优先报告对 AMR-WB 与 G.711 这两类“重退化”条件的恢复增益，以更清楚体现 SC-GridRestore 的实际价值。

上述 coded-only 数值均由统一评测脚本得到。复现实验时，可按如下流程运行：先为每种 codec 生成 eval coded 目录，再生成对应 scp，最后只评测 `clean,coded` 两种条件，不传入生成器 checkpoint。示例命令如下：

```bash
python my_methods_GAN/scripts/make_cnceleb_eval_scp.py \
  --trials_file egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac/eval/lists/trials.lst \
  --eval_clean_dir my_methods_GAN/data/cnceleb_truepair/eval_clean \
  --eval_coded_dir my_methods_GAN/data/cnceleb_truepair/eval_coded_<codec_name> \
  --clean_scp_out my_methods_GAN/exp/sv_codec_restore/eval_clean.scp \
  --coded_scp_out my_methods_GAN/exp/sv_codec_restore/eval_coded_<codec_name>.scp \
  --strict
```

```bash
python my_methods_GAN/scripts/eval_sv_codec_restore_gan.py \
  --clean_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_clean.scp \
  --coded_wav_scp my_methods_GAN/exp/sv_codec_restore/eval_coded_<codec_name>.scp \
  --trials_file egs/3dspeaker/sv-cam++/data/raw_data/CN-Celeb_flac/eval/lists/trials.lst \
  --conditions clean,coded \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --output_json my_methods_GAN/exp/sv_codec_restore/run_main/eval_codec_<codec_name>_coded_only.json \
  --use_cache \
  --cache_incremental \
  --cache_dir my_methods_GAN/exp/sv_codec_restore/run_main/emb_cache_<codec_name> \
  --device cuda
```

绘图时建议使用两张柱状图：第一张绘制不同 codec 的 EER，第二张绘制不同 codec 的 minDCF。图中保留 clean 作为横向参考线或第一组柱，以突出各 codec 相对 clean 的退化幅度。

## 4. 实验模型与训练配置

实验采用 SC-GridRestore 作为前端恢复模型。生成器采用 STFT 复谱输入、CWS 子带重排、TF-GridNet 主干和 iSTFT 残差重建结构。训练采用阶段式策略，其中 Phase 1 仅优化重建损失，Phase 2 在重建基础上加入说话人一致性约束，Phase 3 可进一步加入 GAN 与 WavLM 蒸馏项。中期阶段的主要实验集中在 Phase 1 和 Phase 2，原因是当前目标首先是验证恢复模型是否能够稳定改善 SV 指标，再进一步讨论对抗训练是否必要。

### 4.1 Experiment A：重建基线

Experiment A 作为 rec-only baseline，仅训练生成器并使用重建损失作为优化目标。该阶段不引入 CAMP++ teacher、GAN 或 WavLM 约束，主要用于学习 coded 到 clean 的基础声学映射，并为后续说话人一致性微调提供 warm-start checkpoint。

Experiment A 的损失函数为：

$$
\mathcal{L}_{A}
=\mathcal{L}_{rec}
=\alpha\mathcal{L}_{SI\text{-}SDR}
+\beta\mathcal{L}_{MRSTFT}
+\gamma\mathcal{L}_{CPLX}.
$$

在早期 q25 实验中，为了提高 rec-only 训练稳定性，complex loss 被暂时关闭，即 $\gamma=0$。该设置能够让模型优先学习时域 SI-SDR 和多分辨率幅度谱一致性，减少复谱约束在训练早期造成的优化压力。

### 4.2 Experiment B：CAMP++ 说话人约束微调

Experiment B 在 Experiment A checkpoint 的基础上进行微调，加入冻结 CAMP++ 后端提供的说话人一致性损失。其核心思想是：恢复语音不仅要在波形和频谱上接近 clean，还应在后端说话人嵌入空间中接近 clean。训练时 clean 分支作为 teacher reference，restored 分支保留梯度，使说话人损失可以通过 CAMP++ 网络反传到生成器。

Experiment B 的主要损失为：

$$
\mathcal{L}_{B}
=\mathcal{L}_{rec}
+\lambda_{spk}\left(1-\cos(E_{camp}(\hat{x}),E_{camp}(x))\right).
$$

其中，$E_{camp}(\cdot)$ 表示冻结的 CAMP++ 说话人后端。该实验主要验证 speaker-aware 微调是否能够比 rec-only 恢复更有效地降低 EER/minDCF。

### 4.3 Experiment B2：CAMP++ embedding + deep feature loss

Experiment B2 是当前重点推进的稳定版本。该版本在 Experiment B 的基础上加入 CAMP++ 中间层 deep feature loss，并关闭 AM-Softmax 分类损失，以减少分类目标和重建目标之间的梯度竞争。B2 训练中 CAMP++ 前端采用可微的 log-Mel frontend，而不是不可微的 Kaldi fbank，从而保证 restored 分支上的 teacher loss 能够稳定回传到生成器。

B2 的损失可写为：

$$
\mathcal{L}_{B2}
=\mathcal{L}_{rec}
+\lambda_{emb}\mathcal{L}_{spk\text{-}cos}
+\lambda_{feat}\mathcal{L}_{spk\text{-}feat},
$$

其中：

$$
\mathcal{L}_{spk\text{-}cos}
=1-\cos(E_{camp}(\hat{x}),E_{camp}(x)),
$$

$$
\mathcal{L}_{spk\text{-}feat}
=\frac{1}{|\Omega|}
\sum_{\ell\in\Omega}
\left\|
F_{camp}^{(\ell)}(\hat{x})-F_{camp}^{(\ell)}(x)
\right\|_1.
$$

当前 OPUS+AMR-WB B2 实验配置为：Phase 1 关闭、Phase 2 训练 10 个 epoch、Phase 3 关闭；启用 CAMP++ embedding loss 与 feature loss；关闭 AM-Softmax；关闭 GAN 与 WavLM；speaker loss 权重为 3.0，并从 OPUS+AMR-WB 的 Experiment A checkpoint warm-start。

## 5. 评价指标与评测流程

实验采用 EER 和 minDCF 作为最终说话人验证指标。评测时首先对所有 utterance 提取 CAMP++ embedding，然后根据 trials 文件计算每个 trial pair 的余弦相似度，并基于相似度分数计算 EER 和 minDCF。为了明确前端恢复模块的作用，评测统一比较 clean、coded 和 restored 三种条件。

评测流程如下：

1. 对 clean eval 语音提取 CAMP++ embedding，得到 clean 条件的性能上限。
2. 对 coded eval 语音提取 CAMP++ embedding，衡量 codec 失真造成的性能下降。
3. 对 coded eval 语音先通过生成器得到 restored，再提取 CAMP++ embedding，衡量前端恢复后的性能变化。
4. 在同一 trials 文件上分别计算三组结果，保证条件之间可比。

需要注意的是，训练过程中的 `valid_sv_cos` 是 clean/restored 同句 embedding cosine 均值，它可以反映 restored 是否逐渐接近 clean，但它并不等价于 trial-level EER/minDCF。因此，最终实验结论仍以外部评测脚本得到的 EER 和 minDCF 为准。

## 6. CN-Celeb 实验结果

### 6.1 clean/coded/restored 基础对比

首先在 CN-Celeb 全量 trials 上比较 clean、coded 和 restored 条件。早期 rec-only 恢复模型的结果如下：

| 条件                    | trials 数量 | EER (%) | minDCF |
| ----------------------- | ----------: | ------: | -----: |
| clean                   |     3484292 |  7.1360 | 0.4112 |
| coded                   |     3484292 |  9.1692 | 0.5123 |
| restored (Experiment A) |     3484292 |  9.0999 | 0.5122 |

从表中可以看出，codec 压缩会明显降低 CAMP++ 后端在 CN-Celeb 上的验证性能：coded 条件相对 clean 条件的 EER 从 7.1360% 上升至 9.1692%，绝对上升 2.0332 个百分点，minDCF 也从 0.4112 上升至 0.5123。这说明 codec 失真确实会造成可观的说话人表征偏移。

Experiment A 的 restored 结果相对 coded 有轻微改善，EER 从 9.1692% 降低到 9.0999%，绝对下降 0.0694 个百分点；minDCF 基本持平。这表明 rec-only 生成器能够学习一定的声学恢复能力，但仅依靠重建损失对说话人验证指标的提升有限。其原因在于 SI-SDR、MRSTFT 和复谱损失主要约束波形与频谱相似性，并不直接优化目标说话人验证判别边界。

### 6.2 CAMP++ teacher 微调结果

在 Experiment A 的基础上加入 CAMP++ 说话人一致性约束后，restored 条件在全量 CN-Celeb trials 上取得如下结果：

| 条件                    | trials 数量 | EER (%) | minDCF |
| ----------------------- | ----------: | ------: | -----: |
| coded                   |     3484292 |  9.1692 | 0.5123 |
| restored (Experiment B) |     3484292 |  8.6838 | 0.4814 |

与 coded 条件相比，Experiment B 将 EER 从 9.1692% 降低到 8.6838%，绝对下降 0.4855 个百分点，相对下降约 5.29%；minDCF 从 0.5123 降低到 0.4814，绝对下降 0.0309，相对下降约 6.03%。该结果说明，在重建损失之外引入与目标后端同域的 CAMP++ teacher，可以有效提升恢复语音的说话人判别性。相比 rec-only 方案，speaker-aware 微调带来的收益更明显，也验证了本文“前端恢复应面向后端说话人任务优化”的基本判断。

### 6.3 B2 训练趋势分析

当前 B2 实验在 Phase 2 中同时使用 CAMP++ embedding loss 和 deep feature loss。已完成的单 codec B2 训练摘要显示，valid speaker cosine 和 valid reconstruction loss 均呈稳定改善趋势：

| epoch | valid_rec_loss | valid_sv_cos |
| ----: | -------------: | -----------: |
|     1 |       -13.4947 |       0.8891 |
|     2 |       -13.4898 |       0.8914 |
|     3 |       -13.4927 |       0.8936 |
|     4 |       -13.4967 |       0.8948 |
|     5 |       -13.5139 |       0.8960 |
|     6 |       -13.5141 |       0.8962 |
|     7 |       -13.5197 |       0.8972 |
|     8 |       -13.5286 |       0.8975 |
|     9 |       -13.5368 |       0.8979 |
|    10 |       -13.5381 |       0.8980 |

从训练趋势看，B2 的 valid_sv_cos 从 0.8891 提升至 0.8980，说明 restored embedding 与 clean embedding 的一致性持续增强；valid_rec_loss 也从 -13.4947 改善至 -13.5381，说明引入 feature-level speaker teacher 并未破坏重建能力。进一步统计 phase2 训练日志中的 speaker loss，可得到如下趋势：

| epoch | spk_raw_mean | spk_feat_raw_mean |
| ----: | -----------: | ----------------: |
|     1 |       0.1107 |            0.0393 |
|     2 |       0.1069 |            0.0390 |
|     3 |       0.1046 |            0.0389 |
|     4 |       0.1033 |            0.0388 |
|     5 |       0.1020 |            0.0387 |
|     6 |       0.1011 |            0.0386 |
|     7 |       0.1007 |            0.0386 |
|     8 |       0.1001 |            0.0385 |
|     9 |       0.1001 |            0.0385 |
|    10 |       0.0996 |            0.0385 |

其中，`spk_raw_mean` 从 0.1107 降至 0.0996，下降约 10.0%；`spk_feat_raw_mean` 从 0.0393 降至 0.0385，下降约 2.1%。这说明 B2 中的 CAMP++ embedding loss 和 deep feature loss 均在逐步收敛。由于 step 级 speaker loss 会受说话人、语音内容和裁剪片段影响产生波动，因此更应关注 epoch 均值趋势，而不是单个 step 的变化。

当前 OPUS+AMR-WB 联合训练实验已完成前 10 轮（仅分析前 10 轮）。根据 `epoch_phase2spk_trend.csv`，前 10 轮的 speaker loss 均值如下：

| epoch | spk_raw_mean | spk_feat_raw_mean |
| ----: | -----------: | ----------------: |
|     1 |       0.1294 |            0.0417 |
|     2 |       0.1258 |            0.0414 |
|     3 |       0.1240 |            0.0412 |
|     4 |       0.1226 |            0.0411 |
|     5 |       0.1212 |            0.0410 |
|     6 |       0.1209 |            0.0410 |
|     7 |       0.1199 |            0.0409 |
|     8 |       0.1201 |            0.0409 |
|     9 |       0.1191 |            0.0408 |
|    10 |       0.1180 |            0.0407 |

其中，`spk_raw_mean` 从 0.1294 降至 0.1180，绝对下降 0.0114（相对下降约 8.82%）；`spk_feat_raw_mean` 从 0.0417 降至 0.0407，绝对下降 0.0009（相对下降约 2.26%）。虽然第 8 轮存在轻微回弹，但整体趋势仍为稳定下降，说明 OPUS+AMR-WB 联合条件下 speaker 相关监督能够持续收敛。

基于 `opus16k+amrwb1585实验B评测结果.json` 的全量 trials 评测，当前 10 轮模型结果如下：

| 条件 | trials 数量 | EER (%) | minDCF |
| --- | ---: | ---: | ---: |
| clean | 3484292 | 7.1417 | 0.4112 |
| coded_opus16 | 3484292 | 9.1692 | 0.5123 |
| restored_opus16 | 3484292 | 9.0341 | 0.4844 |
| coded_amrwb1585 | 3484292 | 10.4703 | 0.5149 |
| restored_amrwb1585 | 3484292 | 9.2706 | 0.4934 |

从 codec 分项对比看：  
- Opus 条件下，restored 相对 coded 的 EER 从 9.1692 降至 9.0341（绝对下降 0.1352，约 1.47% 相对下降），minDCF 从 0.5123 降至 0.4844（绝对下降 0.0279，约 5.45% 相对下降）。  
- AMR-WB 条件下，restored 相对 coded 的 EER 从 10.4703 降至 9.2706（绝对下降 1.1997，约 11.46% 相对下降），minDCF 从 0.5149 降至 0.4934（绝对下降 0.0215，约 4.17% 相对下降）。  

这表明当前 10 轮模型已经在两种 codec 上都取得正向增益，且在 AMR-WB 上的 EER 改善更显著。一个合理解释是：AMR-WB 的退化更重、可恢复空间更大，因此在引入 speaker-aware 约束后更容易观察到显著收益；而 Opus 本身 coded 基线较高，当前阶段提升幅度相对温和。后续可继续比较 `best_generator.pt` 与 `best_generator_sv.pt` 在同一 trials 上的最终差异，以判断更长训练是否能进一步拉近 restored 与 clean 的差距。

## 7. 实验分析与阶段性结论

综合上述结果，可以得到以下阶段性结论。

第一，CN-Celeb 上的 clean/coded 对比证明 codec 失真会显著削弱说话人验证性能。coded 条件相对 clean 条件的 EER 和 minDCF 均明显升高，说明 codec 造成的不只是听感或频谱失真，还会影响后端说话人嵌入的判别结构。

第二，仅使用重建损失的 Experiment A 能够带来轻微改善，但提升幅度有限。这说明单纯追求波形或频谱接近 clean，并不一定能够充分恢复对说话人验证最关键的身份线索。对于本任务而言，恢复目标需要显式引入 speaker-aware 约束。

第三，加入 CAMP++ teacher 的 Experiment B 显著优于 rec-only 恢复。其 restored EER 和 minDCF 均相对 coded 下降，说明使用与目标评测域更匹配的中文说话人后端作为 teacher，可以为生成器提供更有效的任务导向梯度。

第四，B2 的训练趋势表明 feature-level teacher supervision 具有进一步提升潜力。相比只约束最终 embedding，CAMP++ deep feature loss 能够在中间表示层面提供更细粒度的监督，使 restored 语音逐步靠近 clean 的说话人表征。当前 OPUS+AMR-WB 前 10 轮结果也验证了这一点：`spk_raw_mean` 与 `spk_feat_raw_mean` 均下降，且 restored 在 Opus 与 AMR-WB 两种条件下都优于 coded。

第五，当前结果支持继续采用“先重建、再 speaker-aware、最后谨慎加入 GAN”的实验路线。GAN 阶段不应以 adversarial loss 本身为成功标准，而应以是否进一步降低 restored EER/minDCF 为判断依据。如果 Phase 3 不能在 Experiment B/B2 的基础上继续降低 EER 或 minDCF，则说明对抗分支对当前 SV 目标收益不足，需要重新调整判别器结构或损失权重。

## 8. 建议放入中期报告的图表

### 表 1：Codec-only 预实验基线表

建议放在“Codec-only 预实验设置与分析”小节，用于说明不同 codec 对 SV 后端的直接影响。

| Codec  | 码率/配置  | EER (%) | minDCF | 主要退化机理                           |
| ------ | ---------- | ------: | -----: | -------------------------------------- |
| Clean  | 无编码压缩 |  7.1360 | 0.4112 | 无 codec 失真                          |
| Opus   | 16 kbps    |  9.1692 | 0.5123 | 感知编码导致频谱细节和瞬态结构损失     |
| AMR-WB | 15.85 kbps | 10.4703 | 0.5149 | 语音参数编码导致谱包络、时延和相位扰动 |
| AAC    | 32 kbps    |  7.9401 | 0.4428 | 变换域量化与掩蔽模型导致高频细节变化   |
| G.711  | 64 kbps    | 12.9019 | 0.6583 | 压扩量化导致非线性幅度失真和量化噪声   |

### 表 2：CN-Celeb 数据与训练配置表

建议放在“实验设置”小节末尾，用于说明训练数据、codec 条件和训练阶段。

| 实验           | codec 条件        | train/valid 样本数 | 训练阶段 | 主要损失                | 备注                 |
| -------------- | ----------------- | ------------------ | -------- | ----------------------- | -------------------- |
| A              | Opus q25          | 24075 / 2639       | Phase1   | Rec                     | rec-only baseline    |
| B              | Opus q25          | 24075 / 2639       | Phase2   | Rec + CAMP++ emb        | 从 A warm-start      |
| B2             | Opus q25          | 24075 / 2639       | Phase2   | Rec + CAMP++ emb + feat | 当前稳定版本         |
| B2 multi-codec | Opus + AMR-WB q25 | 48474 / 5321       | Phase2   | Rec + CAMP++ emb + feat | 启用 AMR-WB 时移补偿 |

### 表 3：CN-Celeb 全量 trials 的 EER/minDCF 对比

建议放在“实验结果”小节，用于体现 codec 损伤和恢复效果。

| 条件     | 模型                | EER (%) | minDCF |
| -------- | ------------------- | ------: | -----: |
| clean    | 无前端              |  7.1360 | 0.4112 |
| coded    | 无前端              |  9.1692 | 0.5123 |
| restored | Experiment A        |  9.0999 | 0.5122 |
| restored | Experiment B        |  8.6838 | 0.4814 |
| restored | Experiment B (Opus) |  9.0341 | 0.4844 |
| restored | Experiment B (AMR)  |  9.2706 | 0.4934 |
| restored | Experiment B2       |  待填写 | 待填写 |
| restored | Experiment C/Phase3 |  待填写 | 待填写 |

### 图 1：不同 codec coded-only EER 柱状图

建议横轴为 codec，纵轴为 EER (%)，并将 clean 作为第一组柱或水平参考线。该图用于说明不同 codec 对说话人验证性能造成不同程度的直接退化。当前可直接使用完整数据：Clean 7.1360、Opus 9.1692、AMR-WB 10.4703、AAC 7.9401、G.711 12.9019。

### 图 2：不同 codec coded-only minDCF 柱状图

建议横轴为 codec，纵轴为 minDCF，和图 1 使用相同的 codec 顺序。该图用于说明不同 codec 不仅改变 EER，也会改变检测代价。当前可直接使用完整数据：Clean 0.4112、Opus 0.5123、AMR-WB 0.5149、AAC 0.4428、G.711 0.6583。

### 图 3：clean/coded/restored 的 EER 柱状图

建议横轴为条件或模型，纵轴为 EER (%)。可填写的数据为：

```text
clean: 7.1360
coded: 9.1692
restored_A: 9.0999
restored_B: 8.6838
```

该图用于直观展示 codec 失真造成的性能下降，以及 speaker-aware 恢复相对 coded 的改善。

### 图 4：clean/coded/restored 的 minDCF 柱状图

建议横轴与图 3 一致，纵轴为 minDCF。可填写的数据为：

```text
clean: 0.4112
coded: 0.5123
restored_A: 0.5122
restored_B: 0.4814
```

该图用于说明 Experiment B 不仅降低了 EER，也降低了检测代价。

### 图 5：B2 训练阶段 valid_sv_cos 趋势图

建议横轴为 epoch，纵轴为 valid speaker cosine。可使用 `train_summary.json` 中的 `history.valid_sv_cos` 字段绘制。该图用于说明 B2 训练过程中 restored 与 clean 的 speaker embedding 一致性在持续增强。

### 图 6：B2 speaker loss epoch 均值趋势图

建议横轴为 epoch，纵轴分别绘制 `spk_raw_mean` 和 `spk_feat_raw_mean`。该图可以使用现有脚本自动生成：

```bash
python my_methods_GAN/scripts/plot_epoch_spk_trend.py \
  --log_file my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/train.log \
  --phase phase2 \
  --metrics spk_raw,spk_feat_raw \
  --output_csv my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/epoch_phase2_spk_trend.csv \
  --output_png my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_3/epoch_phase2_spk_trend.png
```

对于当前 OPUS+AMR-WB 实验，可替换为：

```bash
python my_methods_GAN/scripts/plot_epoch_spk_trend.py \
  --log_file my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_opus_amrwb/train.log \
  --phase phase2 \
  --metrics spk_raw,spk_feat_raw \
  --output_csv my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_opus_amrwb/epoch_phase2spk_trend.csv \
  --output_png my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25_spk_loss_opus_amrwb/epoch_phase2_spk_trend.png
```

### 图 7：评测流程图

建议画成流程图，内容为：

```text
clean / coded / restored waveform
        ↓
冻结 CAMP++ speaker encoder
        ↓
speaker embedding
        ↓
trial cosine scoring
        ↓
EER / minDCF
```

该图放在评价指标小节，可以帮助说明为什么三种条件具有可比性。

## 9. 后续实验计划

后续工作主要包括三部分。首先，完成 OPUS+AMR-WB B2 实验的 full-trial restored 评测，并比较 `best_generator.pt` 与 `best_generator_sv.pt` 在 EER/minDCF 上的差异。其次，在 B2 最优 checkpoint 基础上进行 Phase 3 GAN 微调，判断 STFT-MRD 与 feature matching 是否能够进一步降低 EER/minDCF。最后，补充不同 codec 条件下的分组评测，例如分别统计 Opus、AMR-WB 与后续 G.711 条件下的恢复收益，从而验证 SC-GridRestore 对多 codec 场景的泛化能力。

在报告撰写中，建议将当前已完成的全量 CN-Celeb 结果作为主要证据，将 B2 multi-codec 结果标注为“正在进行/待补充”。这样既能体现当前方法已经获得阶段性有效性，也能为后续实验留下清晰的推进空间。
