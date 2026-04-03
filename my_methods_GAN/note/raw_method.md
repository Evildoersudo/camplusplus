可以，直接开做。
我给你定一条**主实验路线**，不是再发散，而是为了尽快跑出结果：

## 最终主线：SV-CodecRestoreGAN

这条线不是照搬你上传那篇 TS-URGENet 的三阶段，而是**只保留最适合你课题的 restoration stage 思想**：先把失真 PCM 重采样到 48 kHz，在时频域里做 **CWS-TF-GridNet** 式的带宽扩展与 codec artifact reduction，再下采样回 16 kHz 送入固定 CAM++。你上传的论文里，restoration stage 本身就是为 **bandwidth limitation + codec artifacts + residual packet loss** 设计的，而且这一阶段使用了 GAN 判别器；同时论文也用 WavLM 表征来构造下游相关损失。([arXiv][1])

我要先说明一个现实点：**TS-URGENet 作者目前没有公开训练代码计划**，所以不要再等官方仓库了。更实际的做法是：**用 ESPnet 里的 TF-GridNet 实现当主干骨架，自己补 CWS split/merge 和 speaker-aware loss**。ESPnet 已经有 TF-GridNet 分离器实现；这比你从零重写稳定得多。([GitHub][2])

---

## 一、你需要下载什么

### 必需

1. **ESPnet**
   用它现成的 TF-GridNet 代码当生成器骨架。([GitHub][3])

2. **WavLM 官方实现和预训练模型**
   用来做表征一致性损失。微软官方仓库提供了代码和预训练模型。([GitHub][4])

3. **3D-Speaker**
   用来拿 CAM++ 的训练 recipe、推理脚本和预训练模型。3D-Speaker 官方仓库明确写了支持 CAM++ 训练，并且预训练模型在 ModelScope 可获取。([GitHub][5])

### 可选

4. **open-universe**
   这是后面做 diffusion baseline 的最好备选。官方仓库提供 UNIVERSE/UNIVERSE++ 的开源实现，而且可以直接拉取预训练权重做增强推理。([GitHub][6])

5. **VoiceFixer_main**
   这是给你做一个“恢复基线”的备用仓库。它有完整训练和评测脚本，也有预训练模型，但我不建议它当你的最终主线，因为它是 Mel 域 + vocoder 路线，更适合通用恢复，不一定最利于 speaker cue 保持。仓库本身确实提供了训练和评测入口。([GitHub][7])

建议先执行这几个：

```bash

---

## 二、完整网络结构

### 1. 总体流程

你的主模型定义成：

```text
decoded PCM(16k)
   -> resample to 48k
   -> STFT
   -> CWS split (3 subbands)
   -> TF-GridNet-S x 5
   -> CWS merge
   -> iSTFT
   -> skip add
   -> downsample to 16k
   -> restored PCM
   -> fixed CAM++
```

这样设计的依据很直接：你上传的论文里，restoration stage 就是 **48 kHz 内部处理 + STFT/iSTFT + CWS-TF-GridNet + GAN**，而 TF-GridNet 的核心块本身是“**帧内全频建模 + 子带时间建模 + 跨帧自注意力**”的组合。([arXiv][1])

### 2. 生成器 G

我建议你这样定：

* 输入：失真语音 (y)，16 kHz 单通道
* 内部重采样：16 kHz → 48 kHz
* STFT：`win=32 ms, hop=16 ms, n_fft=1536`
* 复谱拆成实部/虚部通道
* 频带按 CWS 分成 3 个 subbands
* 主干：**5 个 TF-GridNet-S block**
* 输出：预测残差 (\Delta S) 或直接预测增强谱
* 时域端加全局 skip：
  [
  \hat{x} = y + G(y)
  ]

你上传论文给出的 TF-GridNet-S 超参可以直接当第一版：embedding dimension 48、block 数 5、hidden units 100、attention heads 4。([arXiv][1])

### 3. 判别器 D

直接上双判别器：

* **MRD**：多分辨率判别器
* **MBD**：多频带判别器

这是因为你上传的 TS-URGENet 在 restoration stage 就是 **CWS-TF-GridNet + MRD + MBD**，而 filling stage 用的是 MPD + MRD。对你的任务，restoration stage 结构更贴切。([arXiv][1])

### 4. 冻结辅助网络

训练时再接两个冻结分支：

* **WavLM Base+**：做语音表征一致性
* **CAM++**：做说话人嵌入一致性

WavLM 官方有预训练模型；CAM++ 可以直接用 3D-Speaker 的现成模型或你自己现在正在用的权重。([GitHub][4])

---

## 三、损失函数，按这个定

你的总损失不要再做成“普通语音增强”的那套，而是分三层。

### 1. 重建损失

先按 TS-URGENet separation stage 的稳定写法来：

[
L_{\text{rec}} = 2L_{\text{SDR}} + 1.5L_{\text{LSD}} + 70L_{\text{mag}} + 30(L_{\text{real}} + L_{\text{imag}})
]

这个公式就是它论文里常规训练那套波形+频谱联合损失。([arXiv][1])

### 2. GAN 损失

[
L_{\text{gan}} = L_{\text{adv}} + 0.2L_{\text{feat}}
]

这里直接沿用它 restoration stage 的风格，不要自己另起一套太复杂的 GAN 损失。论文里 restoration 的 fine-tuning 也是在 metric-aware 基础上再加 adversarial 和 feature-matching。([arXiv][1])

### 3. 任务导向损失

你自己新加两项：

[
L_{\text{wavlm}} = 1 - \cos(h_{\text{wavlm}}(\hat{x}), h_{\text{wavlm}}(x))
]

[
L_{\text{spk}} = 1 - \cos(e_{\text{cam++}}(\hat{x}), e_{\text{cam++}}(x))
]

最终总损失我建议第一版这样设：

[
L = 10L_{\text{rec}} + L_{\text{adv}} + 0.2L_{\text{feat}} + 1.0L_{\text{wavlm}} + 3.0L_{\text{spk}}
]

这里的前半部分沿用 TS-URGENet restoration/JFT 的尺度思路，后半部分是为了把任务重心明确压到说话人保持上。([arXiv][1])

---

## 四、训练全流程

## Phase 0：先做数据

你现在不要全 codec 一起上。先只做一个最重要的，比如：

* `Opus 16k, 16 kbps`
* 或 `AMR-WB, 12.65 kbps`

训练对这样构造：

```text
clean x
 -> encode(codec, bitrate)
 -> decode
 -> coded y
训练样本：(y, x)
```

要求：

* speaker-level 划分 train / val / test
* 训练裁 2 s
* 验证和测试保留完整语音或至少 4–8 s
* 全部先统一单通道

### 推荐目录

```text
data/
  clean/
    train/spk1/xxx.wav
    val/spk2/xxx.wav
    test/spk3/xxx.wav
  coded_opus16k_16kbps/
    train/spk1/xxx.wav
    val/spk2/xxx.wav
    test/spk3/xxx.wav
  trials/
    test_trials.txt
```

---

## Phase 1：只训生成器

先别开 GAN，先把恢复学稳。

* 模型：只用 G
* 损失：`L_rec`
* 优化器：AdamW
* 初始 lr：`5e-4`
* warmup：`10k`
* 调度：cosine
* batch：单卡先试 4 或 6
* 输入长度：2 s

这里你可以借你上传论文 restoration stage 的配置习惯：它 restoration 训练也是 2 秒片段、AdamW、warmup+cosine。([arXiv][1])

**停止条件**：
验证集上 `LSD`、`PESQ` 稳定下降，且 CAM++ embedding 和 clean 的余弦相似度比 coded-only 明显提高。

---

## Phase 2：加 GAN 微调

这一步打开 MRD + MBD。

* 损失：`10L_rec + L_adv + 0.2L_feat`
* 学习率降到：`1e-4`
* 训练步数：先 10k–20k 微调
* AMP 混合精度打开

**看什么**：
不要只盯听感。重点看：

* `coded -> restored` 后的 CAM++ embedding cosine 是否上升
* 验证集 EER 是否开始下降

---

## Phase 3：加 speaker-aware 约束

冻结：

* WavLM Base+
* CAM++

损失切换到完整版本：

[
L = 10L_{\text{rec}} + L_{\text{adv}} + 0.2L_{\text{feat}} + 1.0L_{\text{wavlm}} + 3.0L_{\text{spk}}
]

这一阶段学习率建议：

* G：`5e-5`
* D：`1e-4`

**模型选择标准**：
不要按 PESQ 最优存模型。
按下面这个顺序选：

1. `EER` 最低
2. `minDCF` 最低
3. 若接近，再参考 `LSD/PESQ`

---

## 五、评测全流程

你要分成两组评测。

## A. 恢复效果评测

建议至少跑：

* PESQ
* ESTOI / STOI
* LSD
* SDR

如果你不想自己东拼西凑，URGENT 官方仓库就有数据准备和评测脚本，适合借来做客观指标管线。([GitHub][8])

## B. 说话人识别评测

这是主评测，必须固定 trial list，跑：

* **EER**
* **minDCF**
* score 分布图
* 同说话人 / 异说话人余弦相似度分布

### 必做对照组

至少四组：

1. **Clean**
2. **Coded-only**
3. **Coded + G(pretrain only)**
4. **Coded + G + GAN + speaker-aware**

如果你有精力，再加：

5. **Coded + open-universe（扩散 baseline）**

open-universe 官方仓库已经支持直接调用预训练权重增强 wav 文件，所以它很适合做“热门 diffusion baseline”。([GitHub][6])

---

## 六、你现在最该下载和准备的预训练模型

### 1. WavLM Base+

必须下。微软官方 README 直接给了预训练模型下载入口。([GitHub][4])

### 2. CAM++ 预训练

如果你现在已有 CAM++，就直接沿用。
如果没有，就用 3D-Speaker 的预训练 CAM++。3D-Speaker README 里明确写了 CAM++ 的 recipe 和预训练模型发布信息。([GitHub][5])


## 八、我对你这条实验线的最终判断

你现在不要再在 FBank 修复上耗了。
从你前面的反馈看，**特征域修复已经证明不稳**；那就直接把课题改成：

**“面向说话人识别的 codec artifact reduction 与 bandwidth compensation”**

主模型就是我上面这条：

**CWS-TF-GridNet generator + MRD/MBD discriminator + WavLM/CAM++ 双约束**

这条线有三个优点：

* 和你上传那篇论文的 restoration stage 方向一致，理论上站得住。([arXiv][1])
* 不依赖 TS-URGENet 官方代码，因为它没开源；你可以用 ESPnet TF-GridNet + 自己补模块来落地。([GitHub][2])
* 目标函数直接对齐 speaker verification，而不是只追求“听起来更好”。这才符合你的毕设主线。


[1]: https://arxiv.org/html/2505.18533v1 "https://arxiv.org/html/2505.18533v1"
[2]: https://github.com/Xiaobin-Rong/gtcrn/issues/91 "https://github.com/Xiaobin-Rong/gtcrn/issues/91"
[3]: https://github.com/espnet/espnet "https://github.com/espnet/espnet"
[4]: https://github.com/microsoft/unilm/blob/master/wavlm/README.md "unilm/wavlm/README.md at master · microsoft/unilm · GitHub"
[5]: https://github.com/modelscope/3D-Speaker "GitHub - modelscope/3D-Speaker: A Repository for Single- and Multi-modal Speaker Verification, Speaker Recognition and Speaker Diarization · GitHub"
[6]: https://github.com/line/open-universe "https://github.com/line/open-universe"
[7]: https://github.com/haoheliu/voicefixer_main "https://github.com/haoheliu/voicefixer_main"
[8]: https://github.com/urgent-challenge/urgent2025_challenge "https://github.com/urgent-challenge/urgent2025_challenge"
