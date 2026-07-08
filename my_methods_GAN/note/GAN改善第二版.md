我看完你现在这版代码和曲线后，判断比之前更清楚了：

**问题已经不主要在“模型没梯度”或“teacher 不可微”，而是在训练目标设计和数据裁剪方式上。**
你现在的训练里，generator 同时吃了 `rec_total + embedding cosine + AM-Softmax`，而验证却主要看 `valid_rec` 和同句 clean/restored 的 `valid_sv_cos`，这两者都不等价于最终的 trial-level EER/minDCF，所以很容易出现“训练在动，但 EER 不涨”的情况。 这也符合任务导向 SV 文献的结论：增强目标和说话人目标之间会有 competing gradients，联合训练若没有阶段化或异步优化，很容易彼此拉扯。([ISCA Archive][1])

我先给你最关键的结论：

## 先别继续堆现在这版 B

你现在这版 B 已经说明两件事：

* 只用最终 embedding cosine 的 teacher 约束，不够强。
* 再叠一个 AM-Softmax speaker 分类头，反而更可能把 generator 拉向“分类边界”，而不是“恢复 clean speaker cues”。你现在的 `AMSoftmaxClassifier` 就是直接对 `emb_rest` 做分类交叉熵，没有 clean 对照分支。 

这和 Deep Feature Loss 那条线的经验是一致的：比起只监督最终 embedding，**用预训练 speaker 网络的中间层特征做 deep feature loss 往往更有效**，因为监督更细，梯度也更丰富。([arXiv][2])

---

## 你现在最该优先排查的一件事：配对裁剪是否错位

你的 `SVCodecPairDataset.__getitem__()` 里，`clean` 和 `coded` 是**分别**调用 `crop_or_pad()` 的。

如果 `crop_or_pad()` 内部是随机起点裁剪，那就会出现这种情况：

* clean 裁一段
* coded 裁另一段

虽然它们来自同一条语音，但时间位置不同。
这会直接破坏：

* `SI-SDR`
* `MRSTFT`
* `complex`
* 甚至 teacher 的 same-utterance speaker consistency

这类错位对 restoration 任务是致命的，尤其是你现在又在做 **波形级重建 + speaker teacher**。

### 我建议你立刻改成“同起点配对裁剪”

不要再让 clean/coded 各自 random crop。
应该先取两者公共长度，再生成**同一个 start**，然后同时裁：

```python
# 伪代码
L = min(clean.numel(), coded.numel())
clean = clean[:L]
coded = coded[:L]

if self.segment_len > 0:
    if L >= self.segment_len:
        if self.random_crop:
            start = self._rng.randint(0, L - self.segment_len)
        else:
            start = 0
        clean = clean[start:start+self.segment_len]
        coded = coded[start:start+self.segment_len]
    else:
        clean = pad_to_len(clean, self.segment_len)
        coded = pad_to_len(coded, self.segment_len)
```

这一步我会放在**最高优先级**。
如果这一步不改，后面很多 loss 都是在学“错位配对”。

---

## 第二个关键改动：先关掉 AM-Softmax

你现在训练命令里开了：

* `--use_campplus_train_loss`
* `--use_spk_amsoftmax`

也就是 generator 同时吃：

* `L_rec`
* `L_emb`
* `L_cls`

而 `L_cls` 来自 AM-Softmax 分类头。 

这一步我建议你**先关掉**：

```bash
--no_use_spk_amsoftmax
```

原因很简单：

* 你现在最想恢复的是 **speaker verification utility**
* 不是让 restored embedding 更容易做 speaker classification
* 这两个目标有关，但不等价

Wu 2021 已经明确指出，多目标联合里 speaker classification loss 和 enhancement loss 会出现 competing gradients。([ISCA Archive][1])
你现在这版 AM-Softmax 更适合放到后面做增强版，不适合当前这一步的最小可控实验。

---

## 第三个关键改动：把“最终 embedding loss”升级成“deep feature loss”

这是我认为你现在最值得做的核心升级。

你当前的 teacher loss 是：

[
L_{emb}=1-\cos(E(\hat{x}),E(x))
]

它只有一句话级别一个向量，监督太粗。
更好的做法是：

[
L = L_{rec} + \lambda_1 L_{emb} + \lambda_2 L_{feat}
]

其中：

* `L_emb`：保留最终 embedding cosine
* `L_feat`：取 CAMPPlus 中间层 hidden activation，做 L1 或 cosine loss

这就是 Deep Feature Loss 的思路：在预训练 speaker network 的隐藏激活空间里约束增强网络，而不是只盯最终输出。相关工作在 SV 上报告过一致收益。([arXiv][2])

### 你现在这版代码怎么落地

你现在的 `FrozenCampPlus` 只返回最终 embedding。
下一步建议这样改：

1. 在 `campplus_wrapper.py` 里加 forward hook，抓 1 到 2 个中间层输出
2. 返回：

   * `emb`
   * `feat_mid`
   * `feat_prepool`（任选其一或两个）

伪代码：

```python
class FrozenCampPlus(nn.Module):
    def __init__(...):
        ...
        self._feat = {}
        self.model.some_block.register_forward_hook(self._save("mid"))
        self.model.xvector.register_forward_hook(self._save("prepool"))

    def _save(self, name):
        def hook(m, inp, out):
            self._feat[name] = out
        return hook

    def forward(self, feat_bt80, return_feats=False):
        self._feat = {}
        emb = F.normalize(self.model(feat_bt80), dim=-1)
        if return_feats:
            return emb, self._feat
        return emb
```

然后训练里用：

```python
with torch.no_grad():
    emb_clean, feat_clean = camp(clean_fbank, return_feats=True)

emb_rest, feat_rest = camp(rest_fbank, return_feats=True)

loss_emb = 1 - cosine(emb_rest, emb_clean)
loss_feat = L1(feat_rest["mid"], feat_clean["mid"])
```

### 权重建议

先从这个最小版开始：

[
L = L_{rec} + 5.0L_{emb} + 0.5L_{feat}
]

这里不是因为 5 和 0.5 有神奇性，而是因为你现在 raw `L_emb` 大约在 0.2 左右，`L_feat` 往往量级更大，所以 embedding 用大一点、feature 用小一点更稳。

---

## 第四个关键改动：validation 不要再只看 `valid_sv_cos`

你现在的 `valid_sv_cos` 是 clean/restored **同一句** 的 cosine 相似度均值。
这个指标只能说明：

* restored 有没有更接近 clean embedding

但它**不能代表**：

* same-speaker / different-speaker 的区分能力

也就是它和最终 EER 不是同一个目标。

### 我建议你加一个小 dev trial 集

别每个 epoch 都跑完整 `3,484,292` 条 trial，太慢。
你做一个固定的小 dev list，比如：

* 5k target
* 20k nontarget

每个 epoch 跑一次真实 EER。
然后保存：

* `best_generator_rec.pt`
* `best_generator_dev_eer.pt`

否则你很容易又陷入“训练 loss 变好，但 EER 不动”的循环。

---

## 第五个关键改动：MRSTFT 的 spectral convergence 改成 per-sample

你现在的 `loss_mrstft()` 里这句：

```python
spectral_convergence = torch.linalg.norm(t_mag - p_mag) / torch.linalg.norm(t_mag)
```

是对整个 batch 直接做一个 norm 比值。

这会让：

* 高能量样本主导 batch
* 不同 batch 间 scale 波动更大
* 你图里那种 `rec_total` 大起大落更明显

### 建议改成 per-sample 平均

```python
diff = (t_mag - p_mag).reshape(B, -1)
ref  = t_mag.reshape(B, -1)
spectral_convergence = (torch.linalg.norm(diff, dim=1) / torch.linalg.norm(ref, dim=1).clamp_min(eps)).mean()
```

这是一个很小的改动，但能让 `rec_total` 稳定不少。

---

## 第六个关键改动：phase2 的裁剪长度改到 4~5 秒

你现在 phase2 还是 3 秒。
对 restoration 足够，对 speaker teacher 不够稳。

我建议：

```bash
--segment_seconds 4.0
--phase2_segment_seconds 4.0
```

因为 speaker loss 本身比 rec loss 更噪，3 秒切片会让 `spk_raw` 看起来更抖。

---

## 我给你的下一轮最推荐实验：B2

这一轮不要再加 GAN，不要 AM-Softmax，只做最小升级版：

### 训练命令建议

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_q25.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_q25.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_expB2_camp_feat_q25 \
  --init_generator_ckpt my_methods_GAN/exp/sv_codec_restore/run_expA_rec_only_q25/best_generator.pt \
  --phase1_epochs 0 \
  --phase2_epochs 6 \
  --phase3_epochs 0 \
  --batch_size 24 \
  --num_workers 8 \
  --segment_seconds 4.0 \
  --phase2_segment_seconds 4.0 \
  --train_sample_fraction 1.0 \
  --valid_sample_fraction 1.0 \
  --lr_g_max 5e-5 \
  --lr_g_min 1e-5 \
  --warmup_steps_g 100 \
  --si_sdr_weight 1.0 \
  --mrstft_weight 0.5 \
  --complex_weight 0.0 \
  --use_campplus_train_loss \
  --campplus_frontend diff_mel \
  --spk_loss_weight 5 \
  --no_use_spk_amsoftmax \
  --phase3_no_gan \
  --phase3_no_wavlm \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_cn_cnceleb_16k/campplus_cnceleb.bin \
  --device cuda
```

然后你在代码里把 teacher loss改成：

[
L = L_{rec} + 5L_{emb} + 0.5L_{feat}
]

而不是当前这版 `L_rec + embedding cosine + AM-Softmax`。

---

## 为什么我推荐这条路

因为它同时满足三点：

1. 和你现在代码最接近，改动最小
2. 符合 Deep Feature Loss for SV 的文献思路：隐藏层监督比只盯最终 embedding 更有效。([arXiv][2])
3. 也符合 task-oriented joint optimization 的经验：先做简单、可解释、低冲突的多目标，再考虑更复杂的 GAN。([ISCA Archive][1])

---

## 最后一句最重要的话

你现在这版不是“完全没戏”，而是**监督粒度太粗 + 裁剪可能错位 + 选模指标不对齐**。
先把这三件事修掉，再谈 GAN 或多 codec，成功率会高很多。

你把 `crop_or_pad` 的实现和 `CAMPPlus` 里你想 hook 的层名发我，我可以直接给你写出 **对齐裁剪版 dataset** 和 **B2 的 teacher feature loss 代码**。

[1]: https://www.isca-archive.org/interspeech_2021/wu21c_interspeech.pdf?utm_source=chatgpt.com "Joint Feature Enhancement and Speaker Recognition with ..."
[2]: https://arxiv.org/abs/1910.11905?utm_source=chatgpt.com "Feature Enhancement with Deep Feature Losses for Speaker Verification"
