我看完你补的 `engine.py`、训练脚本和 `train_summary.json` 之后，结论比之前更明确了：

**你现在的 phase3 基本没有真正把 WavLM/CAM++ 约束反向传到生成器；而且你用来选 best model 的 valid_loss 也根本没有评估 phase2/phase3 新增目标。**
所以目前这版结果，**不能说明“speaker-aware 方案无效”**，只能说明**训练代码里存在关键实现问题**。   

我按“影响从大到小”给你拆。

---

## 一、最关键的问题：phase3 的 WavLM 和 CAM++ loss 现在大概率没有梯度

你在 `engine.py` 里 phase3 的逻辑是：

* `wavlm_clean = wavlm(clean)`
* `wavlm_rest = wavlm(restored)`
* `emb_clean = camp(clean_fbank)`
* `emb_rest = camp(rest_fbank)`
* 然后把这两个 cosine loss 加到 `g_loss` 里。

但是你两个 wrapper 都把 `forward` 整个包在了 `@torch.no_grad()` 下面：

* `wavlm_wrapper.py` 的 `forward` 是 `@torch.no_grad()`，而且最后还直接 `rep.mean(dim=1)`。
* `campplus_wrapper.py` 的 `forward` 也是 `@torch.no_grad()`。

这意味着：

### 1）WavLM loss 现在对 `restored` 不回传梯度

因为 `wavlm(restored)` 整个前向都在 no_grad 里，phase3 的 `wavlm_term` 只是一个“数值惩罚项”，**不会推动 generator 参数更新**。
也就是说，你现在 phase3 里所谓的 WavLM loss，**训练上几乎等于没加**。 

### 2）CAM++ loss 也同样不回传

`camp(rest_fbank)` 也在 no_grad 里，所以 `spk_term` 对 generator 也没有梯度。 

### 3）而且 CAM++ 这条链路还有第二层问题

你现在的 speaker loss 前面还经过了 `_fbank_batch()`，里面用的是 `torchaudio.compliance.kaldi.fbank`。这条链路本身就不是你应该依赖的可微前端写法。也就是说，**就算你把 CAMPPlus 的 no_grad 去掉，当前这条 FBank 管线也不适合作为稳定的反传路径**。

### 这个问题的直接结论

你现在 phase3 真正能训练 generator 的，基本还是：

* `10 * rec_term`
* `adv_g`
* `0.2 * feat_g`

而不是你以为的 “rec + GAN + wavlm + spk”。

---

## 二、第二个关键问题：你的 valid_loss 只在看重建项，根本没在验证 phase2/phase3 目标

你当前验证阶段只做了这件事：

```python
rec_dict = loss_rec(restored, clean)
valid_loss += float((10.0 * rec_dict["total"]).detach().cpu())
```

也就是**整个 valid_loss 只看 `10 * loss_rec`**。
phase2 的 `adv_g / feat_g` 没进验证；phase3 的 `wavlm_term / spk_term` 也没进验证。 

这会带来两个后果：

### 1）train_loss 和 valid_loss 根本不是一个目标

phase2/phase3 训练时优化的是更复杂的混合目标，但验证时还是只看 rec。
所以你现在日志里“train_loss 降，valid_loss 升”并不一定是传统意义上的过拟合，而是**训练目标和验证目标不一致**。 

### 2）best_valid 很可能挑错模型

你的 `best_valid=42.69` 出现在 epoch 6，也就是刚进 phase2 时；后面 phase2/phase3 全部变差。这个现象并不能直接说明“GAN 和 task loss 没用”，更可能说明：

* 一进入 phase2，GAN 开始扰动重建目标
* 但你选 best 的标准还是 rec-only
* 所以最优点自然会偏向“还没被 GAN 明显影响”的早期模型。 

### 你必须改成两套验证指标

至少同时保存：

* `valid_rec_loss`
* `valid_sv_eer` 或 `valid_sv_cos`

真正做毕设，最终 `best_model` 应该优先按 **EER / minDCF** 选，而不是只按恢复 loss 选。

---

## 三、第三个问题：你现在的重建损失“系数抄对了，公式没抄对”

你的 `loss_rec` 是：

[
2L_{sdr} + 1.5L_{lsd} + 70L_{mag} + 30(L_{real}+L_{imag})
]

这个系数形式确实和你参考论文的 (L_1) 很像。 

但是论文里的 `Lmag/Lreal/Limag` 不是你现在这种“直接对原始 STFT 实部、虚部、幅度做普通误差”。论文明确写的是：

* (L_{mag} = \mathrm{MSE}(|S|^{0.3}, |\hat S|^{0.3}))
* (L_{real} = \mathrm{MSE}(S_r / |S|^{0.7}, \hat S_r / |\hat S|^{0.7}))
* (L_{imag} = \mathrm{MSE}(S_i / |S|^{0.7}, \hat S_i / |\hat S|^{0.7}))

也就是带**幂次压缩和归一化**的版本。

这件事非常重要，因为你现在沿用了 70/30 这种大权重，却没有对应的归一化形式，结果就是：

* 重建项数值尺度很可能失真
* GAN 和辅助项更容易被淹没
* 训练重心和论文不一致

### 我的建议

不要继续硬贴论文公式了，先用更稳的版本：

[
L_{rec}=L_{SI\text{-}SDR}+L_{MRSTFT}
]

也就是：

* 一个时域 SI-SDR
* 一个多分辨率 STFT loss

这比你当前这版更稳，更容易先把 baseline 跑出来。

---

## 四、第四个问题：phase3 学习率衰减太快，而且所有 phase 共用同一套 scheduler

你现在的 G/D 学习率是用**全训练周期**统一 warmup-cosine 算的。

所以到 phase3 时，G 的学习率已经掉得很低了。你的日志里：

* epoch 11: `lr_g ≈ 1.53e-4`
* epoch 15: `lr_g ≈ 5e-6`

也就是说，你 phase3 只有 5 个 epoch，本来就短；再加上学习率快速降到底，等于“speaker-aware 微调还没真正开始就快结束了”。

### 我建议你改成 phase-local scheduler

也就是：

* phase1 单独 schedule
* phase2 单独 schedule
* phase3 单独 schedule

尤其 phase3，建议用固定小学习率或慢衰减，例如：

* `lr_g = 5e-5`
* `lr_d = 1e-5 ~ 2e-5`

不要再沿用全局 cosine 了。

---

## 五、第五个问题：你的 phase2/phase3 是“加法太猛”，不是渐进式微调

你当前 phase2 一上来就：

* MRD
* MBD
* adversarial
* feature matching

一起开。

这在数据只抽样一部分、epoch 又很少的情况下，非常容易把 generator 拉偏。
而你的日志也正符合这个模式：**最佳点恰好出现在 phase2 刚开始，之后验证迅速恶化。**

### 我建议你把 phase2 改成更保守的形式

先只做：

* `G + rec`
* 然后 `G + rec + 单个判别器(MRD)`

先别同时用 MRD 和 MBD。
你现在不是在追论文的完整恢复音质，而是在追**对说话人识别有效**。对这个目标，过强的 GAN 经常是副作用大于收益。

---

## 六、第六个问题：WavLM 用法也不对，你现在是“整句均值池化”，太粗

你现在 WavLM wrapper 最后返回的是：

```python
rep = self.model.extract_features(x)[0]  # [B, T, D]
return rep.mean(dim=1)
```

也就是把整句时序特征直接平均成一个向量。

但你参考论文里 WavLM loss 的写法，核心是**continuous distillation**，它对 clean / enhanced 的时序表征逐点比较，而不是先把整句全平均再做 cosine。论文也明确写了用的是 **WavLM Base+**。

### 所以你这里至少要改两件事

1. 从 `WavLM-Base.pt` 换成 **Base+**
   你现在脚本默认还是 `WavLM-Base.pt`。

2. loss 改成帧级
   不要再 `mean(dim=1)`，而是保留 `[B, T, D]`，对齐后逐帧算相似度。

---

## 七、第七个问题：你现在没有真正的 SV 验证闭环

你整个训练脚本里没有任何验证阶段的 speaker verification 评测。
也就是说，你现在实际上只是在做“恢复网络训练”，然后希望它顺便对 CAM++ 有帮助。

但你的课题本质是：

**面向说话人识别的 codec restoration**

所以你必须在训练期就建立一个轻量的 SV 验证闭环。
否则你根本不知道：

* 是恢复变好了但识别变差
* 还是恢复没那么好但识别更稳
* 还是 GAN 只改善听感不改善 speaker cue

### 最小可行方案

你搞一个小的 `valid_trials.txt`，每个 epoch 或每 2 个 epoch 跑一次：

* coded-only
* restored

分别过固定 CAM++，算：

* cosine score
* EER
* minDCF（有余力再加）

然后保存：

* `best_rec.pt`
* `best_eer.pt`

否则你现在所有“好不好”的判断都不够可靠。

---

# 我建议你现在怎么改

## 第一优先级：今天就改

### 1）先让 WavLM loss 真正可训练

把 `wavlm_wrapper.py` 的 `@torch.no_grad()` 去掉。
冻结参数靠：

```python
for p in model.parameters():
    p.requires_grad = False
```

就够了。
然后在 `engine.py` 里只对 clean 分支用 `torch.no_grad()`，restored 分支不要关梯度。

### 2）暂时删除 CAM++ loss

不是永久删除，而是**先不用它做训练 loss**。
因为你当前这条 `fbank -> campplus` 路线不适合作为稳定的可微约束。
CAM++ 先只用来做**验证指标**。

### 3）把 valid 改成双指标

保存：

* `valid_rec`
* `valid_eer`

不要再只看 `valid_loss = 10 * rec_total` 了。

---

## 第二优先级：这一轮重训前改

### 4）重写 loss_rec

先不要再用当前这版 `70/30` 配方。
直接换成：

* `SI-SDR`
* `MR-STFT`

简单稳。

### 5）WavLM 改成 Base+

你参考论文就是 Base+，你当前脚本还是 Base。 

### 6）WavLM 改成帧级 loss

不要整句 mean pool。

---

## 第三优先级：训练策略改法

### 7）phase1 拉长

你现在每个 phase 只有 5 epoch，太短。
建议：

* phase1: 20~30
* phase2: 8~10
* phase3: 0 或先不开

### 8）phase2 先只加一个判别器

先只开 MRD，别 MRD+MBD 一起上。

### 9）phase-local 学习率

phase2 / phase3 学习率单独重置，不要接着 phase1 的 global cosine 直接往下掉。

---

# 我给你的最稳实验顺序

## 实验 A：Rec-only baseline

* 去掉 GAN
* 去掉 WavLM
* 去掉 CAM++
* 只训 `SI-SDR + MRSTFT`

目的是先确认：**纯 PCM restoration 能不能稳定下降、不会把 SV 搞得更差。**

## 实验 B：Rec + WavLM(frame-level)

* 加 WavLM Base+
* 帧级 loss
* 不加 CAM++
* 不加 GAN

目的是确认：**speaker-aware 的表征约束是否真的能带来 SV 增益。**

## 实验 C：Rec + WavLM + 单 MRD

* 只加一个判别器
* D 的 lr 比现在更小

目的是确认：**GAN 到底是帮忙还是捣乱。**

## 实验 D：最后再看 CAM++

只有在你把 differentiable frontend 处理好之后，再考虑把 CAM++ 作为训练 loss 接回来。
在那之前，它更适合做验证器，不适合做 loss network。

---

# 你这轮最应该记住的一句话

**现在不是模型思想不行，而是训练实现把最关键的 task loss“做成了常数项”。**

也就是说，你目前 phase3 看起来“加了 speaker-aware”，但实际上对 generator 几乎没起作用；与此同时，best model 还在按 rec-only valid_loss 选，所以整个实验逻辑是错位的。  

你下一条把 **`SVCodecPairDataset`** 和 **manifest 字段格式** 发我，我直接按上面的方案给你重写一版训练主循环，至少包括：

* phase-local scheduler
* WavLM 可训练帧级 loss
* 去掉 CAM++ 训练 loss
* valid EER 验证钩子
* 更稳的 `loss_rec` 实现。
