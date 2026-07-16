先说结论：**你现在的主要问题不是“模型太小”，而是“结构复现不对 + 损失设计不对 + 训练阶段切换太猛”**。
从日志看，最好的 `valid_loss=42.69` 出现在 **epoch 6**，也就是刚进入 phase2 时；后面 phase2/phase3 训练集 loss 还在降，但验证集却一路恶化到 170 左右，这更像是**GAN/任务损失把模型往错误方向推走了**，不是单纯没收敛。

我按“最影响结果”的顺序给你拆。

## 1. 你现在的生成器和论文里的 restoration stage 不是一回事

这是我觉得**最大的问题**。

你现在的 `GridNetBlock` 里，时间建模前先对整个频率维做了 `mean(dim=-1)`，也就是把所有频带先平均，再送进 GRU 和 attention，最后把同一个时间更新广播回所有频率 bin。这样做的结果是：**时间建模阶段根本看不到细粒度频带差异**，而 codec artifact、带宽缺失恰恰就是最依赖频带局部结构的问题。你的实现更像“频域卷积 + 全频平均后的时序建模”，不是 TF-GridNet 那种真正的时频联合建模。代码就在 `xt = x.mean(dim=-1)` 这一句。

另外，论文里的 restoration stage 明确写的是：**三个子带先切出来，然后沿 channel 维拼接作为输入；输出再按 channel 切回三个子带，最后沿 frequency 合并**。而你现在是把三个子带**分别独立跑同一套 block**，最后再拼回来。这样就没有“跨子带交互”，也就偏离了论文里 CWS-TF-GridNet 的关键思想。你现在的 `_cws_split` / `_cws_merge` 加 for-loop 正是这个问题。

再加一个更实际的点：论文的 restoration stage 是针对 **48 kHz 数据**设计的，而你的最终下游是 **16 kHz CAM++**。论文里确实是先升到 48 kHz 再做 restoration，但那是因为 challenge 本身包含带宽恢复和多采样率任务；你这里如果输入和目标本来都是 16 kHz，还强行 `16k -> 48k -> 16k`，很容易让模型把精力花在“幻觉式高频补偿”上，而这些高频最后又被下采样扔掉，对 speaker verification 的帮助未必大。论文本身也写明了 restoration stage 使用的是 48 kHz 训练数据。

### 这里我建议你直接改成两步

第一步，**先去掉 48 kHz 内部流程，全部在 16 kHz 做**。
先证明“16 kHz codec restoration 对 CAM++ 有帮助”，再考虑 48 kHz 版本。

第二步，**把 CWS 改成真正的 channel-wise concat**。
也就是：

* 输入复谱 `[B, 2, T, F]`
* 按 F 切 3 段
* 在 channel 维拼成 `[B, 2*3, T, F/3]`
* `in_proj` 改成 `Conv2d(6, emb_dim, 1)`
* 网络输出 `[B, 6, T, F/3]`
* 再按 channel 切成 3 段，沿 F 拼回去

而不是现在这种“3 段各跑一遍”。

## 2. 你的 loss 权重是“抄了系数，没抄公式”

这是第二个大坑。

TS-URGENet 论文里的重建损失虽然也是
`2*SDR + 1.5*LSD + 70*mag + 30*(real+imag)`
但它的 `Lmag/Lreal/Limag` 不是你现在这种**原始 STFT 直接做 L1**。论文里是带幂次和归一化的 MSE，形式是 `|S|^0.3` 以及 `real / |S|^0.7`、`imag / |S|^0.7` 这种设计，目的是平衡大能量和小能量频点。你当前代码里却是对 `p.abs(), p.real, p.imag` 直接做 L1，然后还保留了 70 和 30 这种大系数。这样一来，重建损失的数值尺度会和论文完全不是一个量级，GAN loss、feature matching、speaker loss 很容易被淹没。`loss_rec` 这段代码就是这个问题。

这也能解释你的日志：
`train_loss` 从 286 降到 222，看起来在降；但 `valid_loss` 在 phase2/3 爆掉，说明模型在拼命优化某个过强的重建项，却没有朝“泛化的 restoration + speaker preservation”走。

### 我建议你这里二选一

**更稳的选法：** 先别硬复现论文那套式子，改成更稳的 restoration 组合：

[
L_{rec} = 1.0 \cdot L_{SI\text{-}SDR} + 1.0 \cdot L_{MRSTFT}
]

其中 `MRSTFT` 用 3 组分辨率，例如：

* `(n_fft, hop, win) = (256, 64, 256)`
* `(512, 128, 512)`
* `(1024, 256, 1024)`

如果你想保留复谱项，可以额外加一个很小的 complex L1，比如 0.5。

**想贴近论文的选法：** 按论文把 `Lmag/Lreal/Limag` 真的改成幂谱/归一化形式，不要继续用当前 raw L1 版。

我的建议是先走第一种，因为更稳。

## 3. WavLM 和 CAM++ 的任务损失目前太“粗”

你现在的 `WavLMFeatureExtractor` 直接把 `extract_features(x)[0]` 做了时间平均，最后输出整句一个向量。也就是说，你当前 WavLM loss 其实是**整句全局均值表征约束**，不是论文里那种基于时序表征的 continuous distillation。论文写得很明确，它的 WavLM loss 是沿时间和特征维逐项构造的，而不是一句话先 mean pool 再比。你现在的做法太粗，会把局部时频恢复错误“平均掉”。

CAM++ 这边还有两个问题：

第一，你现在训练片段只有 **2 秒**。对 restoration 足够，但对 speaker embedding loss 来说偏短，特别是 codec 失真下 2 秒 embedding 方差会更大。
第二，你的 `campplus_wrapper.py` 用了 `strict=False` 加载权重，这在调试期方便，但正式训练前必须检查一次 `missing_keys / unexpected_keys`。不然很可能某些层根本没正确加载，你却一直没发现。

### 这里我的建议

WavLM：

* phase3 不要用当前 mean-pooled WavLM loss 直接上强权重
* 改成**帧级特征 loss**，至少不要先 `mean(dim=1)`
* 再不行就先把 WavLM loss 去掉，只保留 CAM++ speaker loss

CAM++：

* 先确认 checkpoint 加载是否完整，别继续 `strict=False` 默默吞错误
* phase3 的 speaker loss 改成 **4 秒片段**，至少训练时单独把 phase3 的 `segment_seconds` 提高到 4.0
* speaker loss 权重要非常小地开，比如 `0.05 ~ 0.2`

## 4. 你的判别器太“玩具版”，而且现在大概率在压着生成器打

当前 `MultiBandDiscriminator` 的 band split 其实只是平滑 + 残差得到 low/mid/high，属于很轻的 proxy，而不是更严格的时频多带判别。对于 codec artifact 这种本来就细粒度的失真，这种判别器很可能既不够准，又足够不稳定，最后把 G 往奇怪方向推。

从你的日志看，**epoch 6 是最优，之后 phase2 一开就崩**，这很符合“D 介入后把 G 带偏”的特征。

### 这里我建议

现在不要急着同时用 MRD + MBD。
直接做下面这个顺序：

1. **只训 G，纯重建**
2. **G + speaker loss**
3. 最后才上 **单个 MRD**
4. 确定有增益后，再加 MBD

也就是说，GAN 不是你当前的第一优先级。
你现在首先要证明：**不用 GAN，只靠 restoration + speaker-aware loss 就能让 EER 下降。**

## 5. 训练策略也不对：你现在不是在“实验”，而是在“冒烟测试”

你命令里 `train_sample_fraction=0.25`、`valid_sample_fraction=0.25`，而且每个 phase 只有 5 epoch。这个更像 smoke test，不像正式训练。
在这种设置下，验证波动本来就会很大；再叠加 phase2/3 的新损失，结果很容易被噪声主导。

另外，论文里的 restoration 和 JFT 本来就是**大量 step**训练，不是 5 epoch 这种量级；它也用了 warmup + cosine，但前提是总步数足够大。你现在 phase3 只有 5 个 epoch，学习率从 `1.5e-4` 很快掉到 `5e-6`，等于 speaker-aware 微调刚开始就快结束了。

### 我建议你改成这个训练节奏

**阶段 A：Rec only**

* full train set，至少先用 `0.5`，最好 `1.0`
* `phase1_epochs = 20~30`
* 不开 GAN，不开 WavLM，不开 CAM++
* 只看 `SI-SDR / MRSTFT / LSD`

**阶段 B：+ CAM++ speaker loss**

* `phase2_epochs = 8~10`
* `segment_seconds = 4.0`
* 只加 CAM++，先别加 WavLM
* 学习率用固定小值或很慢的 cosine，比如 `5e-5`

**阶段 C：+ GAN**

* `phase3_epochs = 5~8`
* 只加 MRD，先不加 MBD
* `lr_d_max` 降到 `2e-5` 或 `5e-5`
* 判别器每 2 个 G step 更新 1 次

你现在的相反做法是：过早让 GAN 和辅助损失一起上，而且每阶段太短。

## 6. 你应该先做的不是“继续训练”，而是 4 个最小对照实验

你现在最需要的是定位，不是盲目改大网络。
我建议你马上跑下面 4 组，全部只针对一个 codec/码率，比如 q25：

### Exp-1：16k、Rec only

* 去掉 48k internal
* 去掉 GAN
* 去掉 WavLM/CAM++
* 只训 `SI-SDR + MRSTFT`

目标：确认基本 restoration 是否稳定。

### Exp-2：16k、Rec + CAM++

* 在 Exp-1 基础上加一个很小的 speaker loss
* phase2 用 4 秒片段
* 不加 WavLM

目标：看 EER 是否开始优于 Exp-1。

### Exp-3：16k、Rec + WavLM(frame-level)

* 不加 CAM++
* 把 WavLM loss 改成帧级，不做时间平均

目标：看 WavLM loss 是否真有帮助。

### Exp-4：16k、Rec + CAM++ + MRD

* 只加一个判别器
* adv weight 很小

目标：看 GAN 是否真的带来 EER 提升，而不是只改变 valid loss。

如果 Exp-2 就有效，你论文就已经能写了。
如果 Exp-4 反而更差，你就知道 GAN 不是必须项。

## 7. 我给你的“最优先修改清单”

按收益排序，我建议你这么改：

### 第一优先级：必须改

1. **去掉 48k 内部处理，全部先做 16k**
2. **把 CWS 改成真正 channel-wise concat**
3. **去掉 `x.mean(dim=-1)` 这种全频平均时间建模**
4. **重写 `loss_rec`，别再用当前 raw STFT L1 + 70/30**
5. **检查 CAM++ 是否完整加载，不要继续静默 `strict=False`**

### 第二优先级：很建议改

6. **phase3 用 4 秒而不是 2 秒**
7. **WavLM 从 Base 换成 Base+**
8. **WavLM loss 改成帧级，不做全局平均**
9. **GAN 先只保留一个 MRD**
10. **phase1 拉长到至少 20 epoch**

### 第三优先级：后面再做

11. 再试 MBD
12. 再试 48k 版本
13. 再做多 codec 混训

## 8. 一版我建议你直接改的训练命令

先跑一个真正能定位问题的版本：

```bash
python -u my_methods_GAN/scripts/train_sv_codec_restore_gan.py \
  --train_manifest my_methods_GAN/exp/sv_codec_restore/train_manifest_q25.csv \
  --valid_manifest my_methods_GAN/exp/sv_codec_restore/valid_manifest_q25.csv \
  --output_dir my_methods_GAN/exp/sv_codec_restore/run_rec16k_cam_q25 \
  --phase1_epochs 25 \
  --phase2_epochs 8 \
  --phase3_epochs 0 \
  --batch_size 24 \
  --num_workers 8 \
  --segment_seconds 2.0 \
  --phase2_segment_seconds 4.0 \
  --train_sample_fraction 1.0 \
  --valid_sample_fraction 1.0 \
  --emb_dim 64 \
  --num_blocks 6 \
  --hidden_units 128 \
  --attn_heads 4 \
  --lr_g_max 3e-4 \
  --lr_g_min 1e-5 \
  --weight_decay 1e-4 \
  --grad_clip 5.0 \
  --wavlm_root my_methods_GAN/pretrained/WavLM \
  --wavlm_ckpt my_methods_GAN/pretrained/WavLM/WavLM-Base+.pt \
  --campplus_ckpt my_methods_GAN/pretrained/speech_campplus_sv_zh-cn_3dspeaker_16k/campplus_cn_3dspeaker.bin \
  --device cuda
```

注意这版的思想是：

* **先不做 GAN**
* **先只验证 restoration + CAM++ 是否有效**
* **先把结构和 loss 修对**

## 9. 我对你当前结果的最终判断

你这组日志不能说明“GAN 路线不行”，只能说明：

**当前这版实现还没有到可以判断路线优劣的程度。**

因为你现在同时存在这几个问题：

* 生成器和论文 restoration stage 关键结构不一致
* loss 系数借用了论文，但 loss 公式不是同一个
* 48k internal 可能对 16k SV 任务是负担
* WavLM loss 太粗
* GAN 介入过早且可能过强
* 数据量和 epoch 都还是调试级别

所以现在最应该做的不是继续堆文献，而是先把这套实现“收敛到一个能被公平判断的版本”。

