可以，**先只试非均匀CWS是最稳的**。这一轮不要加gate、artifactencoder、magconfidence，也不要改loss。目的只验证一个问题：

```text
均匀CWS → 非均匀CWS
是否能提升codec样本的restored效果？
```

---

# 一、这次只改哪里

你只需要改生成器里的这两个函数：

```python
_cws_concat()
_cws_merge()
```

原来是均匀切：

```python
chunks = torch.chunk(x_pad, self.cws_subbands, dim=-1)
```

现在改成按固定频带边界切：

```text
[0:64], [64:160], [160:257]
```

也就是：

```text
0~2kHz
2~5kHz
5~8kHz
```

因为你的设置是：

```text
sample_rate=16k
n_fft=512
F=257
bin_resolution=16000/512=31.25Hz
```

所以：

```text
2kHz ≈ 64 bins
5kHz ≈ 160 bins
8kHz ≈ 257 bins
```

---

# 二、推荐新增参数

在`__init__`里加两个参数：

```python
cws_mode: str = "uniform"
cws_band_edges: tuple[int, ...] | None = None
```

默认保持旧模型：

```python
self.cws_mode = cws_mode
self.cws_band_edges = cws_band_edges
```

如果启用非均匀CWS：

```python
cws_mode = "nonuniform"
cws_band_edges = (0, 64, 160, 257)
```

---

# 三、非均匀CWS的核心逻辑

非均匀频带长度不同：

```text
Band1: 64
Band2: 96
Band3: 97
```

不能直接`cat`，因为频率维长度不一样。所以要先pad到最大长度：

```text
Lmax = 97
```

最后形状：

```text
Band1: [B,2,T,64] → pad → [B,2,T,97]
Band2: [B,2,T,96] → pad → [B,2,T,97]
Band3: [B,2,T,97]

concat:
[B,6,T,97]
```

merge时再反过来裁剪：

```text
[B,6,T,97]
↓
拆成3个[B,2,T,97]
↓
裁剪回64、96、97
↓
拼回[B,2,T,257]
```

---

# 四、建议你这样改函数

你现在的旧函数可以保留，新增一个分支。

## `_cws_concat`

逻辑上改成：

```python
def _cws_concat(self, x):
    # x: [B,2,T,F]
    if self.cws_mode == "uniform":
        x_pad, orig_freq = self._pad_freq(x)
        chunks = torch.chunk(x_pad, self.cws_subbands, dim=-1)
        x_cat = torch.cat(chunks, dim=1)
        return x_cat, orig_freq

    # nonuniform
    orig_freq = x.shape[-1]
    edges = self.cws_band_edges
    assert edges[0] == 0
    assert edges[-1] == orig_freq

    chunks = []
    lengths = []
    for s, e in zip(edges[:-1], edges[1:]):
        band = x[..., s:e]
        chunks.append(band)
        lengths.append(e - s)

    max_len = max(lengths)
    padded = []
    for band in chunks:
        pad_len = max_len - band.shape[-1]
        if pad_len > 0:
            band = nn.functional.pad(band, (0, pad_len))
        padded.append(band)

    x_cat = torch.cat(padded, dim=1)
    return x_cat, orig_freq
```

---

## `_cws_merge`

逻辑上改成：

```python
def _cws_merge(self, x, orig_freq):
    # x: [B,2K,T,F_sub]
    if self.cws_mode == "uniform":
        chunks = torch.chunk(x, self.cws_subbands, dim=1)
        merged = torch.cat(chunks, dim=-1)
        return merged[..., :orig_freq]

    # nonuniform
    edges = self.cws_band_edges
    chunks = torch.chunk(x, self.cws_subbands, dim=1)

    bands = []
    for chunk, s, e in zip(chunks, edges[:-1], edges[1:]):
        band_len = e - s
        bands.append(chunk[..., :band_len])

    merged = torch.cat(bands, dim=-1)
    return merged[..., :orig_freq]
```

---

# 五、必须注意的细节

## 1. `cws_subbands`仍然等于3

这一版不要改成4个子带。保持：

```python
cws_subbands = 3
```

这样：

```text
in_proj输入通道仍然是6
out_proj输出通道仍然是6
```

旧checkpoint可以加载。

---

## 2. `cws_band_edges`长度必须是`cws_subbands+1`

也就是：

```python
len(cws_band_edges) == cws_subbands + 1
```

对于3子带：

```python
(0, 64, 160, 257)
```

---

## 3. 最后一位必须等于真实频点数

你的`n_fft=512`时：

```text
F = n_fft // 2 + 1 = 257
```

所以最后必须是：

```python
257
```

不要写成256或258。

---

## 4. 旧checkpoint能加载，但语义变了

虽然参数形状没变：

```text
in_proj: 6 → 48
out_proj: 48 → 6
```

但CWS每个channel对应的频带已经变了，所以不能直接拿旧checkpoint评测。建议：

```text
加载旧baseline
小学习率finetune
再评测
```

---

# 六、训练策略

这一轮建议只做coded恢复，不混gate。

## 实验A0：原baseline

```text
cws_mode=uniform
```

记录：

```text
coded EER
restored EER
minDCF
```

---

## 实验A1：非均匀CWS

```text
cws_mode=nonuniform
cws_band_edges=0,64,160,257
use_degradation_gate=False
use_artifact_encoder=False
use_mag_confidence=False
```

训练方式建议：

```text
加载原baseline checkpoint
phase1小学习率finetune 3~5 epoch
phase2小学习率finetune 3~5 epoch
```

学习率建议：

```text
lr = 原来的1/3左右
```

如果你原来phase1是`1e-4`，这次可以用：

```text
3e-5
```

如果原来phase2是`3e-5`，这次可以用：

```text
1e-5
```

---

# 七、评估标准

这一轮不要主要看clean退化，先看coded修复是否增强。

重点看：

```text
coded EER
restored EER
restored相对coded改善
minDCF
```

如果结果是：

```text
restored相对coded改善 > 原baseline
```

说明非均匀CWS有效。

如果结果基本不变，也不亏，说明频带划分本身不是主要瓶颈。

如果结果下降，可能是：

```text
1.非均匀边界不合适
2.加载旧checkpoint后分布变化，需要更长finetune
3.低频/高频长度差异导致pad区域影响建模
```

---

# 八、如果担心pad区域影响

第一版可以先不管。
如果后面发现训练不稳，再考虑给pad位置加mask，或者把band边界改得更均衡一点：

```text
[0:72], [72:168], [168:257]
```

但我建议第一版就用：

```text
[0:64], [64:160], [160:257]
```

因为这个频带解释更清楚。

---

# 九、最终建议

你现在就做一个最小改动版本：

```text
只改CWS concat/merge
保持3个子带
使用[0,64,160,257]
不改loss
不加gate
不加artifact encoder
不加mag confidence
加载原baseline小学习率finetune
```

一句话概括：

> 这一步的目标不是解决clean退化，而是验证“频带先验是否能提升codec恢复能力”。如果非均匀CWS能让restored相对coded改善变大，再继续叠加artifactencoder和magconfidence。
