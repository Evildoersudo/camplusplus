# SC-GridRestore 方法基本原理

## 摘要

针对信源编码（codec）引入的语音失真会显著降低说话人识别性能的问题，本文提出前端恢复方法 SC-GridRestore。该方法在不修改后端说话人识别模型参数的前提下，学习从 coded 语音到 restored 语音的映射。方法采用三阶段递进训练策略：Phase 1 聚焦重建，Phase 2 在重建基础上引入说话人一致性约束，Phase 3 进一步加入对抗学习（及可选表征蒸馏）以增强感知细节。通过多目标联合优化，SC-GridRestore 在保持语音可懂度与自然度的同时，提升了恢复语音中的声纹判别性，从而增强了系统对多 codec、多码率场景的鲁棒性。

## 2.1 实验动机

实际部署中，语音常经过 Opus、AMR-WB、G.711 等不同编码器及码率压缩，导致频谱细节缺失、时间结构扰动与非线性伪影叠加，进而造成说话人嵌入分布偏移。若直接重训后端识别模型，工程代价较高且迁移性受限。基于此，本文选择“前端恢复 + 后端冻结”的技术路线：通过 SC-GridRestore 将 coded 语音映射到更接近 clean 的表示空间，使现有后端在不改动参数的条件下仍保持较好的判别性能。

## 2.2 符号表

| 符号                                         | 含义                                       |
| -------------------------------------------- | ------------------------------------------ |
| $x$                                          | clean 语音波形，$x \in \mathbb{R}^{T}$     |
| $x_c$                                        | coded 语音波形                             |
| $\hat{x}$                                    | SC-GridRestore 输出的 restored 语音        |
| $\mathcal{C}_{m,b}(\cdot)$                   | codec 压缩算子，$m$ 为编码类型，$b$ 为码率 |
| $G_{\theta}(\cdot)$                          | 生成器（SC-GridRestore）                   |
| $D_{\phi}(\cdot)$                            | 判别器（Phase 3 启用）                     |
| $\mathcal{S}(\cdot),\mathcal{S}^{-1}(\cdot)$ | STFT 与 iSTFT 变换                         |
| $\mathcal{L}_{rec}$                          | 重建损失                                   |
| $\mathcal{L}_{spk}$                          | 说话人一致性损失                           |
| $\mathcal{L}_{adv}$                          | 对抗损失（生成器侧）                       |
| $\mathcal{L}_{fm}$                           | 特征匹配损失                               |
| $\mathcal{L}_{wavlm}$                        | 可选高层表征蒸馏损失                       |
| $\lambda_{*}$                                | 各损失加权系数                             |

## 2.3 方法总览

SC-GridRestore 的核心目标是学习映射 $x_c \rightarrow \hat{x}$，使恢复语音同时满足“声学可重建”和“说话人可辨识”两类约束。其整体流程如 Figure 2-1 所示：coded 语音输入生成器后得到 restored 语音；训练时与 clean 语音共同计算多项损失；推理时仅保留生成器模块。

**Figure 2-1 占位（方法总框图）**  
图内容建议：输入（clean/coded）-> 生成器（STFT-CWS-TF-GridNet-iSTFT）-> 输出（restored），并分支到重建损失、说话人损失、GAN 损失与可选蒸馏损失。

## 2.4 数学建模与目标函数

### 2.4.1 问题定义

编码过程可表示为：

$$
x_c = \mathcal{C}_{m,b}(x). \tag{2-1}
$$

SC-GridRestore 学习恢复函数：

$$
\hat{x} = G_{\theta}(x_c). \tag{2-2}
$$

### 2.4.2 生成器时频残差恢复

设 $X_c = \mathcal{S}(x_c)$，网络在时频域预测残差谱 $\Delta X$，并通过 iSTFT 回到波形域：

$$
\Delta X = \Phi_{\theta}(\operatorname{Re}(X_c),\operatorname{Im}(X_c)), \tag{2-3}
$$

$$
\hat{x} = x_c + \mathcal{S}^{-1}(\Delta X). \tag{2-4}
$$

该“残差恢复”机制降低了学习难度：网络重点学习 codec 引入的失真分量，而非重建整段语音。

**Figure 2-2 占位（生成器结构图）**  
图内容建议：STFT 实虚部输入 -> CWS 子带重排 -> TF-GridNet Blocks 堆叠 -> 输出残差频谱 -> iSTFT -> 与输入残差相加。

### 2.4.3 多目标联合损失

总损失定义为：

$$
\mathcal{L}_{\text{total}} =
\lambda_{\text{rec}}\mathcal{L}_{\text{rec}} +
\lambda_{\text{spk}}\mathcal{L}_{\text{spk}} +
\lambda_{\text{gan}}\mathcal{L}_{\text{adv}} +
\lambda_{\text{fm}}\mathcal{L}_{\text{fm}} +
\lambda_{\text{wavlm}}\mathcal{L}_{\text{wavlm}}. \tag{2-5}
$$

其中重建项可写为：

$$
\mathcal{L}_{\text{rec}} =
\alpha\mathcal{L}_{\text{SI-SDR}} +
\beta\mathcal{L}_{\text{MRSTFT}} +
\gamma\mathcal{L}_{\text{CPLX}}. \tag{2-6}
$$

SI-SDR 损失（最小化负 SI-SDR）为：

$$
\mathcal{L}_{\text{SI-SDR}} =
-10\log_{10}
\frac{\left\|\frac{\langle\hat{x},x\rangle}{\|x\|^2}x\right\|^2}
{\left\|\hat{x}-\frac{\langle\hat{x},x\rangle}{\|x\|^2}x\right\|^2}. \tag{2-7}
$$

说话人一致性项可写为：

$$
\mathcal{L}_{\text{spk}} =
\underbrace{\big(1-\cos(f_{\text{spk}}(\hat{x}),f_{\text{spk}}(x))\big)}_{\mathcal{L}_{\text{spk-cos}}}
+\eta\mathcal{L}_{\text{spk-feat}}
+\mu\mathcal{L}_{\text{AM}}. \tag{2-8}
$$

对抗学习项（以 LSGAN 形式为例）为：

$$
\mathcal{L}_{D} =
\mathbb{E}_{x}\big[(D_{\phi}(x)-1)^2\big]
+\mathbb{E}_{x_c}\big[D_{\phi}(\hat{x})^2\big], \tag{2-9}
$$

$$
\mathcal{L}_{\text{adv}} =
\mathbb{E}_{x_c}\big[(D_{\phi}(\hat{x})-1)^2\big]. \tag{2-10}
$$

特征匹配项定义为：

$$
\mathcal{L}_{\text{fm}} =
\sum_{i}\sum_{j}
\frac{1}{N_{ij}}
\left\|D_i^{(j)}(x)-D_i^{(j)}(\hat{x})\right\|_1. \tag{2-11}
$$

若启用高层表征蒸馏，可加入：

$$
\mathcal{L}_{\text{wavlm}} =
\left\|f_{\text{wavlm}}(\hat{x})-f_{\text{wavlm}}(x)\right\|_1. \tag{2-12}
$$

## 2.5 三阶段训练策略

为兼顾训练稳定性与目标完整性，SC-GridRestore 采用课程式三阶段优化：

$$
\min_{\theta}
\begin{cases}
\lambda_{\text{rec}}\mathcal{L}_{\text{rec}}, & \text{Phase 1} \\
\lambda_{\text{rec}}\mathcal{L}_{\text{rec}}+\lambda_{\text{spk}}\mathcal{L}_{\text{spk}}, & \text{Phase 2} \\
\lambda_{\text{rec}}\mathcal{L}_{\text{rec}}+\lambda_{\text{spk}}\mathcal{L}_{\text{spk}}+\lambda_{\text{gan}}\mathcal{L}_{\text{adv}}+\lambda_{\text{fm}}\mathcal{L}_{\text{fm}}+\lambda_{\text{wavlm}}\mathcal{L}_{\text{wavlm}}, & \text{Phase 3}
\end{cases}
	ag{2-13}
$$

其核心逻辑为：先学“可重建”，再学“可识别”，最后学“更自然”。

**Figure 2-3 占位（三阶段训练时序图）**  
图内容建议：横轴为 epoch，按 Phase 1/2/3 分段标注每段启用的损失项与模块（GAN/WavLM 开关状态）。

## 2.6 执行流程

1. 生成配对数据 $(x, x_c)$，覆盖多 codec 与多码率。
2. 执行可选 codec 时移补偿、长度对齐与分段裁剪。
3. 前向计算 $\hat{x}=G_{\theta}(x_c)$。
4. 按阶段计算对应损失并更新参数（Phase 3 同步更新判别器）。
5. 以验证集重建指标与说话人一致性指标选择最优模型。
6. 推理时仅保留生成器，将 coded 语音恢复后送入固定后端识别模型。

**Figure 2-4 占位（推理部署流程图）**  
图内容建议：输入 coded 语音 -> SC-GridRestore 前端恢复 -> 后端说话人嵌入提取 -> 相似度打分与判决。

## 2.7 模型网络结构细节

本节给出 SC-GridRestore 在工程实现中的网络构成，包含生成器与判别器两部分。其中生成器为推理时唯一保留模块；判别器仅在 Phase 3 训练期间参与优化。

### 2.7.1 生成器结构（CWS-TF-GridNet）

生成器以时频残差恢复为主线，计算流程可写为：

$$
\mathbf{R}_0 = \operatorname{RI}(\mathcal{S}(x_c)) \in \mathbb{R}^{2\times T_f\times F}, \tag{2-14}
$$

其中 $\operatorname{RI}(\cdot)$ 表示复谱的实虚部拼接。

为增强局部频带建模能力，采用 CWS（跨子带拼接）操作：

$$
\mathbf{Z}_0 = \operatorname{CWS}(\mathbf{R}_0) \in \mathbb{R}^{(2K)\times T_f\times (F/K)}, \tag{2-15}
$$

其中 $K$ 为子带数。随后经 $1\times1$ 卷积投影至嵌入通道后，串联 $B$ 个 TF-GridNet block：

$$
\mathbf{H}_0 = \operatorname{Conv}_{1\times1}(\mathbf{Z}_0),\quad
\mathbf{H}_{b+1}=\mathcal{G}_b(\mathbf{H}_b),\ b=0,\dots,B-1. \tag{2-16}
$$

每个 $\mathcal{G}_b(\cdot)$ 由三类子模块组成：

1. 频域卷积分支：建模局部频率纹理与伪影模式。
2. 时域序列分支（双向 GRU + 自注意力）：建模长时依赖与跨帧一致性。
3. 频域序列分支（双向 GRU）：建模跨频带相关性。

经输出头预测残差谱并逆 CWS 还原频率维度：

$$
\Delta \mathbf{R} = \operatorname{InvCWS}(\operatorname{Conv}_{1\times1}(\mathbf{H}_B)), \tag{2-17}
$$

最终重建 restored 语音：

$$
\hat{x} = x_c + \mathcal{S}^{-1}(\Delta \mathbf{R}). \tag{2-18}
$$

该结构兼顾局部伪影抑制与全局语音一致性，适合 codec 失真恢复任务。

### 2.7.2 判别器结构（Phase 3）

判别器用于约束恢复语音分布逼近 clean 分布。当前方案包含：

1. 多分辨率时频判别器（MRD）：在多个 STFT 分辨率上对 log-magnitude 图进行二维卷积判别。
2. 可选多频带波形判别器（MBD）：将波形分解为低/中/高频代理分量后，分别使用 1D 卷积判别。

第 $r$ 个分辨率判别分支可表示为：

$$
\mathbf{M}_r(x)=\log\left|\mathcal{S}_r(x)\right|,\quad
d_r(x)=D_r(\mathbf{M}_r(x)). \tag{2-19}
$$

多分支融合由损失层完成（见式 (2-9) 至式 (2-11)）。

### 2.7.3 结构超参数建议（与当前实现一致）

在报告中建议给出默认结构参数，以便复现实验：

1. STFT 参数：$n\_fft=512,\ \text{hop}=128,\ \text{win}=512$。
2. CWS 子带数：$K=3$。
3. GridNet 深度：$B$ 取 5 到 6 层。
4. 嵌入通道：64（实验可配）。
5. 判别器分辨率：$256/512/1024$ 三尺度（MRD）。

**Figure 2-5 占位（网络结构细化图）**  
图内容建议：左侧画生成器细节（STFT -> CWS -> GridNet Blocks -> InvCWS -> iSTFT -> residual add），右侧画 Phase 3 判别器（MRD 三尺度 + 可选 MBD 三频带），并标注“推理仅保留生成器”。

## 2.8 小结

SC-GridRestore 通过“时频残差恢复 + 说话人一致性约束 + 三阶段训练”实现了对 codec 失真语音的前端修复。在保持工程可部署性的前提下，该方法可有效降低编码失真导致的声纹表征偏移，为后端固定的说话人识别系统提供鲁棒性增益。
