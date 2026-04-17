# SC-GridRestore 方法基本原理（中期报告文字版）

## 摘要

针对低码率或多类型信源编码造成的语音失真会显著削弱说话人识别性能的问题，本文提出前端恢复方法 SC-GridRestore。该方法以“前端增强、后端冻结”为总体思路，在不改动既有说话人识别后端参数的前提下，学习 coded 语音到 restored 语音的映射。训练过程中采用三阶段递进策略：Phase 1 侧重声学重建，Phase 2 在重建基础上引入说话人一致性约束，Phase 3 进一步通过对抗学习（及可选高层表征蒸馏）强化感知细节与分布逼近能力。实验设置面向多 codec、多码率场景，目标是在保证可部署性的同时，降低编码失真导致的声纹表征偏移，提升后端系统在复杂传输条件下的鲁棒性。

## 2.1 实验动机

在真实通信链路中，语音通常需要经过 Opus、AMR-WB、G.711 等编码器压缩。编码过程会引入频谱空洞、相位扰动、带宽受限与量化噪声等复合失真，导致同一说话人的 clean/coded 嵌入在特征空间中产生显著偏移。若直接针对每种 codec 和码率重训后端模型，不仅成本高，而且会带来版本管理复杂、迁移困难与维护压力大等工程问题。

基于上述约束，本文采用“前端修复失真、后端保持不变”的方案：将 coded 语音输入 SC-GridRestore，输出更接近 clean 分布的 restored 语音，再送入固定后端进行嵌入提取与打分。该思路既保留了后端模型的稳定性，又能通过单一前端模块适配多种编码条件，从而形成较好的通用性与扩展性。

进一步地，从任务属性来看，codec 失真并非随机噪声，而是与编码器结构、码率配置和带宽限制密切相关的“结构化退化”。这意味着恢复模型不仅需要补偿谱幅值偏差，还需要修正跨帧与跨频带的时频关系。SC-GridRestore 选择时频域建模并引入说话人一致性约束，正是为了同时处理“声学可懂度”与“身份可辨性”两条主线，避免传统降噪范式在说话人任务中出现的过平滑问题。

## 2.2 符号表

| 符号                                         | 含义                                       |
| -------------------------------------------- | ------------------------------------------ |
| $x$                                          | clean 语音波形，$x\in\mathbb{R}^{T}$       |
| $x_c$                                        | coded 语音波形                             |
| $\hat{x}$                                    | SC-GridRestore 输出的 restored 语音        |
| $\mathcal{C}_{m,b}(\cdot)$                   | codec 压缩算子，$m$ 为编码类型，$b$ 为码率 |
| $G_{\theta}(\cdot)$                          | 生成器（SC-GridRestore 主体）              |
| $D_{\phi}(\cdot)$                            | 判别器（仅在 Phase 3 训练时使用）          |
| $\mathcal{S}(\cdot),\mathcal{S}^{-1}(\cdot)$ | STFT 与 iSTFT                              |
| $\mathcal{L}_{rec}$                          | 重建损失                                   |
| $\mathcal{L}_{spk}$                          | 说话人一致性损失                           |
| $\mathcal{L}_{adv}$                          | 对抗损失（生成器侧）                       |
| $\mathcal{L}_{fm}$                           | 判别器特征匹配损失                         |
| $\mathcal{L}_{wavlm}$                        | 可选高层表征蒸馏损失                       |
| $\lambda_{*}$                                | 各损失项加权系数                           |

## 2.3 方法总览

SC-GridRestore 的核心目标是学习映射 $x_c\rightarrow\hat{x}$，使恢复语音同时满足两类要求：第一，声学层面尽可能接近 clean 语音；第二，身份层面保留说话人判别信息。训练时以 clean/coded 成对样本为监督，联合优化重建、说话人一致性与分布对齐目标；推理时仅保留生成器，输入 coded 语音即可得到 restored 语音。

从系统实现角度看，该方法具有“训练复杂、推理轻量”的特点：复杂性主要集中在训练阶段的多目标联合优化与三阶段调度，而部署阶段仅需一次前向恢复即可接入既有后端。该特性非常适合真实业务中的增量改造场景，即在不影响原有识别链路和标定流程的前提下，通过插入前端模块快速获得鲁棒性增益。

**Figure 2-1 占位（方法总框图）**  
图建议包含：输入端（clean/coded）-> 生成器（STFT-CWS-TF-GridNet-iSTFT）-> restored 输出；训练支路连接重建损失、说话人损失、对抗损失与可选蒸馏损失。

## 2.4 数学建模与目标函数

### 2.4.1 问题定义

编码过程建模为：

$$
x_c = \mathcal{C}_{m,b}(x). \tag{2-1}
$$

SC-GridRestore 学习恢复映射：

$$
\hat{x} = G_{\theta}(x_c). \tag{2-2}
$$

其中 $\theta$ 为生成器参数。训练目标是使 $\hat{x}$ 在重建指标与说话人一致性指标上同时逼近 $x$。

### 2.4.2 生成器时频残差恢复

设 $X_c=\mathcal{S}(x_c)$，网络在复谱域估计失真残差：

$$
\Delta X = \Phi_{\theta}(\operatorname{Re}(X_c),\operatorname{Im}(X_c)). \tag{2-3}
$$

随后通过 iSTFT 回到波形并采用残差式恢复：

$$
\hat{x} = x_c + \mathcal{S}^{-1}(\Delta X). \tag{2-4}
$$

该建模方式避免了“从零重建整段语音”的高难度学习，能够将参数容量集中用于编码伪影的补偿与校正。

**Figure 2-2 占位（生成器结构图）**  
图建议包含：STFT 实虚部输入 -> CWS 子带重排 -> 多层 TF-GridNet block -> 逆 CWS -> iSTFT -> 与输入波形残差相加。

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

重建损失采用多项组合：

$$
\mathcal{L}_{\text{rec}} =
\alpha\mathcal{L}_{\text{SI-SDR}} +
\beta\mathcal{L}_{\text{MRSTFT}} +
\gamma\mathcal{L}_{\text{CPLX}}. \tag{2-6}
$$

其中 SI-SDR 项写为：

$$
\mathcal{L}_{\text{SI-SDR}} =
-10\log_{10}
\frac{\left\|\frac{\langle\hat{x},x\rangle}{\|x\|^2}x\right\|^2}
{\left\|\hat{x}-\frac{\langle\hat{x},x\rangle}{\|x\|^2}x\right\|^2}. \tag{2-7}
$$

说话人一致性损失可表示为：

$$
\mathcal{L}_{\text{spk}} =
\underbrace{\left(1-\cos\left(f_{\text{spk}}(\hat{x}),f_{\text{spk}}(x)\right)\right)}_{\mathcal{L}_{\text{spk-cos}}}
+\eta\mathcal{L}_{\text{spk-feat}}
+\mu\mathcal{L}_{\text{AM}}. \tag{2-8}
$$

对抗学习项（以 LSGAN 形式为例）为：

$$
\mathcal{L}_{D} =
\mathbb{E}_{x}\left[(D_{\phi}(x)-1)^2\right]
+\mathbb{E}_{x_c}\left[D_{\phi}(\hat{x})^2\right], \tag{2-9}
$$

$$
\mathcal{L}_{\text{adv}} =
\mathbb{E}_{x_c}\left[(D_{\phi}(\hat{x})-1)^2\right]. \tag{2-10}
$$

特征匹配损失定义为：

$$
\mathcal{L}_{\text{fm}} =
\sum_{i}\sum_{j}\frac{1}{N_{ij}}
\left\|D_i^{(j)}(x)-D_i^{(j)}(\hat{x})\right\|_1. \tag{2-11}
$$

若启用高层表征蒸馏，则进一步加入：

$$
\mathcal{L}_{\text{wavlm}} =
\left\|f_{\text{wavlm}}(\hat{x})-f_{\text{wavlm}}(x)\right\|_1. \tag{2-12}
$$

上述损失项的分工可以概括为：$\mathcal{L}_{\text{rec}}$ 负责“听起来像 clean”，$\mathcal{L}_{\text{spk}}$ 负责“嵌入上像同一个人”，$\mathcal{L}_{\text{adv}}$ 与 $\mathcal{L}_{\text{fm}}$ 负责“分布上像真实语音”，$\mathcal{L}_{\text{wavlm}}$ 则提供更高层的语义与时序一致性约束。实际训练中，不同项之间存在梯度竞争关系，因此通常需要通过分阶段启用和权重退火来避免早期训练不稳定。

在参数设置上，建议先固定重建主权重（$\lambda_{\text{rec}}$）保证收敛，再逐步提升说话人项和对抗项比重。若观察到验证集重建改善但 SV 指标停滞，可提高 $\lambda_{\text{spk}}$；若语音自然度不足但 SV 已提升，可适度提高 $\lambda_{\text{gan}}$ 与 $\lambda_{\text{fm}}$。这种“先稳后强”的调参策略与三阶段训练机制是一致的。

## 2.5 三阶段训练策略

为兼顾收敛稳定性与目标完整性，采用课程式三阶段优化：

$$
\min_{\theta}
\begin{cases}
\lambda_{\text{rec}}\mathcal{L}_{\text{rec}}, & \text{Phase 1}, \\
\lambda_{\text{rec}}\mathcal{L}_{\text{rec}}+\lambda_{\text{spk}}\mathcal{L}_{\text{spk}}, & \text{Phase 2}, \\
\lambda_{\text{rec}}\mathcal{L}_{\text{rec}}+\lambda_{\text{spk}}\mathcal{L}_{\text{spk}}+\lambda_{\text{gan}}\mathcal{L}_{\text{adv}}+\lambda_{\text{fm}}\mathcal{L}_{\text{fm}}+\lambda_{\text{wavlm}}\mathcal{L}_{\text{wavlm}}, & \text{Phase 3}.
\end{cases}
\tag{2-13}
$$

该策略的直观含义是：先保证可重建，再强调可识别，最后提升分布真实性与细节自然度。

具体而言，Phase 1 主要用于建立稳定的声学映射基础，降低后续多目标优化时的搜索难度；Phase 2 在此基础上引入身份约束，使恢复结果不再仅追求波形近似，而是朝着“更利于说话人判别”的方向优化；Phase 3 则通过判别器反馈修复细粒度伪影，改善主观自然度与高频细节一致性。三阶段递进可显著降低“直接全损失训练”常见的震荡、梯度冲突和局部最优问题。

在实验执行中，阶段切换通常结合两类信号：一是重建指标（如 SI-SDR、MRSTFT）是否进入平台期，二是验证集说话人一致性指标是否开始稳定提升。若前一阶段仍在明显改善，一般不建议过早切换到下一阶段，以免引入额外目标造成优化方向漂移。

**Figure 2-3 占位（三阶段训练时序图）**  
图建议包含：横轴 epoch，按 Phase 1/2/3 分段，标出各阶段开启损失项与模块开关（GAN、WavLM）。

## 2.6 执行流程

训练与推理流程可概括为六步：

1. 构建 $(x,x_c)$ 配对数据，覆盖多 codec 与多码率条件；
2. 对样本执行可选 codec 时移补偿、长度对齐和分段裁剪；
3. 前向计算 $\hat{x}=G_{\theta}(x_c)$；
4. 按当前阶段计算对应损失并更新参数（Phase 3 同步更新判别器）；
5. 在验证集上联合监控重建性能与说话人一致性，选择最优模型；
6. 推理时仅保留生成器，将 restored 语音输入固定后端完成识别。

上述流程在工程落地时还需关注两点：其一，训练集应尽量覆盖目标场景中的 codec 类型与码率分布，以避免前端在域外条件下泛化不足；其二，验证与评测建议采用“clean/coded/restored”三路并行对比，明确区分“相对 coded 的增益”和“距 clean 上限的差距”。这样既能证明方法有效，也有助于后续定位剩余瓶颈。

**Figure 2-4 占位（推理部署流程图）**  
图建议包含：coded 输入 -> SC-GridRestore 前端 -> 说话人嵌入提取 -> 相似度打分 -> 判决输出。

## 2.7 模型网络结构细节

### 2.7.1 生成器结构（CWS-TF-GridNet）

生成器以时频残差恢复为主线。首先将 coded 波形映射到复谱并进行实虚部拼接：

$$
\mathbf{R}_0 = \operatorname{RI}(\mathcal{S}(x_c))\in\mathbb{R}^{2\times T_f\times F}. \tag{2-14}
$$

然后执行 CWS 子带重排：

$$
\mathbf{Z}_0 = \operatorname{CWS}(\mathbf{R}_0)\in\mathbb{R}^{(2K)\times T_f\times(F/K)}. \tag{2-15}
$$

经 $1\times1$ 卷积投影后，堆叠 $B$ 个 TF-GridNet block：

$$
\mathbf{H}_0 = \operatorname{Conv}_{1\times1}(\mathbf{Z}_0),\quad
\mathbf{H}_{b+1}=\mathcal{G}_b(\mathbf{H}_b),\ b=0,\dots,B-1. \tag{2-16}
$$

其中 $\mathcal{G}_b(\cdot)$ 同时包含频域卷积分支、时域序列分支（BiGRU + Attention）和频域序列分支（BiGRU），以联合建模局部频带纹理与长程上下文关系。最后输出残差谱并逆 CWS 合并：

$$
\Delta\mathbf{R}=\operatorname{InvCWS}(\operatorname{Conv}_{1\times1}(\mathbf{H}_B)), \tag{2-17}
$$

再经 iSTFT 回到波形并进行残差加和：

$$
\hat{x}=x_c+\mathcal{S}^{-1}(\Delta\mathbf{R}). \tag{2-18}
$$

从建模机理看，CWS 操作将原始频带重组为更紧凑的子带表示，有利于网络在有限参数量下学习 codec 伪影的局部模式；而 TF-GridNet block 同时沿时间与频率两个维度传播信息，可缓解单一卷积或单一序列模型在跨维依赖建模上的不足。该组合使生成器在不显著增加推理开销的情况下，兼顾了恢复精度与稳定性。

### 2.7.2 判别器结构（Phase 3）

Phase 3 引入判别器以约束分布一致性。核心为多分辨率时频判别器（MRD），并可选多频带波形判别器（MBD）。第 $r$ 个分辨率分支可表示为：

$$
\mathbf{M}_r(x)=\log|\mathcal{S}_r(x)|,\quad d_r(x)=D_r(\mathbf{M}_r(x)). \tag{2-19}
$$

多分辨率输出通过式 (2-9) 至式 (2-11) 进行联合优化，从而提升恢复语音在不同时频尺度上的真实性。

其中 MRD 更关注时频纹理层面的分布一致性，尤其适合捕捉编码带来的谱形畸变；MBD 则从波形子带角度补充约束，对瞬态与带间能量关系更敏感。两者在 Phase 3 中形成互补，可在“不过度牺牲说话人一致性”的前提下改善听感质量。

### 2.7.3 结构超参数建议（与当前实现一致）

建议在报告中给出默认结构参数，便于复现实验：

1. STFT 参数：$n\_fft=512$，hop$=128$，win$=512$；
2. CWS 子带数：$K=3$；
3. GridNet block 层数：$B=5\sim6$；
4. 嵌入通道：64（可在消融中调整）；
5. MRD 分辨率：$256/512/1024$ 三尺度。

**Figure 2-5 占位（网络结构细化图）**  
图建议包含：左侧生成器细化结构，右侧 Phase 3 判别器分支，并标注“推理阶段仅保留生成器”。

## 2.8 小结

SC-GridRestore 通过“时频残差恢复 + 说话人一致性约束 + 三阶段课程式优化”形成了一条兼顾效果与工程可部署性的技术路线。该方法能够在不改动后端识别模型的前提下，显著缓解编码失真引起的声纹特征偏移，为多 codec、多码率条件下的说话人识别提供稳定增益。

总体而言，SC-GridRestore 的价值不只体现在单项指标提升，更体现在方法论层面的可迁移性：当编码条件变化或新增 codec 类型时，可优先通过前端继续学习失真补偿，而无需频繁改造后端识别网络。对中期工作而言，后续可围绕“损失权重自适应”“跨 codec 统一建模”和“低延迟推理压缩”三个方向继续推进，以进一步增强方法在真实系统中的可用性。