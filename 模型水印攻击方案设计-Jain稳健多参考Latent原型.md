# 模型水印攻击方案设计：Jain + 稳健多参考 Latent 原型

> 文档性质：项目主记忆文件 / 实现设计稿 / 实验前方案，不是已验证的实验结论。  
> 当前版本：v0.1  
> 最后更新：2026-08-24  
> 当前阶段：先实现一阶段伪造攻击核心实验；结果有效后再扩展移除攻击、质量约束、对比与消融。

## 0. 新对话恢复指令

在本项目文件夹开启新对话时，请 Codex 先执行以下操作：

1. 完整阅读本文件。
2. 阅读项目中 Jain、Simple Averaging、Tree-Ring 对应的论文学习卡片，并按需核对原论文和代码。
3. 将本文件中的“已冻结决策”作为当前实现依据；不要在没有新证据时重新推翻或扩展首轮方案。
4. 严格区分：论文明确提出的方法、原代码实际实现、当前项目新增设计、尚待实验验证的假设。
5. 从“当前进度与下一步”继续工作，并在实现或实验后更新本文件，不要把计划写成结果。

推荐恢复提示词：

> 请先阅读《模型水印攻击方案设计-Jain稳健多参考Latent原型.md》以及 Jain、Simple Averaging、Tree-Ring 的学习卡片，核对相关代码路径，然后从文件记录的“当前进度与下一步”继续。不要重新设计已冻结部分，也不要把未验证设想描述成实验结论。

---

## 1. 项目目标与总体策略

### 1.1 目标

以 Jain 等人的单参考 latent 优化攻击为基准，将 Simple Averaging 的“多个同密钥水印样本共同估计稳定水印成分”思想迁移到 VAE latent 空间，构造稳健多参考 latent 原型，用于模型水印的伪造攻击，并进一步探索移除攻击。

项目优先目标是在较短时间内得到相对 Jain 单参考基准略有提高的核心实验结果。首轮只做足以判断方案是否有效的实验；若结果积极，再完善问题设定、对比实验、消融实验和论文叙述。

### 1.2 结果优先的研究顺序

1. 跑通 Jain 单参考 baseline。
2. 只替换目标 latent，跑通稳健多参考 full 方法。
3. 在相同设置下比较 baseline 与 full。
4. 若 full 有优势，检查重复性并扩展样本量。
5. 再考虑移除攻击、两阶段质量约束、消融和更多水印方法。
6. 根据稳定观察到的优势，收敛论文的问题设定和贡献表述。

“结果优先”不等于修改评价规则迎合结果。所有 baseline/full 对比必须保持相同数据、密钥、预算、模型、预处理、优化器和评价阈值，并保留失败结果。

---

## 2. 已冻结决策

| 项目 | 当前决策 |
|---|---|
| 基准方法 | Jain 单参考 latent 优化攻击 |
| 新增模块来源 | 学习 Simple Averaging 的多参考估计思想，但不声称原论文提出了 latent 原型 |
| 第一目标水印 | Tree-Ring |
| 第一攻击任务 | 伪造攻击 |
| 第一版优化 | 一阶段联合优化 |
| baseline | 单个同密钥水印参考图像产生目标 latent |
| full | 多个同密钥参考图像，经异常样本剔除后形成目标 latent 原型 |
| 初始参考数 | `N = 5` |
| 初始保留数 | `K = 4`，剔除距中位数中心最远的一个参考 |
| 聚合权重 | 等权；第一阶段不使用置信度权重 |
| 图像质量项 | 沿用 Jain 已有的像素 MSE 正则，不把“增加普通正则项”单独作为创新 |
| 第一阶段不加入 | LPIPS 损失、置信度加权、额外低频正则、两阶段优化、在线访问目标检测器早停 |
| 统计阈值 | Tree-Ring 伪造成功按 `p <= 0.05`；移除成功按 `p > 0.05`，最终以实际评价代码语义为准 |

### 2.1 baseline 与 full 的含义

- **baseline**：Jain 原始的单参考目标 latent 攻击。它是公平比较的基准。
- **full**：在完全相同的优化器和攻击预算下，仅将单参考目标 latent 替换为稳健多参考 latent 原型。

首轮对比应尽量保证“唯一主要变量就是目标 latent 的构造方式”。

---

## 3. 论文思想与本项目适配关系

### 3.1 Jain：攻击骨架

Jain 方法提供以下基本思路：

1. 用代理 VAE 编码图像；
2. 以某个参考 latent 为目标；
3. 直接优化输入图像，使其编码结果靠近目标 latent；
4. 用像素级约束抑制视觉失真；
5. 在目标水印检测器上评价伪造或移除是否成功。

代码核对所得事实：Jain 的 `pgd_attack_lamda` 已包含

\[
\operatorname{MSE}(E_\phi(x_{adv}),z_{target})
+\lambda_{pix}\operatorname{MSE}(x_{adv},x_{src}).
\]

因此，本项目第一阶段不是“首次加入质量正则”，而是改进目标 latent 的估计方式。

### 3.2 Simple Averaging：多参考估计思想

Simple Averaging 在图像或残差空间中对多个样本求平均，以削弱图像内容中互不一致的部分，并保留跨同密钥水印样本较稳定的成分。典型形式包括：

配对/灰盒估计：

\[
\hat{\delta}=\frac{1}{N}\sum_{i=1}^{N}(x_i^w-x_i^c).
\]

非配对/黑盒估计：

\[
\hat{\delta}=\operatorname{Mean}(\{x_i^w\})-
\operatorname{Mean}(\{x_j^c\}).
\]

本项目只借鉴“多参考共同估计稳定水印信息”这一原则，并将其适配至 Jain 的代理 VAE latent 优化框架。稳健 latent 聚合及其具体公式是本项目设计，不是 Simple Averaging 原论文的原始公式。

### 3.3 Tree-Ring：第一阶段目标水印和评价对象

第一阶段先在 Tree-Ring 上完成可运行的 baseline/full 闭环。RingID、Gaussian Shading 等方法在核心结果有效后再扩展，以避免首次实现同时引入过多变量。

---

## 4. 符号定义

| 符号 | 含义 |
|---|---|
| `N` | 同一水印密钥下的参考水印图像数量，初始为 5 |
| `K` | 稳健筛选后保留的参考数量，初始为 4 |
| `x_i^w` | 第 `i` 张含相同目标水印密钥的参考图像 |
| `x_i^c` | 与 `x_i^w` 内容匹配的干净图像，仅配对移除估计使用 |
| `x_c` | 需要被伪造成含目标水印的干净载体图像 |
| `x_s^w` | 需要移除水印的源图像 |
| `x^{(t)}` | 第 `t` 次优化迭代的攻击图像 |
| `E_\phi` | 参数为 `\phi` 的代理 VAE 编码映射 |
| `s_\phi` | 代理 VAE 配置中的 latent scaling factor |
| `z_i^w` | 第 `i` 张参考水印图像的代理 VAE latent |
| `m` | 多个参考 latent 的逐元素中位数中心 |
| `D` | 单个 latent 的元素总数，`D=C_zH_zW_z` |
| `d_i` | 第 `i` 个参考 latent 到中位数中心的归一化平方距离 |
| `S` | 距离最小的 `K` 个参考索引集合 |
| `z_{proto}^w` | 稳健多参考水印 latent 原型 |
| `r_{proto}^w` | 多参考估计得到的水印 latent 方向，用于移除 |
| `z_{target}` | 优化希望达到的目标 latent |
| `\lambda_{pix}` | 像素质量损失权重 |
| `\alpha` | 图像优化的学习率或单步步长 |
| `T` | 总优化迭代次数 |
| `\beta` | 移除时从源 latent 中减去水印方向的强度 |
| `\Pi_{[-1,1]}` | 将图像像素投影/裁剪至模型要求的有效范围 |

---

## 5. 代理 VAE latent 的构造

为与 Jain 代码保持一致，对输入图像 `x` 的编码定义为：

\[
E_\phi(x)=
\frac{\operatorname{mode}(\operatorname{VAE}_\phi.\operatorname{encode}(x))}{s_\phi}.
\]

其中：

- `VAE_\phi.encode(x)` 输出 latent 分布；
- `mode(·)` 取该分布的确定性众数，避免随机采样噪声；
- `s_\phi` 来自 `vae.config.scaling_factor`；
- 输入的尺寸、归一化方式和精度必须与 baseline 完全一致。

对 `N` 张同密钥水印参考图像编码：

\[
z_i^w=E_\phi(x_i^w),\qquad i=1,\ldots,N.
\]

重要条件：所有参考图像必须嵌入相同的目标密钥/消息，但应使用不同内容、提示词或生成随机种子，以降低聚合结果对某一图像语义内容的依赖。

---

## 6. 稳健多参考 latent 原型

### 6.1 中位数中心

对 `N` 个 latent 在每一个元素位置取中位数：

\[
m=\operatorname{Median}(z_1^w,z_2^w,\ldots,z_N^w).
\]

`m` 的形状与单个 latent 相同。逐元素中位数只用于确定相对稳健的中心，不直接作为最终优化目标。

### 6.2 样本级异常度

设 latent 的通道、高、宽分别为 `C_z,H_z,W_z`，元素数为

\[
D=C_zH_zW_z.
\]

第 `i` 个 latent 到稳健中心的距离为：

\[
d_i=\frac{1}{D}\lVert z_i^w-m\rVert_2^2.
\]

该距离用于识别整体上偏离其他参考的样本，可能的偏离原因包括图像内容异常、预处理不一致或编码结果不稳定。

### 6.3 截断均值原型

取距离最小的 `K` 个样本索引：

\[
S=\operatorname{TopKSmallest}(\{d_i\}_{i=1}^{N},K).
\]

最终原型为：

\[
z_{proto}^w=\frac{1}{K}\sum_{i\in S}z_i^w.
\]

首轮设 `N=5,K=4`，即剔除距离中位数中心最远的一张参考图，再对其余四张等权平均。这是**样本级异常剔除 + 等权平均**，不是置信度加权。

### 6.4 暂不使用置信度权重的原因

逆方差或距离权重可以从异方差噪声估计角度给出启发式解释，但目前没有证据证明其在本项目的小样本 latent 聚合中稳定有效；当 `N=3–5` 时，权重估计也容易被内容差异主导。为使贡献和变量更清晰，第一阶段只使用稳健筛选后的等权平均。

---

## 7. 伪造攻击设计（第一优先级）

### 7.1 baseline

从一个同密钥水印参考图像构造：

\[
z_{target}^{base}=E_\phi(x_1^w).
\]

### 7.2 full

使用第 6 节的稳健聚合：

\[
z_{target}^{full}=z_{proto}^w.
\]

### 7.3 联合损失

以干净载体图像初始化：

\[
x^{(0)}=x_c.
\]

原型匹配损失：

\[
\mathcal{L}_{proto}^{(t)}=
\operatorname{MSE}(E_\phi(x^{(t)}),z_{target}).
\]

像素质量损失：

\[
\mathcal{L}_{pix}^{(t)}=
\operatorname{MSE}(x^{(t)},x_c).
\]

总损失：

\[
\mathcal{L}_{forge}^{(t)}=
\mathcal{L}_{proto}^{(t)}+
\lambda_{pix}\mathcal{L}_{pix}^{(t)}.
\]

图像更新：

\[
x^{(t+1)}=\Pi_{[-1,1]}\left(
x^{(t)}-\alpha\nabla_x\mathcal{L}_{forge}^{(t)}
\right),\quad t=0,\ldots,T-1.
\]

这里的“一阶段”表示 latent 匹配和像素质量在同一目标函数中联合优化，并不表示没有质量约束。

### 7.4 公平比较要求

baseline 与 full 必须共享：

- 同一组载体图像；
- 同一目标水印密钥；
- 同一代理 VAE 和目标生成/检测模型；
- 同一图像预处理；
- 同一 `\lambda_{pix}`、`\alpha` 和 `T`；
- 同一评价代码、阈值和随机种子集合。

两者的核心区别只能是 `z_target` 来自单参考还是稳健多参考原型。

---

## 8. 移除攻击设计（第二优先级）

移除方向需要估计水印在代理 latent 中造成的变化。优先采用有配对干净图像的设置，以减少内容偏差。

### 8.1 配对/灰盒 latent 方向

对每对水印图像和匹配干净图像：

\[
r_i^w=E_\phi(x_i^w)-E_\phi(x_i^c).
\]

对 `\{r_i^w\}` 使用与第 6 节相同的“中位数中心—距离筛选—截断均值”过程，得到：

\[
r_{proto}^w=\operatorname{RobustAgg}(r_1^w,\ldots,r_N^w).
\]

对待移除图像：

\[
z_s^w=E_\phi(x_s^w),
\]

目标 latent 为：

\[
z_{target}=z_s^w-\beta r_{proto}^w.
\]

移除总损失：

\[
\mathcal{L}_{remove}^{(t)}=
\operatorname{MSE}(E_\phi(x^{(t)}),z_{target})+
\lambda_{pix}\operatorname{MSE}(x^{(t)},x_s^w).
\]

其中 `x^{(0)}=x_s^w`，`\beta` 控制去除方向的强度。

### 8.2 非配对/黑盒备选

若没有严格配对数据，可尝试：

\[
r_{proto}^w=
\operatorname{RobustAgg}(\{E_\phi(x_i^w)\})-
\operatorname{RobustAgg}(\{E_\phi(x_j^c)\}).
\]

但该估计容易混入两组图像的语义或域差异，风险明显高于配对方案，因此不是首轮实现重点。

---

## 9. 代码依据与相关路径

### 9.1 Jain 代码根目录

```text
参考论文和代码/
└─ Forging and Removing Latent-Noise Diffusion Watermarks Using a Single Image/
   └─ watermark_forgery_removal-main/
```

相关文件：

```text
Tree-Ring/forgery.py
Tree-Ring/removal.py
Tree-Ring/utils.py
RingID/forgery.py
RingID/removal.py
RingID/my_utils.py
```

已核对的重要实现点：

- `Tree-Ring/utils.py` 和 `RingID/my_utils.py` 中的 `pgd_attack_lamda` 已实现 latent MSE 与像素 MSE 的联合损失。
- `Tree-Ring/utils.py` 中存在 `pgd_attack_lpips` 等其他攻击函数，可供后续质量阶段参考，但第一阶段不启用。
- `RingID/my_utils.py` 中存在 `pgd_attack_fft_multi`，后续修改前必须核对其真实语义，不能仅凭函数名认定它等同于本方案。
- Jain 的现有运行脚本含硬编码 `/scratch/...` 路径及数据集依赖，需要为本地或 AutoDL 环境参数化。
- 生成多参考时必须固定目标水印密钥/`w_seed`；只能改变图像内容、提示词或普通生成随机种子。不能让每张参考使用不同水印密钥。

### 9.2 Simple Averaging 代码

```text
参考论文和代码/
└─ Can Simple Averaging Defeat Modern Watermarks/
   └─ watermark-steganalysis-master/
      └─ benchmark.py
```

相关函数包括 `sum_images` 和 `get_difference_list`。它们用于理解图像平均、残差平均及伪造/移除流程；本项目需做 latent 空间适配，不能直接把原图像平均代码视为可复用实现。

### 9.3 Tree-Ring 官方代码

```text
参考论文和代码/
└─ Tree-Ring Watermarks Fingerprints for Diffusion Images that are Invisible and Robust/
   └─ tree-ring-watermark-main/
```

正式实验前需用官方代码与 Jain 版本交叉核对密钥生成、嵌入、检测统计量和 p 值方向。

---

## 10. 建议实现结构

保留原论文代码不直接破坏，在 Jain 代码根目录中新建：

```text
experiments/
└─ multiref_latent/
   ├─ prototype.py
   ├─ attack_core.py
   ├─ prepare_tree_ring_refs.py
   ├─ run_forgery.py
   ├─ run_removal.py
   ├─ evaluate.py
   └─ configs/
      └─ tree_ring_stage1.yaml
```

### 10.1 `prototype.py`

建议函数：

```python
encode_vae_latents(images, vae)
robust_aggregate(latents, keep_k)
build_watermark_prototype(watermarked_images, vae, keep_k)
build_watermark_direction(watermarked_images, clean_images, vae, keep_k)
```

`robust_aggregate` 除返回原型外，还应保存：

- 每个参考的 `d_i`；
- 被保留和剔除的索引；
- 聚合前后的形状、dtype 和设备；
- 原型文件及对应元数据。

### 10.2 `attack_core.py`

将 Jain 原来“在攻击函数内部编码单张参考图”的逻辑重构为接收预计算目标：

```python
optimize_to_target_latent(
    source_image,
    target_latent,
    vae,
    lambda_pixel,
    learning_rate,
    num_iterations,
)
```

baseline 和 full 必须调用同一个核心优化函数。

### 10.3 `prepare_tree_ring_refs.py`

职责：

- 固定同一个 Tree-Ring 密钥/消息；
- 用不同内容、提示词或生成种子生成 `N` 张参考；
- 保存图像、密钥标识、提示词、生成种子和模型配置；
- 可选保存对应干净图，为后续配对移除攻击准备数据。

### 10.4 `run_forgery.py`

建议提供：

```text
--mode baseline
--mode full
```

两种模式读取同一配置，只改变目标 latent 构造方式。

### 10.5 `evaluate.py`

统一计算攻击成功率、p 值、图像质量、耗时，并将逐样本结果保存为 CSV/JSON，不能只保存汇总均值。

---

## 11. 第一轮最小核心实验

### 11.1 实验目的

只回答一个问题：在相同攻击预算和质量正则下，稳健多参考 latent 原型是否优于 Jain 单参考目标 latent？

### 11.2 最小设置

| 项目 | 首轮设置 |
|---|---|
| 水印方法 | Tree-Ring |
| 攻击任务 | 伪造 |
| 方法 | baseline vs full |
| baseline 参考数 | 1 |
| full 参考数 | 5 |
| full 保留数 | 4 |
| 载体图像数 | 先用 10–20 张做方向判断 |
| 水印密钥 | 所有参考固定为同一个目标密钥 |
| 评价 FPR/阈值 | 先采用 0.05，并记录具体检测器阈值和代码版本 |
| 超参数 | 从已成功复现的 Jain baseline 配置冻结，不盲信未验证默认值 |

### 11.3 必须记录的指标

- **ASR**：伪造成功率；Tree-Ring 第一阶段按 `p <= 0.05` 判断。
- 每组 p 值的均值、中位数及逐样本值。
- **PSNR、SSIM、LPIPS**：攻击图与原始载体图之间的质量。
- 单图运行时间、总时间和实际迭代数。
- 参考图索引、被剔除样本及其 `d_i`。
- 随机种子、代码提交/文件版本、完整配置和硬件环境。

### 11.4 初步有效标准

满足以下任一稳定现象即可进入扩展验证：

1. 在视觉质量接近时，full 的 ASR 高于 baseline；
2. 在 ASR 接近时，full 的 PSNR/SSIM 更高或 LPIPS 更低；
3. 在 ASR 与质量接近时，full 所需迭代或时间更少。

“略有提高”需要在多个样本和至少若干随机种子上方向一致，不能只选择单个成功案例。

### 11.5 首轮暂不做

- 大规模跨数据集实验；
- 大量竞品对比；
- 全部水印方法迁移；
- 完整消融矩阵；
- 两阶段优化；
- 置信度加权；
- 同时搜索大量 `N/K/\lambda/\alpha/T` 组合。

---

## 12. 结果有效后的扩展顺序

1. 扩大载体图数量并更换随机种子，确认趋势可重复。
2. 加入简单均值版本，区分“多参考收益”和“稳健筛选收益”。
3. 小规模测试 `N`：如 1、3、5、7；测试 `K`：如 `N`、`N-1`。
4. 开展配对 latent 方向的移除攻击。
5. 增加两阶段质量约束。
6. 迁移至 RingID 或 Gaussian Shading。
7. 根据实际优势设计必要的对比和消融，并形成论文问题设定。

---

## 13. 暂缓的两阶段质量约束

只有在一阶段 full 已证明攻击方向有效、但图像质量成为主要瓶颈时，再实现两阶段优化。

### 13.1 阶段 A：优先进入有效攻击区域

使用较低的质量权重，重点减小：

\[
\mathcal{L}_{proto}=
\operatorname{MSE}(E_\phi(x),z_{target}).
\]

### 13.2 阶段 B：在维持攻击约束的同时恢复质量

可采用铰链式约束：

\[
\mathcal{L}_{B}=
\mathcal{L}_{quality}+
\rho\max(0,\mathcal{L}_{proto}-\tau)^2.
\]

其中：

- `\mathcal{L}_{quality}` 可由像素 MSE 与感知损失组成；
- `\rho` 是违反 latent 约束时的惩罚权重；
- `\tau` 是阶段 A 预先标定的 latent 匹配容许阈值。

阶段切换优先依据离线标定的代理 latent 阈值，而不是在每张攻击图优化时反复查询目标检测器。后者会改变无盒/黑盒威胁模型。

两阶段是否有效属于待验证假设，不得在实验前写成既定贡献。

---

## 14. 关键风险与检查项

### 14.1 同密钥条件

若参考图使用不同水印密钥，聚合可能抵消真正的密钥特征。生成参考数据后必须首先检查密钥标识是否一致。

### 14.2 内容偏差

VAE latent 大量编码图像内容，不保证直接等同于扩散初始噪声或纯水印信号。多参考图需保持内容多样，且应检查聚合收益是否来自某类内容偶然接近。

### 14.3 代理空间与目标检测器不完全一致

`E_\phi(x)` 是代理 VAE latent，不是严格的 DDIM 反演初始噪声。其与 Tree-Ring 检测统计量之间的桥梁需要实验证明。

### 14.4 `N=5,K=4` 只是首轮工程选择

该设置便于快速验证，但不是论文已经证明的最优值，也不能在没有消融时声称最优。

### 14.5 检测与归属概念不能混用

Tree-Ring 第一阶段主要考察水印存在性/p 值。若以后讨论多用户密钥归属、误归属或身份伪造，需重新定义指标，不能直接把存在性 ASR 当作归属成功率。

### 14.6 路径和运行环境

原代码含硬编码路径、Hugging Face 数据集和特定服务器环境假设。运行前应把路径、模型位置、设备、精度和数据源全部配置化。

---

## 15. 证据与状态登记

| 条目 | 类型 | 当前状态 |
|---|---|---|
| Jain 使用代理 VAE latent 目标优化图像 | 论文/代码事实 | 已从学习材料和代码确认 |
| Jain 已有像素 MSE 质量正则 | 代码事实 | 已确认 |
| Simple Averaging 使用多样本平均估计稳定水印成分 | 论文/代码事实 | 已确认 |
| 将 Simple Averaging 迁移为稳健 latent 原型 | 本项目设计 | 已确定，未实验 |
| 中位数中心后剔除最远样本 | 本项目设计 | 已确定，未实验 |
| `N=5,K=4` 优于单参考 | 研究假设 | 未验证 |
| 多参考 latent 方向可改善移除攻击 | 研究假设 | 未验证 |
| 两阶段约束可改善质量—攻击权衡 | 研究假设 | 暂缓，未验证 |
| Tree-Ring `p <= 0.05` 为伪造成功 | 评价代码语义 | 实验前需再次端到端核对 |

---

## 16. 当前进度与下一步

### 当前进度

- 已确定基准：Jain 单参考 latent 攻击。
- 已确定模块：Simple Averaging 启发的稳健多参考 latent 原型。
- 已确定首轮范围：Tree-Ring 伪造、一阶段优化、baseline vs full。
- 已确定第一版聚合：`N=5,K=4`、样本级异常剔除、等权平均。
- 本文件仅完成方案固化；尚未实现代码、运行 baseline 或产生实验结果。

### 下一步操作顺序

1. 对照 Jain 与 Tree-Ring 学习卡，逐行核对伪造入口、VAE 编码、密钥生成、检测器及 p 值语义。
2. 清理 Jain Tree-Ring baseline 的路径和配置，使单参考实验可在当前环境运行。
3. 用 1–2 张载体图做端到端 smoke test，保存完整日志。
4. 按第 10 节新增多参考模块，并确保 baseline/full 共用攻击核心。
5. 生成固定同密钥的 5 张多内容参考图。
6. 在 10–20 张相同载体上运行 baseline/full。
7. 汇总逐样本 ASR、p 值、PSNR、SSIM、LPIPS、时间和失败案例。
8. 根据结果决定：扩大验证、调整聚合、转向移除，或启用两阶段质量约束。

---

## 17. 后续讨论时推荐使用的 Skill

本节只记录与当前工作直接相关的 Skill，不包含 documents、pdf 或 Zotero。

- **research-paper-code-study**：阅读论文与代码、建立论文—文件—函数映射、核对实现语义、维护本研究记忆文件。
- **academic-research-suite**：核心结果产生后，用于实验分析、统计解释、贡献边界、论文结构、审稿视角检查和学术完整性检查。

Skill 只辅助流程，不能替代对原论文、原代码和实验日志的直接核验。

---

## 18. 更新规范

每次发生以下事件后更新本文档：

- baseline 首次成功运行；
- 目标公式或威胁模型发生改变；
- 新增/修改代码文件；
- 完成一轮正式实验；
- 某个假设被支持、否定或仍不确定；
- 决定启用移除攻击或两阶段优化。

更新时至少写明：日期、修改内容、证据路径、运行命令/配置、结果文件位置和仍未解决的问题。不要删除失败实验记录；应标注失败原因和后续处理。

### 变更记录

- **2026-08-24 / v0.1**：固化 Jain + 稳健多参考 latent 原型的一阶段方案、公式、实现结构、最小实验和后续扩展顺序。
