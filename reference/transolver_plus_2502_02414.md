# Transolver++: An Accurate Neural Solver for PDEs on Million-Scale Geometries

ICML 2025 · arXiv 2502.02414 · 清华 thuml · 源码已存 `transolver_plus_src/`
（`models/Transolver_plus.py` 里的 `Physics_Attention_1D_Eidetic` 是全部要点）

读它不是为了解 PDE。是因为用户 2026-07-29 提出：**ViT 的 16×16 patch 决定了
细粒度不够**，他在做神经 NS 方程时换成 Transolver++ 的"一个点一个 token"
立刻变好，反推大模型看不清小东西可能是同一个病。这份笔记记的是**哪一部分可
迁移、哪一部分不可**。

## 核心机制（读代码，不是读摘要）

Physics-Attention：不在 N 个网格点上做 O(N²) 注意力，而是

1. **软分配**：每个点算一组 slice logits，`w_i,m = softmax_m(proj(x_i)/τ_i)`
2. **聚合**：`s_m = Σ_i w_im·x_i / Σ_i w_im`（**注意分母**）
3. **注意力**：只在 M 个 slice token 之间做（M≈32-64 ≪ N）
4. **散回**：`out_i = Σ_m w_im·s_m`，且 block 级是残差 `fx = Attn(ln(fx)) + fx`

++ 相对 v1 的两个增量：
- **Ada-Temp**：τ 是**逐点逐头**的，`τ_i = MLP(x_i) + bias`，clamp 到 ≥0.01。
  治的是大 N 下分配趋于均匀、slice 之间同质化（论文叫 slice collapse）。
- **eidetic states** + 分布式：slice 统计量跨卡 all_reduce，线性可扩到百万点。

代码细节两条，摘要里没有、但直接影响移植：
- `in_project_slice` 用 **orthogonal 初始化**（注释原文 "a principled
  initialization"）。
- **Gumbel 噪声在 eval 也采**（`gumbel_softmax` 里没有 `self.training` 判断）。
  照抄会让判官随机，我们的 demo 里关掉了并注明。

## 与我们的关系

**要点不是"逐点 token"，是三件事：**

1. **压缩是内容自适应的，不是几何硬网格。** 这跟本项目已判过的教训同源——
   「组合 π 必须**编码空间查询 + 顶下产生**，自底向上内容寻址 = 检索」。
   ViT 的 patchify 是几何池化 = 输入端的自底向上均值寻址。
2. **点流不死。** scatter-back 是残差，逐点表示一直活着。这是它跟
   Perceiver / Flamingo Resampler 的分野——那些压完就把输入扔了，压缩即终局。
   **点流活着才谈得上"再看一次"**（saccade），这是最强的迁移理由。
3. **归一化 ⇒ 尺寸不变性。** 步骤 2 那个分母是全部关键。
   ~~grid 池化让特征幅度正比于面积（1 像素在 4×4 patch 里只贡献 1/16 均值）~~
   **更正 2026-07-29**：ViT 的 patchify **不是 mean pool**，是学习线性投影
   （`Conv2d`，kernel=stride=p），均值只是它的退化特例。活下来的是**叠加/信噪比**
   版本——patch token 是 p² 个像素贡献的**和**，needle 占其中 s²，它在 token 范数
   里的占比随 s²/p² 增长；东西全在，但是被背景主导的向量上的小扰动，梯度和注意力
   跟着大方向走。**信息存在 ≠ 可寻址。**
   slice 侧不变且更干净：除以分配质量后，被单像素独占的 slice **等于该像素的完整
   幅度**，与 s 无关。**线性投影再聪明也只会做加法叠加，它没有那个分母**——
   这正是 `slice_sum` 控制臂要打的东西。

**Ada-Temp 治的病我们量到过。** 块编码器 32-token 均值池化出来的 write key，
实测 `r99 = 1~2`（48 维只有 2 行被写过，见 `_diag_deltamem_verdict.py`）。
自然图像的局部冗余比 PDE 网格大得多，slice 塌缩压力只会更大，秤现成
（`rank_stats` 的 PR/r99 直接量分配矩阵）。

## 可证伪预测（已落地为 `scripts/_demo_slice_vs_patch.py`）

匹配 token 预算下的 needle-size 扫描，五臂 patch4 / patch8 / patch4_hi /
slice / **slice_sum**。核心是 `slice_sum`——去掉步骤 2 的分母。
**如果 slice_sum 不退化，那"归一化才是机制"这个说法就是错的**，赢的是逐点
token 本身。预测全文写死在该脚本文件头，跑之前写的。

## 限制 / 对我们的潜在反驳

- **PDE 网格 ≠ 自然图像。** 网格点不规则、值有物理意义、无局部冗余结构；
  自然图像极度局部冗余，这正是 patchify 能work的原因。**用户的迁移在"小于
  patch 尺度的结构"上成立，在大目标上大概率无收益。**
- **patchify 不是信息销毁。** 线性 patchify 当 `d ≥ p²·C` 时可逆
  （16×16×3=768，ViT-B 的 d=768 恰好临界）。真正的损失是表征/优化层面：
  一个 token 一个向量要同时编码"主要内容"和"角落有个 3 像素红点"，训练压力
  偏向前者。**这两种病的药不同**，demo 用 patch4(可逆) vs patch8(有损) 分开。
- **竞争解释更可能是主因。** 现在 VLM 看不清小东西，三条独立证据指向"有效
  分辨率 + 再获取"而不是 patch 尺寸：(a) AnyRes/tiling 有效；(b) Kimi K3
  MoonViT-V2 patch=14 且投影前再 2×2 pixel-shuffle（28px/token），它的解法是
  **允许 3584² 输入**；(c) K3 用绝对+归一化双格式坐标监督做 "resolution-robust
  localization" = **寻址修复不是粒度修复**。见 `kimi_k3_moonvit_2607.md`。
- **算力。** 512² = 262k 点，前一两层很贵。可行解是中央凹（只在 saccade 落点
  保全分辨率），但那依赖 saccade 循环先存在——**编码器和读出必须一起做，
  单独做编码器没意义**。
- **没有预训练权重。** SigLIP 背后是巨量投入，像素 token 编码器从零开始。
  业界不这么做，这个理由恐怕比架构理由更硬。K3 的反例值得注意：它证明了
  **从零训 + next-token 能追平 SigLIP init**，这一条削弱了本项反驳。
