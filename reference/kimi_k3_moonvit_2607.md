# Kimi K3 技术报告 — MoonViT-V2 / KDA / Per-Head Muon

Moonshot AI，2026-07-27 开源权重 + 技术报告（47 页，已存
`kimi_k3_tech_report_2607.pdf`）。2.8T 参数，1M 上下文。

读它有两个动机：(1) 用户问"现在 kimi3 的视觉是怎么做的，是不是免 patch 的"；
(2) 它的注意力底座 **KDA = Kimi Delta Attention**，delta rule 第一次在 frontier
规模上当主干，正撞我们上一场"delta rule 明明有能力，我们为什么只值 0.15 nat"。

## 一、MoonViT-V2（视觉）

| | |
|---|---|
| 规模 | 27 层 / 401M / 12 头 / RMSNorm / 线性与注意力投影**全部去 bias** |
| **patch size** | **14** |
| 训练 | **从零训，next-token prediction**（非 SigLIP init，非对比学习）|
| 投影前 | **pixel-shuffle 2×2 降采样，视觉 token 数 ÷4** |
| 最大输入 | **3584 × 3584** |
| 图/视频 | 参数全共享；注意力拆成帧内空间 + 帧间时间；时间维再池化压缩 |
| 坐标监督 | 训练时**同时给绝对值和归一化 [0,1] 两种格式** |

**关键换算：LLM 看到的一个视觉 token = 14×2 = 28×28 像素。**
3584² → 256² = 65536 个 ViT token → pixel-shuffle 后 16384 个 LLM token。

### 三条判决

1. **"任意分辨率" ≠ "免 patch"。** native resolution 免掉的是 **resize**，
   patch 一个没少，而且投影前还多降采一次。用户原话"Gemma4 去掉了编码器但
   没去掉 patch"——**Kimi 同理，而且它连编码器塔都还在**。
2. **Kimi 对"看不清小东西"的答案是抬有效分辨率，用 token 数付账**
   （允许 3584²），不是缩 patch。这是**第三条独立证据**支持
   「有效分辨率 + 再获取 > patch 尺寸」这个竞争假设（前两条：AnyRes/tiling
   有效；LLaVA-NeXT/Qwen-VL 同款路线）。而且是最强一条——frontier 模型明写在
   报告里的设计选择。
3. **双格式坐标监督 = 寻址修复，不是粒度修复。** 报告原话目的是 "precise and
   resolution-robust localization"。跟我们从代码推出的"病在寻址不在信息"
   同方向。

### 从零训 vs SigLIP init（对我们有用的旁枝）

报告做了消融：SigLIP 初始化的 MoonViT-3D 在与 LLM 联合优化时**梯度范数持续
偏高且尖峰频发**；从零训的 V2 全程稳定（Fig. 6），且**视觉评测追平 SigLIP
基线**。⇒ **对比预训练不是必需品。**

这一条直接削弱"像素 token 编码器没有预训练权重所以不可行"这个反驳
（见 `transolver_plus_2502_02414.md` 限制节）。

## 二、KDA / AttnRes / Per-Head Muon（架构，跟主线直接相关）

- **层配比：69 层 KDA + 24 层 Gated MLA。** delta-rule 线性注意力当主干，
  softmax 注意力只留 1/4 左右。这是目前 delta rule 规模化最强的公开证据点。
- **AttnRes（Attention Residuals）**：官方描述 "selectively retrieves
  representations across depth rather than accumulating them uniformly"。
  跟我们天梯"读出时钟归消费者"是同型思路，安装位在深度轴而非时间轴。
  （本项目 task #142 已做过一版 AttnRes + MB-v 融合消融。）
- **Per-Head Muon**：Q/K/V 的**动量矩阵按头切开**，逐头做 Newton-Schulz
  正交化。理由：整体正交化把所有头当一个耦合块，梯度尺度大的头主导共享更新
  方向，小尺度头得不到充分归一化；逐头正交化让各头更新尺度对齐。附带好处是
  瘦长块的 NS 迭代比整块便宜。

  **⚠ 这一条直接撞我们的判例。** 我们的 W self-edit 用 Muon 正交化，
  且刚吃过"单一固定 query 方向导致梯度精确秩 1"的亏（⑨，代数证明
  `GG^T G ∝ G`），修法是 `w_group_self_edit_multiquery`（K 个独立 query
  方向）。Per-Head Muon 说的是同一个病的另一个面：**正交化的粒度错了，
  大尺度分量会吃掉小尺度分量。** 我们的多 query 是"给更多方向"，
  它的逐头是"别让方向互相压制"——**可叠加，不冲突**。

## 三、可证伪预测

- **视觉**：needle-size 扫描（`scripts/_demo_slice_vs_patch.py`）。若
  `patch4_hi`（4× token 换有效分辨率）就能闭合与内容自适应 slice 的差距，
  则**诚实结论是"抬分辨率"，像素 token 编码器不值得造**。预注册在脚本头。
- **Muon 粒度**：把 `_muon_orthogonalize` 从整块改成逐头/逐子槽分块，
  单变量对照 eff_rank + 跨 session 迁移。预测：与 multiquery 的收益**可加**
  （治的是不同的病）；若不可加，说明两者其实是同一个自由度，需要重判 ⑨/⑪。
  **未跑**。
- **从零训视觉塔**：若我们做视觉组，K3 的结论意味着可以跳过对比预训练直接
  next-token 联合训——但那是在 2.8T 规模上得到的，**小模型上不可外推**，
  要自己重测。

## 四、限制 / 对我们的潜在反驳

- 报告**没有**公开 KDA 的更新规则细节（门控形式、状态尺寸、与 GDN-2 的异同），
  只给了层配比。"delta rule 在 frontier 规模成立"这句话目前只能当**存在性
  证据**，不能当机制背书。要引用具体机制得等更细的材料。
- 2.8T / 1M 上下文 / 海量视觉语料的结论，**方向可借、幅度不可比**——
  跟 δ-mem 那条同款纪律（`analysis/deltamem_alignment_prereg.md` 已知局限节）。
- MoonViT-V2 "追平 SigLIP" 是在他们自己的评测集合上，且视觉语料规模巨大；
  小规模从零训是否还能追平，报告没有回答。
- Per-Head Muon 的收益报告描述为"更均衡的学习动态、大规模下更稳"，
  **没给消融数字**。当作假设来源，不当作已验证结论。
