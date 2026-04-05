# Booster + Fast-FoundationStereo 项目备忘录

## 1. 当前项目总目标

### 总目标
在不明显加重 Fast-FoundationStereo 的前提下，优先改善 Booster hard case，尤其是：

- textureless region
- reflective region
- transparent region
- ill-posed region

### 当前阶段目标

1. 固定 baseline 和 benchmark 口径
2. 通过论文阅读明确可迁移的轻量改进方向
3. 先做一个小而可回退、可消融的第一版改进
4. 优先验证 Booster hard case 是否改善，而不是只看平均指标

---

## 2. 当前 baseline 与 benchmark 认知

### 当前本地 baseline

- `paper_protocol = booster_q`
- `max_disp = 416`
- `balanced_input_scale = 0.25`
- `unbalanced_input_scale = 1.0`
- `match_unbalanced_left_to_right = 1`
- `valid_iters = 8`

### 当前最重要的 baseline 文件

- `output_booster_eval_paper_q416_v1/benchmark_quarter_summary.csv`
- `output_booster_eval_paper_q416_v1/summary.json`
- `output_booster_eval_paper_q416_v1/args.yaml`

### 当前官方 benchmark 文件

- `results.txt`

### 当前固定 baseline 结果（本地）

#### Balanced / Train / All / Quarter / All

- Bad2 = `6.6740`
- Bad4 = `4.6726`
- Bad6 = `3.9535`
- Bad8 = `3.5410`
- MAE = `1.5968`
- RMSE = `9.3312`

#### Balanced / Train / No Occ / Quarter / All

- Bad2 = `6.4027`
- Bad4 = `4.6282`
- Bad6 = `3.9565`
- Bad8 = `3.5564`
- MAE = `1.5811`
- RMSE = `9.4598`

#### Unbalanced / Train / All / Quarter / All

- Bad2 = `12.1974`
- Bad4 = `7.0726`
- Bad6 = `4.3984`
- Bad8 = `3.2821`
- MAE = `1.5536`
- RMSE = `5.1102`

### 当前官方 benchmark 关键结果

#### Balanced / All / Full / All

- Bad2 = `22.31`
- Bad4 = `12.26`
- Bad6 = `9.63`
- Bad8 = `8.20`
- AvgErr = `7.67`
- RMS = `18.63`

#### Balanced / All / Half / All

- Bad2 = `12.29`
- Bad4 = `8.21`
- Bad6 = `6.69`
- Bad8 = `5.91`
- AvgErr = `3.84`
- RMS = `9.35`

#### Balanced / All / Quarter / All

- Bad2 = `8.25`
- Bad4 = `5.95`
- Bad6 = `4.98`
- Bad8 = `4.04`
- AvgErr = `1.93`
- RMS = `4.73`

#### Balanced / No Occ / Quarter / All

- Bad2 = `7.86`
- Bad4 = `5.77`
- Bad6 = `4.88`
- Bad8 = `3.93`
- AvgErr = `1.85`
- RMS = `4.40`

#### Unbalanced / All / Full / All

- Bad2 = `47.99`
- Bad4 = `27.38`
- Bad6 = `18.89`
- Bad8 = `14.12`
- AvgErr = `6.46`
- RMS = `12.28`

### 当前 baseline 结果的解释

- 本地 baseline 主要用于：
  - 建立可复现本地对照
  - 做 hard-case / class / scene 分析
- 官方 benchmark 主要用于：
  - 判断模型在隐藏测试集上的真实表现
- 本地 train 结果与官方 test 结果不能混为同一口径
- 当前最重要的趋势判断是：
  - `Balanced / Quarter` 是合理 baseline 口径
  - `class 2/3` 是主要提升空间
  - 官方 balanced 下 `Full -> Half -> Quarter` 持续改善

### 当前最重要的 benchmark 认知

后续改进不能只追求总分提升，更重要的是：

- `Balanced / Quarter / All`
- `Balanced / Quarter / class 2/3`
- hard scenes
- `All / No Occ`

同时还要关注：

- 参数量
- 推理时间
- 显存

### 当前研究判断

Booster 的主要困难不是普通区域，而是：

- non-Lambertian materials
- high-resolution
- unbalanced stereo
- weak texture
- edge ambiguity
- reflective / transparent surfaces

---

## 3. 已读论文总结

## 3.1 Booster 2022

### 论文角色

- 这不是方法论文，而是问题定义和 benchmark 论文
- 主要作用是定义 stereo 还没解决好的开放难题

### 最重要结论

1. Booster 的主要挑战有两个：
   - non-Lambertian surfaces
   - high-resolution images
2. `balanced` 和 `unbalanced` 是数据集原生定义
3. 数据集提供：
   - stereo pairs
   - dense GT disparity
   - material segmentation masks
4. `class 0/1/2/3` 不是物体类别，而是材质难度等级
5. `All / No Occ / Class` 是不同像素子集
6. `Full / Half / Quarter` 是 benchmark 协议的一部分
7. Booster 的主要难点不只是 occlusion，透明、镜面、弱纹理、高分辨率才是核心
8. `unbalanced` 的标准 baseline 做法是把高分辨率参考图对齐到低分辨率图再匹配

### 对当前项目的启发

- Booster 是真正适合做 hard-case 改进的 benchmark
- 后续不能只看 `All`，要重点看 `class 2/3`
- 不能把问题简单归结成遮挡
- 不能把 resize trick 当成本质创新

---

## 3.2 Booster 2024

### 论文角色

- Booster 的升级版 benchmark 说明书
- 相比 2022，更明确地把自己定义为 depth benchmark

### 最重要升级

1. 数据集扩展到：
   - `85 scenes`
   - `606 samples`
2. 有：
   - train
   - stereo test
   - monocular test
3. 继续保留：
   - balanced / unbalanced
   - dense GT
   - material masks

### 最重要实验结论

1. `class 0 -> 3` 的难度递增在 unbalanced 下也成立
2. stereo fine-tuning 对 hard classes 的帮助明显
3. monocular fine-tuning 的收益没有 stereo 那么明显
4. 专门分析了 `transparent` 和 `mirror`

### Transparent vs Mirror 关键结论

- 对 stereo：
  - `mirror` 比 `transparent` 更难
- 对 monocular：
  - `transparent` 比 `mirror` 更难

### 对当前项目的启发

- hard case 不是一种统一错误，而是多种机制混在一起
- 对 stereo 而言，mirror 可能是最难的一类
- 后续即使先改善 transparent / weak-texture，也不代表 mirror 会同步被完全解决

---

## 3.3 FoundationStereo

### 论文角色

- Fast-FoundationStereo 的母模型
- 核心目标是 strong zero-shot stereo generalization

### 能力来源

FoundationStereo 的强，不是来自单一模块，而是：

1. 大规模高质量 synthetic stereo 训练数据
2. monocular foundation priors
3. 更强的 cost filtering + iterative refinement

### 主要结构

1. `STA`
   - Side-Tuning Adapter
   - 引入 monocular foundation prior
2. `Hybrid Cost Volume`
3. `AHCF`
   - Axial-Planar Convolution
   - Disparity Transformer
4. `Iterative Refinement`

### 最重要判断

- `Iterative Refinement` 在这个家族里不是配角，而是核心能力
- `cost filtering` 也很关键，但更重
- `STA` 很强，但不适合当前 Fast 路线第一步直接照搬

### 对当前项目的启发

- 不要优先动 extractor
- `update / refinement` 仍是最合理的第一优先改进位置
- `cost filtering` 可作为第二阶段备选

---

## 3.4 Fast-FoundationStereo

### 论文角色

- 当前项目核心 baseline 论文
- 目标是桥接 `real-time` 和 `zero-shot`

### 核心结论

Fast 不是全新体系，而是对 FoundationStereo 的系统化加速。

### 三块加速策略

1. feature extraction
   - `knowledge distillation`
2. cost filtering
   - `blockwise NAS`
3. disparity refinement
   - `structured pruning`

### 能力来源

Fast 的强，不只是结构压缩，还来自：

- 强 teacher：FoundationStereo
- `1.4M in-the-wild pseudo-labeled stereo pairs`

### 与 Booster 的关系

- Fast 论文明确把 Booster 当作 `non-Lambertian robustness benchmark`
- 在 Booster-Q 上，Fast 已经是很强的实时 baseline

### 最重要判断

1. 当前 baseline 不是弱 baseline，而是很强的实时 zero-shot stereo baseline
2. 后续工作应定位为：
   - 在不明显加重 Fast 的前提下
   - 增量补强其在 Booster hard case 上的短板
3. Fast 三块中：
   - extractor 已蒸馏过，不适合先动
   - cost filtering 已搜索过，敏感且易复杂化
   - refinement 被剪枝过，但仍关键，也存在优化空间

### 对当前项目的启发

- `update / refinement` 继续是最值得优先动的位置
- `cost filtering` 不宜第一步大改
- 不要一开始加 monocular 大分支

---

## 3.5 LoS

### 论文角色

- 当前最适合做第一阶段轻量 hard-case 改进参考的论文之一

### 它在解决什么问题

- challenging areas 是 stereo 性能瓶颈
- optimization-based 方法在这些区域收敛慢、效果差

### LoS 定义的 challenging areas

1. 左图中右图看不到的区域
2. occlusion 区域
3. textureless 区域
4. edge 区域

注意：
- 这不是 Booster 的材质 class
- 这是几何/匹配困难类型分类

### 最重要概念

- `LSI = {G, O, R}`

其中：

- `G`
  - disparity gradients
  - 局部趋势
- `O`
  - disparity offset details
  - 偏离平面趋势的细节
- `R`
  - local relations
  - 邻域传播关系

### 核心模块

- `LSGP`
  - Local Structure-Guided Propagation

### LSGP 核心思想

- 低不确定度区域帮助高不确定度区域
- 而且是在局部结构信息引导下进行传播和更新

### 最值得当前项目学习的东西

1. 困难区域不应该和普通区域用同一套更新规则
2. 高不确定区域应该被有针对性地处理
3. 局部结构不应只理解成平面，还应表达边界和非平面细节

### 当前不建议直接搬的部分

- 完整 monocular branch
- 完整 `PAM + LSGP + MiDaS` 系统
- 全流程完整复刻

### 对当前项目的启发

- LoS 强烈支持：第一阶段优先改 `update / refinement`
- 最值得借的是：
  - uncertainty-aware update
  - local-structure-aware modulation
  - 可靠邻域帮助困难区域的传播思想

---

## 3.6 GREAT-Stereo

### 论文角色

- GREAT 是当前已读论文中，最能代表“全局上下文调制”路线的一篇
- 它和 LoS 互补：
  - `LoS` 更偏局部结构传播
  - `GREAT` 更偏全局上下文调制

### 它在解决什么问题

GREAT 直接把 iterative stereo 在以下区域上的失败当作核心问题：

- occlusions
- textureless regions
- repetitive patterns
- ill-posed regions

作者认为这些问题的重要原因之一是：

- iterative stereo 主要依赖局部信息
- 缺乏 global context 与更强几何信息
- 导致 iterative refinement 在 hardest cases 上不够有效

### 核心结构

GREAT 通过三个 attention 模块为现有 iterative stereo 引入 global context：

1. `SA = Spatial Attention`
   - 在空间维度聚合 global context
   - 更偏全局几何结构建模
2. `MA = Matching Attention`
   - 沿 epipolar line 聚合 global matching context
   - 对 textureless / repetitive pattern 尤其重要
3. `VA = Volume Attention`
   - 把 global context 注入 cost volume
   - 提升 cost representation 的鲁棒性

### 最重要思想

1. ill-posed regions 不只是局部结构问题，也是 global disambiguation 问题
2. 对于 iterative stereo，global context 可以帮助：
   - 从 non-occluded 向 occluded 区域传播结构
   - 沿极线缓解 textureless / repetitive ambiguities
3. global context 不一定只加在 update 内部，也可以通过调制：
   - cost volume
   - motion-related features
   - geometry-related features
   来实现

### 当前项目最值得借的东西

- 不建议完整复刻 `SA + MA + VA`
- 更值得吸收的是一个研究判断：
  - 第一版即使走 LoS 风格的局部增强，也不能让 update 只看特别窄的局部信息
  - 高不确定区域的更新，最好能看到比纯局部更大的上下文

### 当前不建议直接照搬的部分

- 全套 `SA + MA + VA` 原样接入 Fast
- 在 Fast 的 cost volume 主体里直接加完整全局 attention
- 把 GREAT 作为第一版直接实现对象

### 当前项目上的定位

- GREAT 更适合作为“第二种思想来源”
- 第一版仍然优先走 LoS 风格的轻量 update 增强
- 若第一版有效，第二阶段可考虑加入轻量 global modulation

---

## 4. 当前综合研究判断

### 当前最合理路线

1. `update / refinement` 主攻
2. `cost filtering` 备选
3. `extractor` 暂不动

### 当前最不该做的事

- 第一版大改 extractor
- 第一版引入 monocular 大分支
- 第一版重做 cost filtering 主体
- 只追总分不看 hard case
- 混淆本地 train eval 和官方 test benchmark

### 当前最该关注的指标

1. `Balanced / Quarter / All`
2. `Balanced / Quarter / class 2/3`
3. hard scenes
4. `All / No Occ`
5. 参数量
6. 推理时间
7. 显存

### 当前最重要的 hard-case 判断

- 后续改进不能只追求 `All` 提升
- 更应重点观察：
  - `class 2/3`
  - transparent / reflective / mirror-like hard regions
  - hard scenes
- 对 stereo 而言，mirror 类区域可能比 transparent 更难
- 如果方法：
  - `All` 稍有提升
  - 但 `class 2/3` 没有改善
  其研究价值有限

---

## 5. 当前代码结构摘要

## 5.1 eval / submit 脚本定位

### `scripts/eval_booster.py`

作用：

- 在 Booster `train` split 上做本地评测
- 读取：
  - 左右图
  - GT disparity
  - occ mask
  - material mask
- 负责：
  - 按协议推理
  - 构造 `All / No Occ / Class`
  - 计算 `Bad2/4/6/8, MAE, RMSE`
  - 写出 `summary.json` / `per_sample_metrics.csv` / `benchmark_quarter_summary.csv`

可按五层理解：

1. 参数层
2. 数据组织层
3. 推理层
4. 评测层
5. 输出层

### `scripts/submit_booster.py`

作用：

- 在 Booster `test` split 上推理并生成官方提交格式
- 读取：
  - 左右图
- 不读取：
  - GT
  - occ mask
  - class mask
- 负责：
  - 按协议推理
  - 保存 16-bit disparity png
  - 按官方结构打包 zip

### eval 与 submit 的关系

- 前半段推理逻辑尽量一致
- 后半段分叉：
  - `eval` 负责算分和分析
  - `submit` 负责保存 png 和打 zip
- 当前 baseline 的一个关键原则是：
  - 本地 `eval` 和官方 `submit` 协议必须尽量对齐

## 5.2 当前最关键函数

### `infer_single_pair()`

当前用户已经基本能理解这个函数，是脚本中最关键的推理主线。

它的主流程可概括为：

1. 读左右图
2. 若是 `unbalanced` 且左右尺寸不同，则把左图对齐到右图尺寸
3. 按协议 scale 缩放
4. 转 tensor 并 pad
5. 调模型 forward
6. 去 pad，并把 disparity resize / 还原到原图坐标系

### `compute_track_masks()`

这是 `eval_booster.py` 里最关键的评测辅助函数之一。

它负责构造：

- `All`
- `No Occ`
- `class_k`
- `noc_class_k`

本质是：

- `valid_mask`
- `occ_mask`
- `material_mask`

三者的不同组合。

## 5.3 当前不要轻易动的地方

- 不要随意改 metric 公式
- 不要随意改 valid mask 定义
- 不要随意改 disparity decode / save protocol
- 不要在没有明确理由前大改 `core/` 主体
- 当前默认不要动：
  - extractor 主干
  - monocular prior 大分支
  - 提交格式逻辑

---

## 6. 第一版改进候选

## 6.1 候选 1：Uncertainty-Aware Update Gate

### 核心想法

在 `update / refinement` 中加入一个很小的 gate，让高不确定区域触发更强更新，低不确定区域保持轻更新。

### 来源

- 主要受 LoS 启发

### 优点

- 最轻量
- 最容易做成开关
- 最容易消融
- 最符合 Fast 约束

### 当前评价

**最推荐，第一优先**

---

## 6.2 候选 2：Local-Structure-Aware Motion/Geo Modulation

### 核心想法

在 motion feature / geo feature / volume lookup 周边加一个轻量局部结构调制，让 update 在边界、弱纹理和复杂局部区域更稳。

### 来源

- LoS 的 `LSI` 思想

### 优点

- 仍在 update 周边
- 不需要完整单目分支
- 结构上更有解释性

### 风险

- 比候选 1 更复杂
- 设计容易发散

### 当前评价

**推荐，第二优先**

---

## 6.3 候选 3：Confidence-Weighted Neighbor Propagation Lite

### 核心想法

不完整复刻 `LSGP`，而是在 Fast update 里做一个极简版“高置信邻域优先传播”。

### 来源

- LoS 的 propagation 思想

### 优点

- 与 hard case 很对口
- 理论解释性强

### 风险

- 更接近新模块
- 第一版不如 gate 稳

### 当前评价

**可作为第二阶段候选，不建议第一版先做**

---

## 7. 当前推荐的第一版方案草稿

### 暂定题目方向

在不明显增加 Fast-FoundationStereo 推理成本的前提下，针对 Booster 中的 non-Lambertian 与弱纹理 hard cases，提升 refinement 阶段的鲁棒性和纠错能力。

### 第一版方案目标

- 不动大 backbone
- 不加 monocular 大分支
- 不重做 cost filtering 主体
- 优先在 `update / refinement` 中加入轻量 hard-case 定向增强

### 第一版最推荐方案

**Uncertainty-Aware Update Gate**

#### 论文动机版本

Booster benchmark 表明，当前 Fast-FoundationStereo 的主要瓶颈并非普通区域，而是 non-Lambertian 与 weak-texture hard cases，尤其体现在：

- `class 2/3`
- hard scenes
- reflective / transparent / mirror-like ambiguous regions

现有 Fast-FoundationStereo 虽然已经具备较强的 zero-shot 泛化能力，但其 lightweight iterative update 仍然主要采用统一的更新策略。结合 `LoS` 与 `GREAT` 的阅读结论，可以得到如下判断：

1. hard case 往往具有更高不确定性，不能与普通区域采用同样的更新策略
2. 困难区域既需要局部结构约束，也需要比纯局部更大的上下文
3. 对当前 Fast 路线而言，最合理的第一步不是引入新大分支，而是在不明显加重的前提下增强 update 对 hard case 的感知能力

因此，第一版方案拟在 `update / refinement` 中引入一个轻量的 uncertainty-aware update gate，使网络对高不确定区域进行更有针对性的纠错，同时尽量保留 Fast 的实时性。

#### 方法草稿描述

在 update 模块内部引入一个很小的 uncertainty-aware gate：

- 输入：
  - 当前更新特征
  - disparity 或其变化量
  - 可选的简单置信度/不确定度提示
  - 可选的轻量上下文摘要
- 输出：
  - 一个 gating 权重或调制系数
- 作用：
  - 对高不确定区域增强更新力度
  - 对低不确定区域保持较轻更新

#### 设计原则

1. **轻量**
   - 不引入 monocular 大分支
   - 不重做 backbone
   - 不重写 cost filtering 主体
2. **可开关**
   - 必须支持 on/off，方便和 baseline 做公平对照
3. **可消融**
   - 至少能验证：
     - gate 是否有效
     - 不确定度/置信度输入是否有效
     - 是否需要轻量上下文摘要
4. **不明显破坏 Fast**
   - 参数和延迟增加应尽量小

#### 预期优势

- 符合 LoS “困难区域区别处理”的思想
- 吸收 GREAT “困难区域不能只看过窄局部信息”的提醒
- 插入位置自然，和当前 Fast update 结构兼容
- 参数和计算开销有望较小
- 容易做开关和消融

#### 当前对模块风格的限制

- 第一版不要做成完整邻域传播模块
- 第一版不要做成完整 attention 模块
- 第一版更像：
  - 一个轻量 gate
  - 或一个小的 modulation 单元
  - 作用在 update / motion / geo feature 附近

#### 当前最希望验证的核心假设

- 如果 update 能显式区分高不确定区域和普通区域，
- 并在困难区域给予更合适的调制，
- 那么在不明显加重 Fast 的前提下，
- `Balanced / Quarter / class 2/3` 与 hard scenes 的表现应优先改善。

#### 预期实验目标

优先看：

- `Balanced / Quarter / All`
- `Balanced / Quarter / class 2/3`
- hard scenes
- 额外记录：
  - 参数量
  - 推理时间
  - 显存

#### 实验成功标准

第一版并不要求大幅超越 FoundationStereo，而应满足：

1. `Balanced / Quarter / All` 不下降，最好提升
2. `class 2/3` 有可见改善
3. hard scenes 有改善
4. 参数 / 时间 / 显存增加小
5. 方法解释清楚，能说明为什么对 hard case 有帮助

---

## 8. 后续论文阅读顺序建议

已完成较完整阅读：

1. Booster 2022
2. Booster 2024
3. FoundationStereo
4. Fast-FoundationStereo
5. LoS
6. GREAT-Stereo

后续推荐继续读：

1. `Selective-Stereo`
   - 理解当前代码中的 selective 风格来源，避免重复创新
2. `IGEV / IGEV++`
   - 作为 cost / geometry 参考

---

## 9. 后续工作提醒

### 后续任何实验必须保证

- baseline 固定
- protocol 固定
- metric 固定
- 输出目录固定可回溯
- 关键结果能回溯到：
  - 代码版本
  - 参数
  - 结果文件

### 后续任何方法改进都不能忘记

- 不能只追总分
- 必须盯 Booster hard case
- 不能明显破坏 Fast 的 real-time / lightweight 定位

---

## 10. 给下一位 Codex 的交接说明

### 下一位 Codex 接手后，默认应该先做什么

1. 先完整阅读本 memo
2. 默认承认当前 baseline、benchmark 口径和改进优先级
3. 如果要提出不同意见，必须明确指出：
   - 与本 memo 的哪一点冲突
   - 为什么冲突
4. 在真正改代码前，优先：
   - 继续读 `GREAT-Stereo`
   - 或帮助用户把第一版方法草稿进一步收敛

### 当前不应重新争论的问题

- 为什么选 Booster
- 为什么不先改 backbone
- 为什么优先看 `class 2/3`
- 为什么不建议直接加 monocular 大分支
- 为什么当前优先改 `update / refinement`

### 当前允许继续推进的方向

- 继续论文阅读：`GREAT-Stereo`
- 辅助用户完善项目备忘录
- 明确第一版改进设计
- 如果进入实现阶段，优先做：
  - 开关式
  - 轻量
  - 可消融
  - 不破坏 baseline 的改动

### 当前实现阶段的底线

- 保证结果正确
- 保证实验公平
- 保证改动可回退
- 保证 hard-case 指标被显式观察
