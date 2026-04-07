# AGENTS.md

## 项目
这是一个用于复现、评测、误差分析和小范围改进的双目匹配科研仓库，主线是 Fast-FoundationStereo。
默认目标是可复现、可比较、可解释，不做无关重构。

## 结构重点
Fast-FoundationStereo 可以按三块理解：
- feature extractor
- cost volume / cost filtering
- iterative update / refinement

本项目后续改进默认优先：
1. update / refinement
2. cost filtering
3. feature extractor

原因：当前 hard-case 主要是 textureless / ill-posed region，且不能明显把模型做重。

## 默认工作方式
- 先读代码，再决定改哪里
- 优先最小改动，保留 baseline
- 新实验优先加新脚本、新参数或模块开关
- 一次尽量只验证一个变量
- 非必要不要改旧脚本输入输出

## 协作原则
- 不要机械照做，先判断用户当前方向在工程、深度学习和实验设计上是否成立
- 如果用户方向正确，继续推进并补齐实现细节
- 如果用户方向不完整，主动补上关键缺口，尤其是数据划分、metric、mask、decode、对照设计等等
- 如果用户方向有风险，要明确提醒风险来源和可能后果
- 如果用户方向会破坏实验有效性，例如训练测试污染、指标口径混乱、数据泄漏、将分析集误当最终测试集，要直接拦住并给出更合理方案
- 默认目标不是“顺着用户做”，而是帮助用户走向更符合深度学习科研范式的最佳方案
- 总体来说就是：先分析用户所提出的概念是否符合深度学习的标准，是否漏掉了深度学习标准的流程而导致风险，进行全方面的专业判断，不需要一个只会执行命令的助手，需要一个会帮用户纠偏、会提醒用户研究范式风险、会在用户认知不完整时主动刹车的搭档。力求科研级别，符合顶刊以及nature级别的流程和思路。


## 评测前先对齐
比较结果前先确认以下设置一致：
- checkpoint
- dataset / split
- resize / padding
- max_disp
- disparity 解码方式
- valid mask 定义
- metric 公式
- 可视化范围与 colormap

## 改进约束
- 默认不要引入明显更重的 monocular prior / foundation prior 分支
- 默认不要大改 extractor
- 优先考虑轻量的 update 或 cost filtering 改进
- 新模块尽量支持开关，方便消融和回退

## 常用目录
- `core/`: 模型主干与模块
- `my_learning/`: 评测、hard-case 挖掘、分析脚本
- `weights/`: 权重与配置
- `exp_*/`: 实验输出

## 写脚本时
脚本名尽量清楚：`eval_xxx.py`、`analyze_xxx.py`、`ablation_xxx.py`
运行时尽量打印：数据路径、权重路径、样本数、关键参数、输出目录、metric summary。

## 输出要求
完成任务时优先说明：
1. 改了什么
2. 为什么改
3. 改了哪些文件
4. 如何验证
5. 还存在哪些假设、限制或风险

## 不要做的事
- 大规模重构
- 静默改 metric / decode / 数据流程
- 覆盖旧实验结果不说明
- 混淆 disparity 和 depth
- 为了“顺手整理”改动无关代码

## 一句话
先保证结果正确、实验公平、改动可回退，再考虑进一步优化。
