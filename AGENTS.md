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
