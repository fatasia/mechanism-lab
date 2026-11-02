# 冻结研究指标

这里发布完整参数网格、分区、指标与来源哈希，不含原始波形、逐样本厚度真值或模型。

| 文件 | 内容 |
|---|---|
| [sensitivity.json](sensitivity.json) | CoMoPI 27 格与 ZeMA 72 格，保留失败设置 |
| [rejection.json](rejection.json) | 源校准阈值、三个种子、记录分区、覆盖率和全分母指标 |
| [care.json](care.json) | 12 例逐案例九种设置及随机平移对照 |
| [electrode.json](electrode.json) | 两种材料、两种分组、两种模型、两组输入和分组配对区间 |
| [clean-verification.json](clean-verification.json) | 干净源码与数据验收；复用已安装依赖 |

解读和命令见[公开数据补强](../PUBLIC_BENCHMARKS.md)。研究指标由 `export_public_evidence.py` 导出；`metrics_sha256` 对应本地完整原报告，`source_hash` 对应该运行源码快照。原始数据和快照不随仓库分发。

`care.json` 作为 CARE 数据衍生研究记录按 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 发布，不适用项目源码 MIT。数据作者 Christian Gück、Cyriana Roelofs，来源 [Zenodo v6](https://zenodo.org/records/15846963)。本项目进行了子集选择、特征处理、模型训练及自定义评估，不代表作者认可。其他数据署名见[第三方声明](../THIRD_PARTY_NOTICES.md)。

2026-11-02 干净克隆验收通过：206 项测试、前端构建、UCI 新下载与两轮训练、签名校验、2,205 个窗口 HTTP 回放、缺同步不评分/对齐评分、监控、两条预测记录备份恢复和查询参数日志脱敏。检查中修复了依赖本机产物的测试，以及前端锁文件绑定镜像地址的问题。72 小时长稳测试不在本轮验收内。
