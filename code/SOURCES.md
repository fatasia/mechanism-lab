# 方法与数据来源

独立实现；引用不表示复刻来源产品或论文全部设置。

## 2026-11-02 新增来源（P5 品类扩展）

| 来源 | 用途与归属 |
|---|---|
| https://zenodo.org/records/3929385 | ZeMA 3 台机电缸 2 kHz 寿命试验周期（axis3/5/7），CC BY 4.0；下载器按 8 MB 分块取回并核对发布方 MD5 |
| https://zenodo.org/records/7572501 | CoMoPI 8 台包装机 10 分钟聚合传感器与设备自报报警，CC BY 4.0；报警为设备自生成，非维修确认 |
| https://data.mendeley.com/datasets/g28trvywnx | 直流伺服电机多传感器数据，CC BY 4.0；只取无反转电流波形，`download_servo.py` 以发布方 qc_per_file.csv 的 SHA-256 映射文件并逐文件校验 |
| https://zenodo.org/records/1341583 | PRONTO 多相流基准，CC BY 4.0；仅审计压缩包头部，不解压 |

`download_zenodo.py` 因 Zenodo 对不带 Range 的整文件请求在当前网络下挂起而采用分块下载。数据留在 data/raw，不随源码发布。

## 新增外部设备基准（2026-10-31）

[Virginia Tech NSI液压执行器数据](https://github.com/vtnsi/actuator_dataset)，固定提交f9926eb53e1c324d7ffafc4536f2494b7508d7ed；6台设备、598条记录，仓库MIT许可。只下载和读取数据，不执行或复制仓库训练源码。文件含Prediction/F_Values等派生列，实验排除这些列；设备与故障身份混淆，留出实验不作为跨设备故障分类成功证据。归档与逐文件哈希在data/raw/vtnsi-actuator/source.json，原始数据不随代码发布。

| 来源 | 用途 | 内容进入方式 |
|---|---|---|
| https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems | 数据、标签和单位 | CC BY 4.0；数据留在 data，下载器记录哈希 |
| https://arxiv.org/abs/1607.00148 | 重构误差异常检测背景 | 引用概念，未复制代码；本项目是卷积 AE，不是该文 LSTM |
| https://arxiv.org/abs/1803.01271 | 因果卷积序列建模背景 | 独立的小型因果卷积实验，未复制实现 |
| https://arxiv.org/abs/2109.05257 | 时序评价方法 | 不进行 point adjustment |
| https://www.mdpi.com/2227-9717/7/9/589 | V10 换向与 PS1/PS2 压力响应的公开依据 | 独立实现 8～12 s 原生 100 Hz 窗口与时序特征；未复制论文代码或图像 |
| https://www.mathworks.com/help/predmaint/ug/Rolling-Element-Bearing-Fault-Diagnosis.html | 轴承参考频率与频谱背景 | 独立公式实现，无 MATLAB 源码或图像 |
| https://onnx.ai/sklearn-onnx/supported.html | 模型转换接口 | 通过已安装依赖调用 |
| https://docs.pytorch.org/docs/stable/onnx.html | PyTorch 导出接口 | 通过已安装依赖调用 |

依赖遵循各自许可证。锁文件不是许可证清单；发布时仍需保留实际分发依赖要求的许可证和声明。项目未引入外部 UI 模板、图标或供应商文案；0.4 视觉图像来源另列于下。

## 0.4 新增来源

| 来源 | 用途与归属 |
|---|---|
| https://doi.org/10.17862/cranfield.rd.5097649 | 执行器实测记录，CC BY4.0；25Hz原单位和完整记录分组 |
| https://engineering.case.edu/bearingdatacenter/download-data-file | 原始轴承振动，下载哈希；未获明确数据再分发许可 |
| https://engineering.case.edu/bearingdatacenter/bearing-information | 6205故障参考阶次与转速换算 |
| https://archive.ics.uci.edu/dataset/791/metropt+3+dataset | MetroPT-3在役数据与维修表，CC BY4.0；未匹配告警不当作健康误报 |
| https://arxiv.org/abs/2012.08791 | MiniRocket；隔离环境调用aeon原版并独立实现数值参数ONNX映射 |
| https://arxiv.org/abs/2102.00457 | MultiRocket原版强基线；未导出其全变换 |
| https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.transformations.collection.convolution_based.MiniRocket.html | 原版接口与参数；BSD3声明见licenses |
| https://arxiv.org/abs/1909.04939 | 多尺度时序网络参考；本项目小型残差网络不是原版InceptionTime |
| https://arxiv.org/abs/2106.08265 | PatchCore正常patch记忆思路；本项目冻结ResNet18＋有界记忆，非论文同配置复现 |
| https://github.com/amazon-science/spot-diff | VisA candle图像、掩膜、官方1cls分区，数据CC BY4.0 |
| https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html | 官方预训练ResNet18；许可证与权重归属见THIRD_PARTY_NOTICES |
| https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data | NASA FD001仿真寿命任务；不代表锂电设备RUL，原始数据不分发 |

0.4 演示增加了 VisA 授权公开图像及生成热力图；这不属于供应商私有素材。数据、第三方模型权重、本项目代码的许可分别记录。

## 后续独立研究

| 来源 | 用途与归属 |
|---|---|
| https://data.mendeley.com/datasets/ypzswhhzh9/3 | 气动压力、电流与PLC动作记录；CC BY 4.0；仅确认动作标签，未确认故障真值 |
| https://zenodo.org/records/19857425 | Marine Engine Fault，CC BY 4.0；文件、时间和通道审计，未训练 |
| https://data.mendeley.com/datasets/6s3dggj9mw/1 | MCC5-THU Motor，发布方CC BY 4.0；原始归档已核SHA-256，288个CSV逐文件数值审计已通过；不用镜像的MIT声明替代发布方数据许可 |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC12936562/ | 原始9列格式、电压标度、工况定义及轴承阶次系数；论文282条与归档288条差异保留，未复制外部代码或图像 |
| https://arxiv.org/abs/1607.01719 | 特征协方差对齐背景；本项目是源域类条件矩匹配，不是Deep CORAL原配置复现 |
| https://openreview.net/forum?id=lQdXeXDoWtI | 域泛化中的选型与ERM对照依据；未复制DomainBed实现 |

新数据仅存放在独立 data 目录，不随源码包发布。新增研究结果与在线模型分离。

Plant Dataset for StictionGPT v1：https://data.mendeley.com/datasets/dc9fp3nppf/1 ，DOI 10.17632/dc9fp3nppf.1，CC BY 4.0，作者Tianci Xue、Chao Shang。80份原始文本已核对发布方SHA-256；仅使用现场时序数据，不使用文件名Stick1或已有fs参数作为模型输入。论文 DOI 10.1016/j.conengprac.2025.106588 供任务定义参考，没有复制其模型或训练实现。

