# 机理实验室

用 Python 算法和 TypeScript 页面验证设备健康模型，面向算法研究与受控影子运行。

- 四部件异常检测、阀门机理消融、执行器多信号诊断和原版强基线。
- 电机振动/电流/阶次对照、现场阀门黏滞、视觉与寿命研究。
- 独立推理与ONNX一致性检查；各包明确原始窗口或工程特征输入。
- 持久告警、设备档案、维修复核、点检与备件领退，测试库与主库隔离。
- 同步门控、模型目录签名、脱敏请求日志、基础监控与独立快照备份。

## 启动

在项目根目录执行，已验证 Python 3.12 / Node 24：

```powershell
python -m venv code/python/.venv
code/python/.venv/Scripts/python.exe -m pip install -r code/python/requirements.lock.txt
code/python/.venv/Scripts/python.exe -m pip install --no-deps -e code/python
cd code/ts/demo
npm ci
npm run build
cd ../../..
./code/run.ps1 train
./code/run.ps1 serve
```

打开 http://127.0.0.1:8765 。有实验产物时跳过 train。测试用 `./code/run.ps1 test`，已发布模型复验用 `./code/run.ps1 verify`。train 会保留四部件基线、阀门消融、验证集选定部署版本和 TCN 对照。

## 扩展算法

阀门局部残差、电机与现场黏滞的复现入口见[EXPERIMENTS.md](EXPERIMENTS.md)。这些研究不更新在线绑定，结果可在“算法基准 → 独立报告”打开；包含合成扰动的报告单独标记。

原版 MiniRocket / MultiRocket 使用隔离环境，不改变主环境的 NumPy/训练依赖：

```powershell
python -m venv work/reference-env
work/reference-env/Scripts/python.exe -m pip install -r code/python/requirements.reference.lock.txt
work/reference-env/Scripts/python.exe -m pip install --no-deps -e code/python
./code/run.ps1 benchmark
```

先完成 train，再运行 benchmark；后者下载公开数据、运行三种子/工况外对照、真实时间回测、视觉与寿命，创建新的不可变版本。VisA 只读取原始流的 candle 子集；需系统 curl。完整强基线可能运行数分钟以上。

单独实验脚本：`study_actuator.py`、`study_actuator_ensemble.py`、`study_rotating.py`、`study_metro.py`、`study_rul.py`、`study_vision.py`。具体协议、结果和失败分析见项目外置 docs/12-算法扩展与强基线实验.md。

## 独立推理

只安装 `pip install "./code/python[infer]"`，不需要 Torch、aeon、FastAPI 或页面。

```python
from pathlib import Path
from phm_core.portable import PortableScorer

model = PortableScorer(Path("解压后的schema2模型目录"))
result = model.score(raw_btc, channels, units, sample_rate_hz)
```

`BundleScorer` 读取四部件 schema1 包；`PortableScorer` 读取统一原始窗口 schema2 包；`DomainScorer` 读取视觉 BCHW 与寿命 BTC 专用包。视觉 resize/RGB/range、寿命周期定义和区间后处理分别写在 manifest，不能混用。MultiRocket 未导出，MiniRocket 的完整变换和分类器已导出。机理 DSP/告警状态不属于神经网络图。

## 服务与资料

同步契约、候选封装、签名验签、监控和备份命令见[运行维护](OPERATIONS.md)。历史实验结果与剩余限制见[结果](RESULTS.md)。

默认本机回环运行。服务模式设置 `MECHANISM_MODE=service` 及环境管理员凭据，可配置 operator/reader；放在 TLS 反向代理后。数据库路径由 `MECHANISM_RUNTIME_DB` 指定，默认 work/runtime.sqlite。部署和恢复见外置 docs/13-运行服务与页面验收.md。

训练由独立工作者执行，任务、取消请求和心跳进入SQLite；API重启后继续查询。`python/soak_service.py`启动隔离HTTP长测，`python/capacity_service.py`执行并发内存检查。告警通知进入事务队列，`python/deliver_notifications.py`只向明确配置的URL投递，默认不外发。

每个实验保存数据/切分、源码快照、指标、误判与 ONNX 对照；失败实验不更新 latest。16 个统一部署包是已有计算图的统一封装，不应重复计算为 16 种新算法。

阀门 AE 测试 FP0/FN12；执行器已覆盖工况集成宏 F1 .9674，但正常仅12条、误报1条；新工况迁移仍差。视觉 AUROC .9615，在低误报阈值仍漏检84/100，不能自动放行；Metro 三次检出均晚于报告起点，不是提前预警。寿命 FD001 RMSE13.27周期来自仿真发动机，不是锂电设备。完整结果留在实验报告。

阀门局部残差＋机理三种子在相同历史测试集上均FP0/FN0，但50ms时间对齐扰动会显著增加漏检。电机RF/CNN的3000rpm外推仍未达标，现场黏滞整回路留出也有漏报。候选导出成功不等于工业验收通过。

数据、产物、私有参考和过程文件不入代码 Git。源码采用[MIT](../LICENSE)；第三方归属见 [SOURCES.md](SOURCES.md)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [RESEARCH.md](RESEARCH.md)。
