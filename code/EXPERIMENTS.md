# 独立研究入口

在项目根目录使用工程虚拟环境，数据与结果分别写入data、outputs，日志放work。新运行创建独立目录，不覆盖旧报告。以下步骤不接双平台。

```powershell
$py = './code/python/.venv/Scripts/python.exe'
```

## 阀门局部残差与机理

需要UCI数据及已完成的DenseAE基线：task=valve、profile=valve-switch-100hz、architecture=dense、normal-policy=all、seed=42、epochs=80。新工作区可用 `download_hydraulic.py` 和 `train_hydraulic.py` 的对应参数生成；后者会更新该工作区实验latest，不应在正在验收的工作区随意重训。

依次使用每一步打印的run_id，不把历史ID当作新运行结果：

```powershell
& $py code/python/study_valve_residual.py --baseline-run-id <valve-run-id>
& $py code/python/study_valve_fusion.py --residual-run-id <residual-run-id>
& $py code/python/export_valve_fusion.py --baseline-run-id <valve-run-id> --fusion-run-id <fusion-run-id>
& $py code/python/study_valve_multiseed.py --baseline-run-id <valve-run-id>
& $py code/python/study_valve_stress.py --bundle outputs/valve-fusion-export/runs/<export-run-id>
```

导出只支持验证集选中top10的双证据候选。三种子检查会先验证种子42能否复现输入基线；不同训练配置不应强行跳过一致性检查。扰动报告是敏感性实验，不是新增实测精度。

## 变工况电机

MCC5-THU原始归档约9.56GB，CSV约34.28GB；还需要缓存和实验空间。解压使用系统tar，不运行数据中的外部代码。

```powershell
& $py code/python/download_motor.py
& $py code/python/extract_motor.py
& $py code/python/audit_motor.py
& $py code/python/extract_motor_features.py outputs/motor-audit/runs/<audit-run-id>/metrics.json
& $py code/python/study_motor.py outputs/motor-features/runs/<feature-run-id>
& $py code/python/study_motor.py outputs/motor-features/runs/<feature-run-id> --task binary
& $py code/python/study_motor_cnn.py outputs/motor-features/runs/<feature-run-id>
& $py code/python/extract_motor_orders.py outputs/motor-features/runs/<feature-run-id>
& $py code/python/study_motor_cnn.py outputs/motor-order-features/runs/<order-feature-run-id> --representation order_spectra
& $py code/python/study_motor_cnn.py outputs/motor-order-features/runs/<order-feature-run-id> --representation envelope_order_spectra
```

只有manifest.json存在才表示缓存完成。24类RF/频谱CNN分别进行192/48次拟合，角域每项再进行48次；二分类RF另有96次拟合，只比较故障筛查，不与24类宏F1直接排名。源工况留出决定准入和训练轮数，目标工况不参与调参。逐记录概率、分区和失效结果随实验保存。

RF研究包导出：

```powershell
& $py code/python/export_motor_rf.py outputs/motor-study/runs/<study-run-id> outputs/motor-features/runs/<feature-run-id>
```

它只把列选择和分类器放进ONNX，输入是114维工程特征；不是原始波形端到端图。独立推理只需NumPy/ONNX Runtime，原特征变换另需SciPy。完整使用方法在每个模型包README中。

## 现场阀门黏滞

```powershell
& $py code/python/download_stiction.py
& $py code/python/audit_stiction.py
& $py code/python/study_stiction.py outputs/stiction-audit/runs/<audit-run-id>/metrics.json
```

按完整控制回路留出，不随机拆同回路记录。原文件的Stick1和fs字段不入模，时间异常或离线标志记录从主实验排除并保留覆盖率分母。记录级指标与回路等权指标同时报告。

## 执行器健康参考

```powershell
& $py code/python/study_actuator_mechanism.py
& $py code/python/study_commissioning_budget.py
& $py code/python/study_commissioning_budget.py --directional-study
```

使用data/raw/cranfield-actuator中的发布方数据。前者比较健康运动回归残差，后者固定同一测试集比较2/4/6条投运健康参考与工况尺度收缩。目标健康参考是明确输入条件，不称为零样本迁移；未通过源准入的候选不晋升。

## 数据扩展与事件级评估

下载器对 Zenodo 分块取回并核对发布方 MD5（整文件请求在当前网络下会挂起）。新增依赖 h5py 仅用于读取 ZeMA 的 HDF5。

```powershell
& $py code/python/download_zenodo.py comopi
& $py code/python/download_zenodo.py zema-emc
& $py code/python/study_valve_block_evidence.py <valve-multiseed-run-dir> <valve-stress-run-dir>
& $py code/python/study_false_alarm_decomposition.py --budgets 2 4 6 8 --draws 200
& $py code/python/study_comopi_early_warning.py
& $py code/python/study_zema_degradation.py
& $py code/python/download_zenodo.py zema-axis11-si
& $py code/python/study_zema_degradation.py --axis11 data/raw/zema-axis11-si/axis11_2kHz_ZeMA_PTB_SI.h5
& $py code/python/download_servo.py --speeds 10 20 30 40 50 60
& $py code/python/study_servo_tuning.py
& $py code/python/download_servo.py --sensor vibration
& $py code/python/study_servo_tuning.py --sensor vibration
& $py code/python/study_servo_tuning.py --sensor vibration_orders
& $py code/python/download_zenodo.py pronto
& $py code/python/study_pronto_audit.py
```

块级区间、事件匹配和方差分量的统计函数在 `phm_core/evidence.py` 与 `phm_core/event_metrics.py`，阀门适用域门控在 `phm_core/support.py`，退化趋势在 `phm_core/degradation.py`，均有单元测试。公开结果见[RESULTS.md](RESULTS.md)；完整解读在本地docs/28，不随源码分发。

## 检查与适用范围

```powershell
& $py -m pytest code/python/tests -q
cd code/ts/demo
npm run build
```

运行层长测期间保持src及服务入口冻结，独立研究不改变在线绑定。实验复现、模型导出一致性和现场验收是三项不同检查。公开数据来源及署名见[SOURCES.md](SOURCES.md)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)；论文/开源发布约定见[RESEARCH.md](RESEARCH.md)。
