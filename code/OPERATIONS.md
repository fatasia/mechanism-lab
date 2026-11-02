# 运行维护

本页描述单机研究与影子服务的同步门控、签名、日志、基础监控和备份。主运行库默认 `work/runtime.sqlite`；所有试验用隔离数据库，不更改主库或当前绑定。

## 同步输入与阀门候选

`POST /api/runtime/predict` 可携带：

```json
{
  "sync": {
    "command_edge_s": 1000.0,
    "window_start_s": 1008.0,
    "channel_skew_s": 0.0
  }
}
```

这是附加字段片段；完整请求还需要 task、entity、sequence、values、channels、units、sample_rate_hz。时间单位为秒，偏斜必须是实际已确认的信息，未知时不填零。同步对象拒绝额外字段和非有限时间值。

仅模型清单声明 `sync.required=true` 时强制执行。预期偏移 `expected_lead_s` 和不超过1采样点的容差来自模型包，不能由请求覆盖。缺命令边沿、窗口时刻或偏斜，或偏移/偏斜超限时，返回 `status=unscored`、`score=null`、`quality.status=unsupported`，不按正常处理。旧包未声明该要求时保留旧行为，不因此获得新的同步保证。

冻结双证据候选可创建新的运行层兼容版本，不重训、不覆盖原包、不修改latest或绑定：

```powershell
$py = './code/python/.venv/Scripts/python.exe'
& $py code/python/prepare_valve_runtime.py --export-run-id <fusion-export-run-id>
```

需要原UCI数据及对应导出结果，校验全量2205个原窗口。新版本输入固定为100 Hz、PS1/PS2 bar、400×2、周期[8,12)秒窗口。该版本的`ae`是运行层异常评分槽位，图内包含冻结AE与机理双证据，不是重新训练的均值AE。清单和报告标为研究候选，只能绑定`shadow`，不能以历史0/0直接激活生产报警。

## 模型目录签名

运行封装保存到`outputs/valve-runtime/runs/<run-id>`，不混入完整训练实验列表；通过`/api/runtime/activate/valve`显式提交run_id并选择shadow。

安装安全依赖：`pip install "./code/python[security]"`。使用[cryptography Ed25519](https://cryptography.io/en/stable/hazmat/primitives/asymmetric/ed25519/)对**完整运行目录**签名，包括bundle、清单、数值验证文件和指标。私钥口令通过`MECHANISM_SIGNING_PASSWORD`传入，不放在命令行或日志中。

```powershell
# 先在部署机设置口令；密钥放在仓库外、受限目录中。
& $py code/python/sign_models.py keygen --private <outside-repo/private.pem> --public <outside-repo/public.pem>
& $py code/python/sign_models.py sign --directory outputs/valve-runtime/runs/<run-id> --key <outside-repo/private.pem>
& $py code/python/sign_models.py verify --directory outputs/valve-runtime/runs/<run-id> --key <outside-repo/public.pem>
$env:MECHANISM_TRUSTED_MODEL_KEY = '<outside-repo/public.pem>'
./code/run.ps1 serve
```

公钥由部署方独立配置，不信任包内自带公钥。配置后，注册层加载模型前拒绝缺签名、错误签名或任何新增/删改文件。签名后的目录保持只读；添加检查报告或重新打包会改变清单，需要形成新发布版本。默认本地模式未配置公钥时兼容历史未签名包，这不等于已经验签。进程缓存已加载的模型，切换信任配置后需重启服务。

私钥不放入模型目录、源码包或仓库。密钥生成不覆盖已有文件；仓库忽略`.pem`、`.key`及`secrets`目录。签名不改变数据和第三方权重许可。

## 日志与基础监控

`code/run.ps1 serve`关闭原始Uvicorn访问日志，使用JSON请求日志：时间、服务生成的请求ID、方法、路由模板、状态码和耗时。不会记录原始路径中的设备标识、查询串、请求体或认证头。业务审计对敏感键及常见内联口令/令牌做脱敏；自由文本脱敏不是任意秘密识别器，不要在表单内容中填凭据。

`GET /api/runtime/monitor`返回当前进程请求/状态码计数、最近最多512次请求的延迟P95、磁盘空间、运行库/WAL体积和通知队列计数。服务模式仅admin可读；本地模式仍限回环。该接口是基础遥测，不是完整告警平台，不包含现场模型漂移准入或多进程汇总。

如绕过启动脚本直接启动Uvicorn，请同时设置`--no-access-log`，避免额外记录原始URL。

## 独立备份与恢复

SQLite备份使用[在线备份API](https://docs.python.org/3.12/library/sqlite3.html#sqlite3.Connection.backup)。完成后执行检查点、切换DELETE日志模式并检查完整性，生成不依赖WAL的快照。备份超时或仍有读者阻止检查点时明确失败，不声明成功。

```powershell
# 单次备份
& $py code/python/runtime_ops.py --database work/runtime.sqlite --backup-dir work/backups
# 显式启动循环：每900秒备份，保留最近96份
& $py code/python/runtime_ops.py --database work/runtime.sqlite --backup-dir work/backups --interval 900 --keep 96
```

循环在前台持续运行，需由部署方纳入进程管理；它不安装系统计划任务。本次开发没有后台启动该循环。每份快照有SHA-256/体积/完成时间清单；只清理备份目录内本工具创建且有清单的快照，保留其它文件。备份目录不是异机灾备，RPO/RTO尚未做现场验收。

恢复步骤：停止写入，校验快照SHA-256和`PRAGMA integrity_check`，复制到**新的运行路径**，设置`MECHANISM_RUNTIME_DB`后启动并核对绑定、事件、历史和任务。不要直接把备份文件作为持续写入库，也不要在运行服务上覆盖数据库。数据、模型和源码需另行备份。
