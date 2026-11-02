import { useCallback, useEffect, useRef, useState } from "react";
import { Cycle, Experiment, Model, request } from "./domain";

type Prediction = {
  version: string;
  sequence: number;
  score: number | null;
  threshold: number;
  status: string;
  alarm: boolean;
  duplicate: boolean;
  quality: { status: string; issues: { code: string }[] };
};
type Alarm = {
  id: number;
  entity: string;
  version: string;
  start: number;
  end: number | null;
  status: string;
  reason: string;
  operator: string | null;
  verdict: string | null;
};
type Binding = { run_id: string; model: string; mode: string };
type Status = {
  bindings: Record<string, Binding | null>;
  events: Alarm[];
  notifications: {
    counts: Record<string, number>;
    recent: {
      id: number;
      event_id: number;
      status: string;
      attempts: number;
      last_error: string | null;
    }[];
  };
};
const jsonPost = (body: unknown) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export function RuntimeConsole({
  experiment,
  model,
  cycle,
  setCycle,
}: {
  experiment: Experiment;
  model: Model;
  cycle: number;
  setCycle: (cycle: number) => void;
}) {
  const [status, setStatus] = useState<Status | null>(null),
    [history, setHistory] = useState<Prediction[]>([]),
    [entity, setEntity] = useState(`lab/${experiment.task}`),
    [operator, setOperator] = useState(""),
    [sequence, setSequence] = useState(0),
    [error, setError] = useState(""),
    [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false),
    [peerJson, setPeerJson] = useState(""),
    [peerResult, setPeerResult] = useState<unknown>(null),
    [driftJson, setDriftJson] = useState(""),
    [driftResult, setDriftResult] = useState<unknown>(null);
  const metric = experiment.models[model],
    binding = status?.bindings[experiment.task];
  const activeEntity = useRef(entity);
  activeEntity.current = entity;
  const fetching = useRef<AbortController | null>(null);
  const isBusy = useRef(busy);
  isBusy.current = busy;
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [connectionError, setConnectionError] = useState("");
  const [live, setLive] = useState(true);
  const sequenceEdited = useRef(false);
  const normal = metric.confusion[0].reduce((a, b) => a + b, 0);
  const gatePassed =
    metric.recall !== null &&
    metric.recall >= 0.95 &&
    normal > 0 &&
    metric.confusion[0][1] / normal <= 0.01 &&
    (metric.onnx_max_abs_error ?? Infinity) <= 1e-4;
  const reload = useCallback(
    async (resetSequence = false) => {
      if (activeEntity.current !== entity) return;
      fetching.current?.abort();
      const controller = new AbortController();
      fetching.current = controller;
      const signal = AbortSignal.any([
        controller.signal,
        AbortSignal.timeout(10_000),
      ]);
      try {
        const [s, h] = await Promise.all([
          request<Status>("/api/runtime/status", { signal }),
          request<Prediction[]>(
            `/api/runtime/history?entity=${encodeURIComponent(entity)}`,
            { signal },
          ),
        ]);
        if (controller.signal.aborted || activeEntity.current !== entity)
          return;
        setStatus(s);
        setHistory(h);
        if (resetSequence || !sequenceEdited.current) {
          setSequence((h[0]?.sequence ?? -1) + 1);
          sequenceEdited.current = false;
        }
        setLastUpdated(new Date());
        setConnectionError("");
      } catch (e) {
        if (!controller.signal.aborted && activeEntity.current === entity)
          setConnectionError(String(e));
      } finally {
        if (fetching.current === controller) fetching.current = null;
      }
    },
    [entity],
  );
  useEffect(() => {
    setEntity(`lab/${experiment.task}`);
  }, [experiment.task]);
  useEffect(() => {
    setHistory([]);
    sequenceEdited.current = false;
    setLastUpdated(null);
    void reload(true);
    return () => fetching.current?.abort();
  }, [reload, experiment.run_id]);
  useEffect(() => {
    if (!live) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      if (
        document.visibilityState === "visible" &&
        !isBusy.current &&
        !fetching.current
      )
        await reload();
      if (!stopped) timer = setTimeout(poll, 3000);
    }
    timer = setTimeout(poll, 3000);
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [live, reload]);
  async function action(fn: () => Promise<void>) {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await fn();
      await reload(true);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function activate(mode: string) {
    await action(async () => {
      await request(
        `/api/runtime/activate/${experiment.task}`,
        jsonPost({ run_id: experiment.run_id, model, mode }),
      );
      setMessage(
        mode === "shadow"
          ? "影子版本已切换，生产参考不受训练发布影响。"
          : "离线门槛通过，版本已激活；现场验收单独记录。",
      );
    });
  }
  async function infer() {
    await action(async () => {
      const channels = await Promise.all(
        experiment.channels.map((_, i) =>
          request<Cycle>(
            `/api/cycle/${experiment.task}/${cycle}?channel=${i}&run_id=${experiment.run_id}`,
          ),
        ),
      );
      const values = channels[0].actual.map((_, i) =>
        channels.map((c) => c.actual[i]),
      );
      const result = await request<Prediction>(
        "/api/runtime/predict",
        jsonPost({
          task: experiment.task,
          entity,
          sequence,
          values,
          channels: experiment.channels,
          units: experiment.units,
          sample_rate_hz: experiment.sample_rate_hz,
        }),
      );
      setMessage(
        `${result.status === "scored" ? "评分完成" : "质量拒算"} · ${result.alarm ? "告警开启" : "告警未开启"}${result.duplicate ? " · 重复窗口已去重" : ""}`,
      );
    });
  }
  async function feedback(id: number, verdict: string) {
    await action(async () => {
      if (!operator.trim()) throw Error("请先填写复核人");
      await request(
        `/api/runtime/events/${id}/feedback`,
        jsonPost({ operator, verdict }),
      );
      setMessage("复核结果已保存");
    });
  }
  return (
    <>
      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>模型运行与持久状态</h2>
          </div>
          <div className="inline-actions">
            <button disabled={busy} onClick={() => void action(async () => {})}>
              刷新状态
            </button>
            <a href="/api/runtime/backup">下载本地状态备份 ↓</a>
          </div>
        </div>
        <div className="lab-controls">
          <label className="check-label">
            <input
              type="checkbox"
              checked={live}
              onChange={(e) => setLive(e.target.checked)}
            />
            自动刷新（3秒）
          </label>
          <span role="status">
            {connectionError
              ? "连接中断 · 显示最后成功读取的数据"
              : lastUpdated
                ? `更新于 ${lastUpdated.toLocaleTimeString("zh-CN", { hour12: false })}`
                : "正在读取运行状态"}
          </span>
        </div>
        {connectionError && (
          <p className="error">
            {connectionError}。
            {live ? "保持自动重试。" : "自动刷新已暂停，可手动刷新。"}
          </p>
        )}
        <div className="evidence-strip">
          <strong>
            {binding
              ? `${binding.mode === "shadow" ? "影子" : "激活"} · ${binding.model}`
              : "尚未绑定模型"}
          </strong>
          <span>{binding?.run_id ?? "选择实验版本后启用"}</span>
        </div>
        <p className={gatePassed ? "" : "metric-warning"}>
          离线准入门槛：{gatePassed ? "通过" : "未通过"} · 召回 ≥95% · 正常误报
          ≤1% · 导出误差 ≤10⁻⁴
        </p>
        <div className="lab-controls">
          <button
            className="primary"
            disabled={busy}
            onClick={() => void activate("shadow")}
          >
            影子启用 / 切换所选版本
          </button>
          <button
            disabled={busy || !gatePassed}
            onClick={() => void activate("active")}
          >
            激活通过门槛的版本
          </button>
        </div>
        <div className="lab-controls">
          <label>
            回放实体
            <input
              aria-label="回放实体"
              disabled={busy}
              value={entity}
              onChange={(e) => setEntity(e.target.value)}
            />
          </label>
          <label>
            数据周期
            <input
              aria-label="推理数据周期"
              type="number"
              min={0}
              max={experiment.cycle_count - 1}
              value={cycle}
              onChange={(e) => setCycle(Number(e.target.value))}
            />
          </label>
          <label>
            提交序号
            <input
              aria-label="提交序号"
              type="number"
              min={0}
              value={sequence}
              onChange={(e) => {
                sequenceEdited.current = true;
                setSequence(Number(e.target.value));
              }}
            />
          </label>
          <button
            className="primary"
            disabled={busy || !binding || !lastUpdated || !!connectionError}
            onClick={() => void infer()}
          >
            提交当前周期推理
          </button>
        </div>
        {message && <p role="status">{message}</p>}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        <details open={history.length > 0}>
          <summary>实体最近评分 · {history.length} 条</summary>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>序号</th>
                  <th>分数</th>
                  <th>阈值</th>
                  <th>质量</th>
                  <th>持续告警</th>
                </tr>
              </thead>
              <tbody>
                {history.slice(0, 30).map((h) => (
                  <tr key={`${h.version}/${h.sequence}`}>
                    <td>{h.sequence}</td>
                    <td>{h.score?.toFixed(5) ?? "—"}</td>
                    <td>{h.threshold.toFixed(5)}</td>
                    <td>{h.quality.status}</td>
                    <td>{h.alarm ? "开启" : "未开启"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </section>
      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>告警复核</h2>
          </div>
          <label>
            复核人
            <input
              aria-label="告警复核人"
              value={operator}
              onChange={(e) => setOperator(e.target.value)}
            />
          </label>
        </div>
        {!status?.events.length ? (
          <p>暂无持久告警。</p>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>实体 / 版本</th>
                  <th>起止序号</th>
                  <th>终止原因</th>
                  <th>人工复核</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {status.events.map((e) => (
                  <tr key={e.id}>
                    <td>
                      {e.entity}
                      <small className="block">{e.version}</small>
                    </td>
                    <td>
                      {e.start}～{e.end ?? "持续"}
                    </td>
                    <td>{e.reason}</td>
                    <td>
                      {e.verdict ?? "未复核"}
                      {e.operator ? ` · ${e.operator}` : ""}
                    </td>
                    <td>
                      <div className="inline-actions">
                        <button
                          onClick={() => void feedback(e.id, "confirmed_fault")}
                        >
                          确认故障
                        </button>
                        <button
                          onClick={() => void feedback(e.id, "false_alarm")}
                        >
                          误报
                        </button>
                        <button
                          onClick={() => void feedback(e.id, "unconfirmed")}
                        >
                          待查
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      <section className="panel">
        <h2>同类群比较与漂移复核</h2>
        <div className="editor-grid">
          <form
            className="editor"
            onSubmit={(e) => {
              e.preventDefault();
              void action(async () =>
                setPeerResult(
                  await request(
                    "/api/runtime/peers",
                    jsonPost(JSON.parse(peerJson)),
                  ),
                ),
              );
            }}
          >
            <label>
              同类群请求 JSON
              <textarea
                aria-label="同类群请求"
                required
                rows={7}
                placeholder={
                  "target_entity、cohort、values、peers[{entity,cohort,values}]、scale_floor"
                }
                value={peerJson}
                onChange={(e) => setPeerJson(e.target.value)}
              />
            </label>
            <button disabled={busy}>计算同类群偏离</button>
            {peerResult !== null && (
              <pre>{JSON.stringify(peerResult, null, 2)}</pre>
            )}
          </form>
          <form
            className="editor"
            onSubmit={(e) => {
              e.preventDefault();
              void action(async () =>
                setDriftResult(
                  await request(
                    "/api/runtime/drift",
                    jsonPost(JSON.parse(driftJson)),
                  ),
                ),
              );
            }}
          >
            <label>
              漂移请求 JSON
              <textarea
                aria-label="漂移请求"
                required
                rows={7}
                placeholder={"reference、current：至少各20个同契约特征向量"}
                value={driftJson}
                onChange={(e) => setDriftJson(e.target.value)}
              />
            </label>
            <button disabled={busy}>计算漂移证据</button>
            {driftResult !== null && (
              <pre>{JSON.stringify(driftResult, null, 2)}</pre>
            )}
          </form>
        </div>
      </section>
      <section className="panel">
        <h2>告警投递队列</h2>
        <p>
          待发送 {status?.notifications.counts.pending ?? 0} · 投递中{" "}
          {status?.notifications.counts.leased ?? 0} · 已送达{" "}
          {status?.notifications.counts.delivered ?? 0} · 失败待处理{" "}
          {status?.notifications.counts.dead ?? 0}
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>通知 / 告警</th>
                <th>状态</th>
                <th>尝试次数</th>
                <th>处理</th>
              </tr>
            </thead>
            <tbody>
              {status?.notifications.recent.map((n) => (
                <tr key={n.id}>
                  <td>
                    {n.id} / {n.event_id}
                  </td>
                  <td>
                    {
                      (
                        {
                          pending: "待发送",
                          leased: "投递中",
                          delivered: "已送达",
                          dead: "失败待处理",
                        } as Record<string, string>
                      )[n.status]
                    }
                  </td>
                  <td>{n.attempts}</td>
                  <td>
                    {n.status === "dead" ? (
                      <button
                        disabled={busy}
                        onClick={() =>
                          void action(async () => {
                            await request(
                              `/api/runtime/notifications/${n.id}/retry`,
                              jsonPost({}),
                            );
                          })
                        }
                      >
                        重新投递
                      </button>
                    ) : (
                      "—"
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
