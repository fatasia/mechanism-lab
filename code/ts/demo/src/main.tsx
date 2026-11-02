import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { createRoot } from "react-dom/client";
import { PageBoundary } from "./PageBoundary";
const Overview = lazy(() =>
  import("./Overview").then((m) => ({ default: m.Overview })),
);
const Analysis = lazy(() =>
  import("./Analysis").then((m) => ({ default: m.Analysis })),
);
const Experiments = lazy(() =>
  import("./Experiments").then((m) => ({ default: m.Experiments })),
);
const AlgorithmLab = lazy(() =>
  import("./AlgorithmLab").then((m) => ({ default: m.AlgorithmLab })),
);
const RuntimeConsole = lazy(() =>
  import("./RuntimeConsole").then((m) => ({ default: m.RuntimeConsole })),
);
const Maintenance = lazy(() =>
  import("./Maintenance").then((m) => ({ default: m.Maintenance })),
);
import {
  Experiment,
  Replay,
  Job,
  Task,
  Model,
  request,
  names,
  algorithms,
} from "./domain";
import "./style.css";

const THEME_KEY = "mechanism-lab-theme";
type Theme = "dark" | "light";

function readTheme(): Theme {
  try {
    return localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

function App() {
  const [theme, setTheme] = useState<Theme>(readTheme),
    [page, setPage] = useState("总览"),
    [task, setTask] = useState<Task>("valve"),
    [model, setModel] = useState<Model>("ae"),
    [runs, setRuns] = useState<Experiment[]>([]),
    [runId, setRunId] = useState(""),
    [replay, setReplay] = useState<Replay | null>(null),
    [cycle, setCycle] = useState(0),
    [job, setJob] = useState<Job>({ status: "idle", task: null, log: "" }),
    [error, setError] = useState(""),
    [jobConnectionError, setJobConnectionError] = useState(false),
    [loading, setLoading] = useState(true),
    [refresh, setRefresh] = useState(0);
  const experiment = runs.find((e) => e.run_id === runId);
  const observedJob = useRef<Job | null>(null);
  const globalPage = page === "算法基准" || page === "运维工作台";
  const titles: Record<string, string> = {
    总览: "设备健康总览",
    部件分析: "部件信号与异常证据",
    实验与模型: "实验对比与模型",
    算法基准: "多信号诊断与工况验证",
    运行与告警: "模型运行与告警复核",
    运维工作台: "设备与维护工作台",
  };
  const selectCycle = useCallback((index: number) => {
    setCycle(index);
    setPage("部件分析");
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // 隐私模式下无法持久化，主题仅在本次会话生效。
    }
  }, [theme]);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setRuns([]);
    setRunId("");
    setCycle(0);
    setError("");
    request<Experiment[]>(`/api/runs/${task}`, { signal: controller.signal })
      .then((v) => {
        setRuns(v);
        setRunId(v[0]?.run_id ?? "");
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(String(e));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [task, refresh]);
  useEffect(() => {
    if (!experiment) {
      setReplay(null);
      return;
    }
    const controller = new AbortController();
    setReplay(null);
    request<Replay>(
      `/api/replay/${task}?model=${model}&run_id=${experiment.run_id}`,
      { signal: controller.signal },
    )
      .then(setReplay)
      .catch((e) => {
        if (!controller.signal.aborted) setError(String(e));
      });
    return () => controller.abort();
  }, [task, model, experiment?.run_id]);
  // 运行时高频读取，空闲时发现其它窗口发起的任务；完成后才刷新版本列表。
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await request<Job>("/api/job", {
          signal: AbortSignal.any([
            controller.signal,
            AbortSignal.timeout(10_000),
          ]),
        });
        if (controller.signal.aborted) return;
        setJobConnectionError(false);
        const previous = observedJob.current;
        observedJob.current = next;
        setJob(next);
        if (
          next.status === "complete" &&
          previous &&
          (previous.id !== next.id || previous.status !== "complete")
        )
          setRefresh((v) => v + 1);
        timer = setTimeout(poll, next.status === "running" ? 1500 : 10_000);
      } catch {
        if (!controller.signal.aborted) {
          // 读取失败不代表训练失败；保留最后状态并重试。
          setJobConnectionError(true);
          timer = setTimeout(poll, 3000);
        }
      }
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [job.status === "running"]);
  async function startTrain(profile: string) {
    setError("");
    try {
      const started = await request<Job>(`/api/train/${task}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          profile,
          epochs: profile === "valve-switch-100hz" ? 80 : 40,
          architecture: profile === "valve-switch-100hz" ? "dense" : "conv",
          normal_policy: profile === "valve-switch-100hz" ? "all" : "stable",
        }),
      });
      setJob({ ...started, log: "" });
    } catch (e) {
      setError(String(e));
    }
  }
  async function cancel() {
    if (!job.id) return;
    const jobId = job.id;
    try {
      await request("/api/job/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: jobId }),
      });
      setJob((current) =>
        current.id === jobId ? { ...current, cancel_requested: true } : current,
      );
    } catch (e) {
      setError(String(e));
    }
  }
  return (
    <div className="shell">
      <aside>
        <div className="brand">
          <span className="brand-mark">M</span> 机理实验室
        </div>
        <div className="nav-label">设备健康研究</div>
        <nav aria-label="主导航">
          {[
            "总览",
            "部件分析",
            "实验与模型",
            "算法基准",
            "运行与告警",
            "运维工作台",
          ].map((p, i) => (
            <button
              aria-current={page === p ? "page" : undefined}
              className={page === p ? "active" : ""}
              key={p}
              onClick={() => setPage(p)}
            >
              <span>0{i + 1}</span>
              {p}
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <span className="status" data-tone="info">
            历史数据
          </span>
          <button
            className="theme-toggle"
            aria-pressed={theme === "light"}
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            {theme === "dark" ? "浅色主题" : "深色主题"}
          </button>
        </div>
      </aside>
      <main>
        <header>
          <div>
            <div className="eyebrow">ALGORITHM ENGINEERING / MECHANISM LAB</div>
            <h1>{titles[page]}</h1>
            {!globalPage && (
              <p className="subtitle">
                UCI 液压试验台 ·{" "}
                {experiment?.cycle_count.toLocaleString() ?? "2,205"} 个周期 ·{" "}
                {experiment?.profile ?? "选择实验版本"}
              </p>
            )}
          </div>
          <span className="badge">MECHANISM LAB / 0.4</span>
        </header>
        {error && (
          <div role="alert" className="error">
            {error}
            <button onClick={() => setRefresh((v) => v + 1)}>重新读取</button>
          </div>
        )}
        <PageBoundary key={page + theme}>
          <Suspense
            fallback={
              <div role="status" aria-label="正在加载页面">
                <div
                  className="skeleton"
                  style={{ height: 96, marginBottom: 14 }}
                />
                <div className="skeleton" style={{ height: 320 }} />
              </div>
            }
          >
            {globalPage ? (
              page === "算法基准" ? (
                <AlgorithmLab />
              ) : (
                <Maintenance />
              )
            ) : (
              <>
                <section className="toolbar">
                  <div className="tabs">
                    {(Object.entries(names) as [Task, string][]).map(
                      ([id, name]) => (
                        <button
                          aria-pressed={task === id}
                          key={id}
                          className={task === id ? "selected" : ""}
                          onClick={() => setTask(id)}
                        >
                          {name}
                          <small>{id.toUpperCase()}</small>
                        </button>
                      ),
                    )}
                  </div>
                  <select
                    aria-label="算法"
                    value={model}
                    onChange={(e) => setModel(e.target.value as Model)}
                  >
                    {(Object.entries(algorithms) as [Model, string][]).map(
                      ([k, v]) => (
                        <option value={k} key={k}>
                          {v}
                        </option>
                      ),
                    )}
                  </select>
                </section>
                {loading ? (
                  <section className="empty" role="status">
                    正在读取实验版本…
                  </section>
                ) : !experiment ? (
                  <section className="empty">
                    <h2>{names[task]}尚无实验</h2>
                    <button
                      className="primary"
                      disabled={job.status === "running"}
                      onClick={() => void startTrain("process-10hz")}
                    >
                      开始训练
                    </button>
                  </section>
                ) : (
                  <>
                    <div className="version-bar">
                      <label>
                        实验版本{" "}
                        <select
                          aria-label="实验版本"
                          value={runId}
                          onChange={(e) => {
                            setRunId(e.target.value);
                            setCycle(0);
                          }}
                        >
                          {runs.map((r) => (
                            <option key={r.run_id} value={r.run_id}>
                              {r.profile} ·{" "}
                              {r.run_id === "legacy" ? "旧版基线" : r.run_id}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                    {page === "总览" && (
                      <Overview
                        key={runId + model}
                        experiment={experiment}
                        model={model}
                        replay={replay}
                        onSelect={selectCycle}
                      />
                    )}{" "}
                    {page === "部件分析" && (
                      <Analysis
                        key={task + runId}
                        experiment={experiment}
                        replay={replay}
                        cycle={cycle}
                        setCycle={setCycle}
                      />
                    )}{" "}
                    {page === "实验与模型" && (
                      <Experiments
                        key={task}
                        experiment={experiment}
                        model={model}
                        onSelect={selectCycle}
                        onTrain={(p) => void startTrain(p)}
                        running={job.status === "running"}
                      />
                    )}
                    {page === "运行与告警" && (
                      <RuntimeConsole
                        key={task + runId}
                        experiment={experiment}
                        model={model}
                        cycle={cycle}
                        setCycle={setCycle}
                      />
                    )}
                  </>
                )}
              </>
            )}
          </Suspense>
        </PageBoundary>
        {job.status !== "idle" && (
          <details className="job" open={job.status === "failed" || job.stale}>
            <summary>
              训练任务 ·{" "}
              {job.stale
                ? "心跳中断"
                : job.worker_status === "queued"
                  ? "排队中"
                  : (
                      {
                        running: "运行中",
                        complete: "完成",
                        cancelled: "已停止",
                        failed: "失败",
                      } as Record<string, string>
                    )[job.status]}{" "}
              · {job.task ? names[job.task] : ""}
            </summary>
            {job.status === "running" && (
              <button
                disabled={!job.id || job.cancel_requested}
                onClick={() => void cancel()}
              >
                {job.cancel_requested ? "正在停止…" : "停止训练"}
              </button>
            )}
            <pre>{job.log || "等待训练日志…"}</pre>
          </details>
        )}
        {jobConnectionError && (
          <p role="status" className="error">
            训练状态连接中断，正在重试
          </p>
        )}
      </main>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
