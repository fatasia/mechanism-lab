import { useEffect, useState } from "react";
import { request } from "./domain";

type Entry = {
  seed: number;
  features: number;
  seconds: number;
  validation: { macro_f1: number };
  test: { macro_f1: number; records: number };
};
type Reference = {
  run_id: string;
  results: Record<string, Record<string, Entry[]>>;
};
type Export = {
  run_id: string;
  reference_run: string;
  models: Record<
    string,
    {
      max_abs_error_vs_aeon: number;
      decision_mismatches: number;
      parity_windows: number;
      latency_ms_p50: number;
      latency_ms_p95: number;
    }
  >;
};
type Deployment = {
  run_id: string;
  models: {
    id: string;
    manifest: {
      input: { shape: (number | null)[]; sampleRateHz: number };
      validation: { maxAbsErrors: Record<string, number> };
    };
  }[];
};
const names: Record<string, string> = {
  "record-holdout": "已覆盖工况 · 记录留出",
  "held-out-load": "负载外留出",
  "held-out-profile": "轨迹外留出",
};

export function ReferenceLab() {
  const [reference, setReference] = useState<Reference | null>(null),
    [exported, setExported] = useState<Export | null>(null),
    [protocol, setProtocol] = useState("record-holdout"),
    [error, setError] = useState("");
  const [deployments, setDeployments] = useState<Deployment | null>(null);
  useEffect(() => {
    const ctrl = new AbortController();
    Promise.all([
      request<Reference | null>("/api/benchmarks/reference-study", {
        signal: ctrl.signal,
      }),
      request<Export | null>("/api/benchmarks/minirocket-export", {
        signal: ctrl.signal,
      }),
      request<Deployment | null>("/api/benchmarks/deployments", {
        signal: ctrl.signal,
      }),
    ])
      .then(([r, e, d]) => {
        setReference(r);
        setExported(e);
        setDeployments(d);
      })
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(String(e));
      });
    return () => ctrl.abort();
  }, []);
  return (
    <>
      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>原版强基线 · MiniRocket / MultiRocket</h2>
          </div>
          {reference && (
            <a
              href={`/api/benchmarks/reference-study/download?run_id=${reference.run_id}`}
            >
              下载基准结果 ↓
            </a>
          )}
        </div>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        {!reference ? (
          <p role="status">强基线实验尚未完成或正在读取。</p>
        ) : (
          <>
            <label>
              对照协议
              <select
                aria-label="强基线对照协议"
                value={protocol}
                onChange={(e) => setProtocol(e.target.value)}
              >
                {Object.entries(names).map(([id, label]) => (
                  <option value={id} key={id}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>方法</th>
                    <th>种子</th>
                    <th>特征数</th>
                    <th>验证宏 F1</th>
                    <th>留出宏 F1</th>
                    <th>拟合与全量评分 / 秒</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(reference.results[protocol]).flatMap(
                    ([name, entries]) =>
                      entries.map((e) => (
                        <tr key={`${name}/${e.seed}`}>
                          <td>{name}</td>
                          <td>{e.seed}</td>
                          <td>{e.features.toLocaleString()}</td>
                          <td>{(e.validation.macro_f1 * 100).toFixed(1)}%</td>
                          <td
                            className={
                              e.test.macro_f1 < 0.9 ? "metric-warning" : ""
                            }
                          >
                            {(e.test.macro_f1 * 100).toFixed(1)}%
                          </td>
                          <td>{e.seconds.toFixed(1)}</td>
                        </tr>
                      )),
                  )}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>
      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>MiniRocket · 完整 ONNX 变换链</h2>
          </div>
          {exported && (
            <a
              className="primary"
              href={`/api/benchmarks/minirocket-export/download?run_id=${exported.run_id}`}
            >
              下载 ONNX 模型包 ↓
            </a>
          )}
        </div>
        {exported ? (
          <>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>协议</th>
                    <th>原版对照窗口</th>
                    <th>最大分数差</th>
                    <th>分类差异</th>
                    <th>CPU p50 / p95 · 毫秒</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(exported.models).map(([name, m]) => (
                    <tr key={name}>
                      <td>{names[name]}</td>
                      <td>{m.parity_windows}</td>
                      <td>{m.max_abs_error_vs_aeon.toExponential(2)}</td>
                      <td>{m.decision_mismatches}</td>
                      <td>
                        {m.latency_ms_p50.toFixed(1)} /{" "}
                        {m.latency_ms_p95.toFixed(1)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <p>ONNX 数值检查尚未完成。</p>
        )}
      </section>
      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>统一原始窗口部署包 · 契约 2</h2>
          </div>
          {deployments && (
            <a
              className="primary"
              href={`/api/benchmarks/deployments/download?run_id=${deployments.run_id}`}
            >
              下载全部部署包 ↓
            </a>
          )}
        </div>
        {deployments ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>模型</th>
                  <th>输入 BTC</th>
                  <th>采样率 / Hz</th>
                  <th>对照最大差</th>
                  <th>模型包</th>
                </tr>
              </thead>
              <tbody>
                {deployments.models.map((m) => (
                  <tr key={m.id}>
                    <td>{m.id}</td>
                    <td>
                      {m.manifest.input.shape
                        .map((v) => v ?? "batch")
                        .join(" × ")}
                    </td>
                    <td>{m.manifest.input.sampleRateHz}</td>
                    <td>
                      {Math.max(
                        ...Object.values(m.manifest.validation.maxAbsErrors),
                      ).toExponential(2)}
                    </td>
                    <td>
                      <a
                        href={`/api/benchmarks/deployments/package/${m.id}?run_id=${deployments.run_id}`}
                      >
                        下载 ↓
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>部署包尚未完成校验。</p>
        )}
      </section>
    </>
  );
}
