import { useEffect, useMemo, useState } from "react";
import { Chart, lines, palette } from "./Chart";
import { request } from "./domain";

type RegressionMetric = { rmse_cycles: number; mae_cycles: number };
type LifeRecord = {
  unit: number;
  observed_cycles: number;
  truth: number;
  prediction: number;
  lower: number;
  upper: number;
};
type LifeStudy = {
  run_id: string;
  selected: string;
  models: Record<
    string,
    {
      validation: RegressionMetric;
      test: RegressionMetric;
      capped_test: RegressionMetric;
      interval_coverage: number;
      mean_interval_width_cycles: number;
    }
  >;
  records: Record<string, LifeRecord[]>;
  manifest: { parity_max_abs_error: number };
};
export function LifeLab() {
  const [study, setStudy] = useState<LifeStudy | null>(null),
    [model, setModel] = useState(""),
    [unit, setUnit] = useState(1),
    [error, setError] = useState("");
  useEffect(() => {
    const abort = new AbortController();
    request<LifeStudy | null>("/api/benchmarks/rul-study", {
      signal: abort.signal,
    })
      .then((r) => {
        setStudy(r);
        if (r) setModel(r.selected);
      })
      .catch((e) => {
        if (!abort.signal.aborted) setError(String(e));
      });
    return () => abort.abort();
  }, []);
  const records = study?.records[model] ?? [];
  const option = useMemo(
    () =>
      lines(
        records.map((r) => r.unit),
        [
          {
            name: "真实剩余周期",
            data: records.map((r) => r.truth),
            color: palette().muted,
          },
          {
            name: "预测剩余周期",
            data: records.map((r) => r.prediction),
            color: palette().primary,
          },
          {
            name: "区间下界",
            data: records.map((r) => r.lower),
            color: palette().info,
          },
          {
            name: "区间上界",
            data: records.map((r) => r.upper),
            color: palette().info,
          },
        ],
        undefined,
        "留出发动机编号",
        "剩余运行周期",
      ),
    [records],
  );
  const record = records.find((r) => r.unit === unit);
  return (
    <section className="panel">
      <div className="section-heading">
        <div>
          <h2>剩余寿命 · 周期与不确定性</h2>
        </div>
        {study && (
          <a
            className="primary"
            href={`/api/benchmarks/rul-study/download?run_id=${study.run_id}`}
          >
            导出寿命模型
          </a>
        )}
      </div>
      {error && <p role="alert">{error}</p>}
      {!study ? (
        <p role="status">尚无寿命实验结果，运行 study_rul.py 后查看。</p>
      ) : (
        <>
          <p className="muted">
            70 条训练 · 15 条选型 · 15 条区间校准 · 100 条官方测试
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>方法</th>
                  <th>选型 RMSE</th>
                  <th>测试 RMSE / 周期</th>
                  <th>测试 MAE</th>
                  <th>封顶目标 RMSE</th>
                  <th>区间覆盖</th>
                  <th>平均区间宽度</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(study.models).map(([name, m]) => (
                  <tr key={name}>
                    <td>
                      {name === "ridge" ? "岭回归" : "极端随机树"}
                      {name === study.selected ? " · 验证集选中" : ""}
                    </td>
                    <td>{m.validation.rmse_cycles.toFixed(2)}</td>
                    <td>{m.test.rmse_cycles.toFixed(2)}</td>
                    <td>{m.test.mae_cycles.toFixed(2)}</td>
                    <td>{m.capped_test.rmse_cycles.toFixed(2)}</td>
                    <td>{(m.interval_coverage * 100).toFixed(1)}%</td>
                    <td>{m.mean_interval_width_cycles.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <label>
            寿命对照模型
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {Object.keys(study.models).map((name) => (
                <option key={name} value={name}>
                  {name === "ridge" ? "岭回归" : "极端随机树"}
                </option>
              ))}
            </select>
          </label>
          <Chart label="留出发动机剩余寿命对照" option={option} />
          <label>
            留出发动机
            <input
              type="number"
              min={1}
              max={100}
              value={unit}
              onChange={(e) => setUnit(Number(e.target.value))}
            />
          </label>
          {record && (
            <div className="kpis">
              <div>
                已观测周期<strong>{record.observed_cycles}</strong>
              </div>
              <div>
                预测剩余周期<strong>{record.prediction.toFixed(1)}</strong>
              </div>
              <div>
                校准区间
                <strong>
                  {record.lower.toFixed(1)}–{record.upper.toFixed(1)}
                </strong>
              </div>
            </div>
          )}
          <p className="muted">
            ONNX 对照 64 个留出窗口 · 最大误差{" "}
            {study.manifest.parity_max_abs_error.toExponential(2)} · 版本{" "}
            {study.run_id}
          </p>
        </>
      )}
    </section>
  );
}

type VisionStudy = {
  output_grid?: number;
  operating_points?: {
    calibration_alpha: number;
    threshold: number;
    false_positives: number;
    false_negatives: number;
  }[];
  run_id: string;
  counts: Record<string, number>;
  threshold: number;
  metrics: {
    image_auroc: number;
    image_ap: number;
    false_positives: number;
    normal_images: number;
    false_negatives: number;
    anomalous_images: number;
    pixel_auroc_14x14?: number;
    pixel_auroc?: number;
  };
  baseline: {
    image_auroc: number;
    image_ap: number;
    false_positives: number;
    false_negatives: number;
  };
  records: {
    id: number;
    file: string;
    truth: boolean;
    flagged: boolean;
    score: number;
  }[];
  latency_ms: { p50: number; p95: number };
};
export function VisionLab() {
  const [study, setStudy] = useState<VisionStudy | null>(null),
    [recordId, setRecordId] = useState(0),
    [error, setError] = useState("");
  useEffect(() => {
    const abort = new AbortController();
    request<VisionStudy | null>("/api/benchmarks/vision-study", {
      signal: abort.signal,
    })
      .then(setStudy)
      .catch((e) => {
        if (!abort.signal.aborted) setError(String(e));
      });
    return () => abort.abort();
  }, []);
  const record = study?.records[recordId];
  return (
    <section className="panel">
      <div className="section-heading">
        <div>
          <h2>视觉异常 · 局部证据</h2>
        </div>
        {study && (
          <a
            className="primary"
            href={`/api/benchmarks/vision-study/download?run_id=${study.run_id}`}
          >
            导出视觉 ONNX
          </a>
        )}
      </div>
      {error && <p role="alert">{error}</p>}
      {!study ? (
        <p role="status">
          尚无视觉实验结果。prepare_visa.py 准备公开图像，study_vision.py
          完成实验。
        </p>
      ) : (
        <>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>方法</th>
                  <th>图像 AUROC</th>
                  <th>图像 AP</th>
                  <th>正常误报</th>
                  <th>异常漏报</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>整图深层特征参考</td>
                  <td>{study.baseline.image_auroc.toFixed(3)}</td>
                  <td>{study.baseline.image_ap.toFixed(3)}</td>
                  <td>{study.baseline.false_positives}</td>
                  <td>{study.baseline.false_negatives}</td>
                </tr>
                <tr>
                  <td>局部 patch 记忆</td>
                  <td>{study.metrics.image_auroc.toFixed(3)}</td>
                  <td>{study.metrics.image_ap.toFixed(3)}</td>
                  <td>
                    {study.metrics.false_positives}/
                    {study.metrics.normal_images}
                  </td>
                  <td>
                    {study.metrics.false_negatives}/
                    {study.metrics.anomalous_images}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="muted">
            训练 {study.counts.train} · 健康验证 {study.counts.validation} ·
            健康校准 {study.counts.calibration} · 正式测试 {study.counts.test} ·
            定位 AUROC{" "}
            {(
              study.metrics.pixel_auroc ??
              study.metrics.pixel_auroc_14x14 ??
              0
            ).toFixed(3)}{" "}
            ({study.output_grid ?? 14}×{study.output_grid ?? 14})
          </p>
          <div className="inline-actions">
            <details>
              <summary>阈值与漏检对照</summary>
              <table>
                <thead>
                  <tr>
                    <th>校准尾部水平</th>
                    <th>阈值</th>
                    <th>测试正常误报 / 100</th>
                    <th>测试异常漏报 / 100</th>
                  </tr>
                </thead>
                <tbody>
                  {study.operating_points?.map((p) => (
                    <tr key={p.calibration_alpha}>
                      <td>{(p.calibration_alpha * 100).toFixed(0)}%</td>
                      <td>{p.threshold.toFixed(3)}</td>
                      <td>{p.false_positives}</td>
                      <td>{p.false_negatives}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
            <label>
              测试图像
              <select
                value={recordId}
                onChange={(e) => setRecordId(Number(e.target.value))}
              >
                {study.records.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.id} · {r.truth ? "异常" : "正常"} ·{" "}
                    {r.flagged ? "超阈值" : "未超限"}
                  </option>
                ))}
              </select>
            </label>
            <button
              onClick={() => {
                const next =
                  study.records.find(
                    (r) => r.id > recordId && r.truth !== r.flagged,
                  ) ?? study.records.find((r) => r.truth !== r.flagged);
                if (next) setRecordId(next.id);
              }}
            >
              定位误判图像
            </button>
          </div>
          {record && (
            <>
              <div className="vision-pair">
                <figure>
                  <img
                    alt={`VisA 原始测试图 ${recordId}`}
                    src={`/api/benchmarks/vision/image/${recordId}?run_id=${study.run_id}&layer=raw`}
                  />
                  <figcaption>
                    原图 · {record.file.split("/").at(-1)}
                  </figcaption>
                </figure>
                <figure>
                  <img
                    alt={`视觉异常热力图 ${recordId}`}
                    src={`/api/benchmarks/vision/image/${recordId}?run_id=${study.run_id}&layer=heat`}
                  />
                  <figcaption>局部距离 · 颜色尺度固定于健康校准集</figcaption>
                </figure>
              </div>
              <p>
                分数 {record.score.toFixed(3)} / 阈值{" "}
                {study.threshold.toFixed(3)} ·{" "}
                {record.flagged ? "需检查" : "未超阈值"} · 数据集标签：
                {record.truth ? "异常" : "正常"}
              </p>
            </>
          )}
          <p className="muted">
            CPU 单图 p50 {study.latency_ms.p50.toFixed(1)} ms / p95{" "}
            {study.latency_ms.p95.toFixed(1)} ms · 图像：Zou 等，VisA，CC BY 4.0
            · 版本 {study.run_id}
          </p>
        </>
      )}
    </section>
  );
}
