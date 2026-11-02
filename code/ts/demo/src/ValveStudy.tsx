import { useEffect, useState } from "react";
import { Metric, request } from "./domain";

type StudyMetric = Pick<Metric, "test_pr_auc" | "recall"> & {
  false_positives: number;
  false_negatives: number;
};
interface Arm {
  seeds: Record<
    string,
    { test: StudyMetric; original_operating_point?: StudyMetric }
  >;
  summary?: Record<string, { mean: number; min: number; max: number }>;
  onnx_max_abs_error?: number;
}
interface Study {
  run_id: string;
  arms: Record<string, Arm>;
  selected_waveform: { arm: string };
  protocol: { seeds: number[]; baseline_run_id: string };
}
const armNames: Record<string, string> = {
  archived_full_cycle: "原整周期 AE · 重校准",
  roi_conv_stable: "仅截取换向窗口",
  roi_conv_all: "窗口 + 健康覆盖",
  roi_dense_all: "窗口 + 全局瓶颈 AE",
  timing_ae: "换向时序特征 AE",
  timing_rules: "换向时序 · 稳健规则",
};

/** 研究对照独立于普通模型表，不将不同输入的网络冒充同一模型。 */
export function ValveStudy() {
  const [study, setStudy] = useState<Study | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    request<Study | null>("/api/valve-study", { signal: controller.signal })
      .then(setStudy)
      .catch((e) => {
        if (!controller.signal.aborted) setError(String(e));
      });
    return () => controller.abort();
  }, []);
  if (error)
    return (
      <section className="panel">
        <p role="alert">阀门研究读取失败：{error}</p>
      </section>
    );
  if (!study) return null;
  const original =
    study.arms.archived_full_cycle.seeds["42"].original_operating_point!;
  return (
    <section className="panel">
      <div className="section-heading">
        <div>
          <h2>阀门专项 · 机理与 AE 消融</h2>
          <p>测试集 112 正常 · 152 异常周期</p>
        </div>
        <a
          className="primary"
          href={`/api/valve-study/download?run_id=${study.run_id}`}
        >
          下载对照实验包 ↓
        </a>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>对照路线</th>
              <th>AP · 种子42</th>
              <th>误报</th>
              <th>漏报</th>
              <th>召回率</th>
              <th>三种子误报 / 漏报范围</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>原整周期 AE · 原阈值</td>
              <td>{original.test_pr_auc?.toFixed(3)}</td>
              <td>{original.false_positives}</td>
              <td>{original.false_negatives}</td>
              <td>
                {original.recall === null
                  ? "—"
                  : (original.recall * 100).toFixed(1) + "%"}
              </td>
              <td>历史单次</td>
            </tr>
            {Object.entries(study.arms).map(([key, arm]) => {
              const m = arm.seeds["42"].test;
              return (
                <tr key={key}>
                  <td>
                    {armNames[key]}
                    {key === study.selected_waveform.arm ? " · 已选" : ""}
                  </td>
                  <td>{m.test_pr_auc?.toFixed(3)}</td>
                  <td>{m.false_positives}</td>
                  <td>{m.false_negatives}</td>
                  <td>
                    {m.recall === null
                      ? "—"
                      : (m.recall * 100).toFixed(1) + "%"}
                  </td>
                  <td>
                    {arm.summary
                      ? `${arm.summary.false_positives.min}～${arm.summary.false_positives.max} / ${arm.summary.false_negatives.min}～${arm.summary.false_negatives.max}`
                      : "确定性规则 / 历史基线"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
