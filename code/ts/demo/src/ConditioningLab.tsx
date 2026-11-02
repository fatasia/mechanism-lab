import { useEffect, useState } from "react";
import { request } from "./domain";

type Metric = {
  false_positives: number;
  false_negatives: number;
  macro_f1?: number;
  normal?: number;
  faults?: number;
  normal_cycles?: number;
  test_positive_cycles?: number;
};
type Entry = { test: Metric };
type ContextStudy = {
  run_id: string;
  protocols: Record<
    string,
    {
      selected: string | null;
      source_summary: Record<string, { macro_f1: number; normal_fpr: number }>;
      target: Record<string, Record<string, Metric>>;
    }
  >;
};
type ExternalStudy = {
  run_id: string;
  folds: Record<string, { models: Record<string, Entry> }>;
};
type Study = {
  run_id: string;
  valve_selected: string | null;
  valve: Record<string, Record<string, Entry>>;
  actuator: Record<
    string,
    {
      selected: string;
      commissioning: number[];
      arms: Record<string, Record<string, Entry>>;
    }
  >;
};
const names: Record<string, string> = {
  frozen_ae: "原波形 AE",
  timing_rules: "时序规则",
  timing_rf: "时序机理 RF",
  pressure_rf: "时序＋压力 RF",
  pressure_ae_rf: "时序＋压力＋AE RF",
  raw_rf: "原始特征 RF",
  conditioned_rf: "工况健康参考 RF",
};

export function ConditioningLab() {
  const [study, setStudy] = useState<Study | null>(null);
  const [error, setError] = useState("");
  const [seed, setSeed] = useState("42");
  const [protocol, setProtocol] = useState("held-out-profile");
  const [external, setExternal] = useState<ExternalStudy | null>(null);
  const [externalError, setExternalError] = useState("");
  const [selection, setSelection] = useState<ContextStudy | null>(null);
  const [selectionError, setSelectionError] = useState("");
  useEffect(() => {
    request<ContextStudy | null>("/api/benchmarks/context-selection-study")
      .then(setSelection)
      .catch((e) => setSelectionError(String(e)));
    request<Study | null>("/api/benchmarks/conditioning-study")
      .then(setStudy)
      .catch((e) => setError(String(e)));
    request<ExternalStudy | null>("/api/benchmarks/external-actuator-study")
      .then(setExternal)
      .catch((e) => setExternalError(String(e)));
  }, []);
  if (error) return <p role="alert">{error}</p>;
  if (!study) return <p role="status">专项实验尚未完成或正在读取。</p>;
  const transfer = study.actuator[protocol];
  const table = (arms: Record<string, Record<string, Entry>>) => (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>方法</th>
            <th>正常 / 故障</th>
            <th>误报</th>
            <th>漏报</th>
            <th>宏 F1</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(arms).map(([name, seeds]) => {
            const row = (seeds[seed] ?? seeds["42"]).test;
            return (
              <tr key={name}>
                <td>{names[name]}</td>
                <td>
                  {row.normal ?? row.normal_cycles} /{" "}
                  {row.faults ?? row.test_positive_cycles}
                </td>
                <td>{row.false_positives}</td>
                <td>{row.false_negatives}</td>
                <td>{row.macro_f1?.toFixed(4) ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
  return (
    <section className="panel">
      <div className="section-heading">
        <div>
          <h2>漏报与跨工况误报</h2>
        </div>
        <label>
          随机种子{" "}
          <select value={seed} onChange={(e) => setSeed(e.target.value)}>
            {[42, 43, 44].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
      </div>
      <h3>阀门：用已标注故障训练的机理模型</h3>
      <p>验证集选择：{names[study.valve_selected ?? ""] ?? "无合格模型"}</p>
      {table(study.valve)}
      <h3>执行器：健康投运参考的收益与代价</h3>
      <label>
        评价协议{" "}
        <select value={protocol} onChange={(e) => setProtocol(e.target.value)}>
          <option value="held-out-profile">轨迹留出</option>
          <option value="held-out-load">负载留出</option>
          <option value="record-holdout">已覆盖工况</option>
        </select>
      </label>
      <p>
        目标健康参考 {transfer.commissioning.length} 条 · 源工况验证选择：
        {names[transfer.selected]}
      </p>
      {table(transfer.arms)}
      <h3>源域整工况选型：绝对残差消融</h3>
      {selectionError ? (
        <p role="alert">{selectionError}</p>
      ) : selection ? (
        <>
          <p>正常误报预算 5%</p>
          {Object.entries(selection.protocols).map(([name, experiment]) => (
            <details key={name} open={name === protocol}>
              <summary>
                {name === "held-out-load" ? "负载留出" : "轨迹留出"} ·{" "}
                {experiment.selected ?? "未通过准入"}
              </summary>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>表示</th>
                      <th>源验证误报率</th>
                      <th>目标误报</th>
                      <th>目标漏报</th>
                      <th>目标宏F1</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(experiment.target).map(([arm, seeds]) => (
                      <tr key={arm}>
                        <td>
                          {
                            {
                              raw: "原始特征",
                              centered: "工况中心化",
                              absolute_residual: "绝对工况残差",
                            }[arm]
                          }
                        </td>
                        <td>
                          {(
                            experiment.source_summary[arm].normal_fpr * 100
                          ).toFixed(1)}
                          %
                        </td>
                        <td>{seeds[seed].false_positives}</td>
                        <td>{seeds[seed].false_negatives}</td>
                        <td>{seeds[seed].macro_f1?.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          ))}
          <p>
            版本 {selection.run_id} · 负载测试 正常 16 / 故障 240 · 轨迹测试
            正常 24 / 故障 360
          </p>
        </>
      ) : (
        <p role="status">工况选型结果正在读取。</p>
      )}
      <h3>新公开数据：六台液压执行器</h3>
      <p>598 条记录 · 健康设备逐台留出</p>
      {externalError ? (
        <p role="alert">{externalError}</p>
      ) : external ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>留出健康设备</th>
                <th>正常 / 故障</th>
                <th>IF 误报</th>
                <th>IF 漏报</th>
                <th>稳健统计误报 / 漏报</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(external.folds).map(([device, fold]) => {
                const m = fold.models[`iforest_${seed}`].test;
                const r = fold.models.robust.test;
                return (
                  <tr key={device}>
                    <td>{device}</td>
                    <td>
                      {m.normal_cycles} / {m.test_positive_cycles}
                    </td>
                    <td>{m.false_positives}</td>
                    <td>{m.false_negatives}</td>
                    <td>
                      {r.false_positives} / {r.false_negatives}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p>版本 {external.run_id}</p>
        </div>
      ) : (
        <p role="status">外部设备基准尚未完成或正在读取。</p>
      )}
      <details>
        <summary>版本和模型包</summary>
        <p>{study.run_id}</p>
        <a
          href={`/api/benchmarks/conditioning-study/download?run_id=${study.run_id}`}
        >
          下载专项模型与协议
        </a>
      </details>
    </section>
  );
}
