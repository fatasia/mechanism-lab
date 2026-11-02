import { useState } from "react";

type Metric = {
  normal: number;
  fault: number;
  false_positives: number;
  false_negatives: number;
  macro_f1: number;
  records?: number;
  supported_records?: number;
};
type Row = {
  arm: string;
  label?: string;
  seed: string;
  kind: string;
  metric: Metric;
  loopRates?: { normal_fpr: number; fault_recall: number; macro_f1: number };
};
type Report = {
  runId: string;
  protocols: Record<string, Row[]>;
  sourceProtocols: Record<string, Row[]>;
  selected: Record<string, string | null>;
  scope?: string;
  synthetic?: boolean;
  classNames?: string[];
};

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw Error("报告结构不符合工况研究格式");
  return value as Record<string, unknown>;
}

function metric(value: unknown): Metric {
  const row = object(value);
  for (const key of [
    "normal",
    "fault",
    "false_positives",
    "false_negatives",
    "macro_f1",
  ]) {
    if (
      typeof row[key] !== "number" ||
      !Number.isFinite(row[key]) ||
      row[key] < 0
    )
      throw Error(`指标无效：${key}`);
  }
  const result = row as Metric;
  if (
    result.false_positives > result.normal ||
    result.false_negatives > result.fault ||
    result.macro_f1 > 1
  )
    throw Error("计数或F1超出有效范围");
  for (const key of [
    "normal",
    "fault",
    "false_positives",
    "false_negatives",
  ] as const)
    if (!Number.isInteger(result[key])) throw Error("记录数必须为整数");
  if ("records" in row || "supported_records" in row) {
    // 旧报告只写总记录数；已提供的正常/故障计数构成实际评分分母。
    if (result.supported_records === undefined && result.records !== undefined)
      result.supported_records = result.normal + result.fault;
    if (
      !Number.isInteger(result.records) ||
      !Number.isInteger(result.supported_records) ||
      result.records! < 0 ||
      result.supported_records! < 0 ||
      result.supported_records! > result.records! ||
      result.normal + result.fault !== result.supported_records
    )
      throw Error("记录覆盖计数不一致");
  }
  return result;
}

function binaryMetric(value: unknown): Metric {
  const source = object(value);
  const m = metric({
    normal: source.normal_cycles,
    fault: source.test_positive_cycles,
    false_positives: source.false_positives,
    false_negatives: source.false_negatives,
    macro_f1: 0,
  });
  const errors = m.false_positives + m.false_negatives;
  const f1 = (correct: number) =>
    2 * correct + errors ? (2 * correct) / (2 * correct + errors) : 0;
  m.macro_f1 =
    (f1(m.normal - m.false_positives) + f1(m.fault - m.false_negatives)) / 2;
  return m;
}

// 只读取报告JSON；不上传文件、不调用训练代码、不切换在线模型。
function parseReport(text: string): Report {
  const input = object(JSON.parse(text));
  if (typeof input.run_id !== "string") throw Error("报告缺少run_id");
  const protocols: Record<string, Row[]> = {};
  const sourceProtocols: Record<string, Row[]> = {};
  const selected: Record<string, string | null> = {};
  if ("scenarios" in input) {
    const specification = object(input.protocol);
    if (!Array.isArray(specification.grid)) throw Error("扰动报告缺少条件网格");
    const conditions = object(input.scenarios);
    const kinds: Record<string, string> = {
      identity: "原始输入",
      offset_bar: "压力偏置",
      gain_fraction: "信号增益",
      shift_samples: "时间平移",
      noise_std_bar: "人工噪声",
    };
    for (const entry of specification.grid) {
      const scenario = object(entry);
      if (
        typeof scenario.name !== "string" ||
        typeof scenario.kind !== "string" ||
        !(scenario.kind in kinds)
      )
        throw Error("扰动条件类型无效");
      const title = kinds[scenario.kind];
      if (
        typeof scenario.value !== "number" ||
        !Number.isFinite(scenario.value)
      )
        throw Error("扰动幅度无效");
      const channels = scenario.kind === "identity" ? [] : scenario.channels;
      if (
        !Array.isArray(channels) ||
        !channels.every((c) => c === 0 || c === 1)
      )
        throw Error("扰动通道无效");
      const units: Record<string, string> = {
        identity: "无扰动",
        offset_bar: `${scenario.value} bar`,
        gain_fraction: `${scenario.value * 100}%`,
        shift_samples: `${scenario.value * 10} ms`,
        noise_std_bar: `标准差 ${scenario.value} bar`,
      };
      (protocols[title] ??= []).push({
        arm: scenario.name,
        label:
          `${channels.map((c) => `PS${c + 1}`).join("＋")} ${units[scenario.kind]}`.trim(),
        seed: "冻结模型",
        kind: "人工扰动 · 保留原标签",
        metric: binaryMetric(conditions[scenario.name]),
      });
    }
    if (!Object.keys(protocols).length) throw Error("扰动网格为空");
    return {
      runId: input.run_id,
      protocols,
      sourceProtocols,
      selected,
      synthetic: true,
      scope:
        "固定模型与阈值的输入敏感性检查；扰动数值不是传感器规格，原标签保持不变的假设未经现场核实。",
    };
  }
  if ("arms" in input && !("protocols" in input)) {
    // 阀门二分类研究以计数派生宏F1；不与执行器四分类指标直接排名。
    for (const [partition, title] of [
      ["test", "测试集"],
      ["validation", "验证集"],
    ]) {
      protocols[title] = Object.entries(object(input.arms)).map(
        ([arm, value]) => {
          const m = binaryMetric(object(value)[partition]);
          return {
            arm,
            seed: "42",
            kind: "二分类 · 宏F1由计数计算",
            metric: m,
          };
        },
      );
      if (!protocols[title].length) throw Error("报告没有评估记录");
    }
    return { runId: input.run_id, protocols, sourceProtocols, selected };
  }
  for (const [name, value] of Object.entries(object(input.protocols))) {
    const partition = object(value);
    if (partition.selected === null || typeof partition.selected === "string")
      selected[name] = partition.selected;
    if (partition.source_summary) {
      sourceProtocols[name] = Object.entries(
        object(partition.source_summary),
      ).flatMap(([arm, seeds]) =>
        // 旧报告只有跨种子均值，没有可还原的计数；不把均值当单次验证。
        Object.entries(object(seeds))
          .filter(([seed]) => /^\d+$/.test(seed))
          .map(([seed, entry]) => ({
            arm,
            seed,
            kind: "源工况留出验证",
            metric: metric(entry),
          })),
      );
    }
    const rows: Row[] = [];
    for (const [arm, seeds] of Object.entries(object(object(value).target))) {
      for (const [seed, entry] of Object.entries(object(seeds))) {
        const result = object(entry);
        let loopRates: Row["loopRates"];
        if (input.loop_weighted && arm in object(input.loop_weighted)) {
          const rates = object(object(object(input.loop_weighted)[arm])[seed]);
          for (const key of ["normal_fpr", "fault_recall", "macro_f1"]) {
            if (
              typeof rates[key] !== "number" ||
              !Number.isFinite(rates[key]) ||
              rates[key] < 0 ||
              rates[key] > 1
            )
              throw Error("回路等权指标无效");
          }
          loopRates = rates as Row["loopRates"];
        }
        if ("gated" in result) {
          rows.push({
            arm,
            seed,
            kind: "源域门槛",
            metric: metric(result.gated),
          });
          rows.push({
            arm,
            seed,
            kind: "直接分类",
            metric: metric(result.argmax),
          });
        } else
          rows.push({
            arm,
            seed,
            kind: "直接分类",
            metric: metric(result),
            loopRates,
          });
      }
    }
    if (!rows.length) throw Error("报告没有目标评估记录");
    protocols[name] = rows;
  }
  if (!Object.keys(protocols).length) throw Error("报告没有工况协议");
  const classNames =
    Array.isArray(input.classes) &&
    input.classes.every((name) => typeof name === "string")
      ? input.classes
      : undefined;
  const specification = input.protocol ? object(input.protocol) : {};
  return {
    runId: input.run_id,
    protocols,
    sourceProtocols,
    selected,
    classNames,
    scope:
      typeof specification.scope === "string" ? specification.scope : undefined,
  };
}

const armNames: Record<string, string> = {
  all_features_binary_rf: "全部特征 · 二分类RF",
  mechanism_binary_rf: "无量纲机理 · 二分类RF",
  raw_features: "原始运动特征",
  commissioned_absolute: "健康参考绝对残差",
  mechanical_residual: "健康运动回归残差",
  mechanical_standardized: "健康运动回归 · 标准化残差",
  pooled_2: "统一尺度 · 2条健康参考",
  pooled_4: "统一尺度 · 4条健康参考",
  pooled_6: "统一尺度 · 6条健康参考",
  shrunk_2: "工况尺度收缩 · 2条健康参考",
  shrunk_4: "工况尺度收缩 · 4条健康参考",
  shrunk_6: "工况尺度收缩 · 6条健康参考",
  vibration_rf: "振动与阶次 · RF",
  current_rf: "三相电流 · RF",
  fusion_no_orders_rf: "振动＋电流 · 无阶次",
  fusion_rf: "振动＋电流＋阶次 · RF",
  spectral_cnn: "全带宽频谱 · CNN",
  order_cnn: "角域谱 · CNN",
  envelope_order_cnn: "包络角域谱 · CNN",
  statistics_rf: "单信号统计 · RF",
  relationships_rf: "控制—响应关系 · RF",
  combined_rf: "统计＋控制关系 · RF",
  combined_logistic: "统计＋控制关系 · 逻辑回归",
  mean_ae: "均值残差 · AE",
  top10_ae: "局部top10残差 · AE",
  local200ms_ae: "200ms局部残差 · AE",
  point_scaled_top10_ae: "逐点尺度局部残差 · AE",
  timing_rule: "换向机理时序",
  mean_and_timing: "均值残差＋机理",
  top10_and_timing: "局部top10＋机理",
  local200ms_and_timing: "200ms局部＋机理",
  point_scaled_top10_and_timing: "逐点尺度局部＋机理",
};

const percent = (value: number, total: number) =>
  total ? `${((100 * value) / total).toFixed(2)}%` : "无样本";

export function ReportLab() {
  const [report, setReport] = useState<Report | null>(null);
  const [protocol, setProtocol] = useState("");
  const [seed, setSeed] = useState("42");
  const [error, setError] = useState("");
  const [partition, setPartition] = useState("target");
  const rows =
    (partition === "source"
      ? report?.sourceProtocols[protocol]
      : report?.protocols[protocol]) ?? [];
  const seeds = [...new Set(rows.map((row) => row.seed))];
  return (
    <section className="panel">
      <h2>独立实验报告</h2>
      <label className="editor">
        选择实验报告
        <input
          type="file"
          accept=".json,application/json"
          aria-label="选择实验报告"
          onChange={async (event) => {
            const file = event.target.files?.[0];
            if (!file) return;
            setError("");
            setReport(null);
            try {
              const next = parseReport(await file.text());
              const first = Object.keys(next.protocols)[0];
              setReport(next);
              setProtocol(first);
              setPartition("target");
              setSeed(next.protocols[first][0].seed);
            } catch (e) {
              setError(String(e));
            }
          }}
        />
      </label>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {report ? (
        <>
          <div className="evidence-strip">
            <strong>报告版本</strong>
            <span>{report.runId}</span>
          </div>
          {report.synthetic && (
            <span className="status" data-tone="info">
              合成扰动
            </span>
          )}
          <div className="lab-controls">
            <label>
              {report.synthetic ? "扰动分组" : "留出协议"}
              <select
                aria-label="报告留出协议"
                value={protocol}
                onChange={(event) => {
                  setProtocol(event.target.value);
                  setPartition("target");
                  setSeed(report.protocols[event.target.value][0].seed);
                }}
              >
                {Object.keys(report.protocols).map((name) => (
                  <option key={name}>{name}</option>
                ))}
              </select>
            </label>
            {Boolean(report.sourceProtocols[protocol]?.length) && (
              <label>
                评价分区
                <select
                  aria-label="报告评价分区"
                  value={partition}
                  onChange={(event) => {
                    const next = event.target.value;
                    setPartition(next);
                    setSeed(
                      (next === "source"
                        ? report.sourceProtocols
                        : report.protocols)[protocol][0].seed,
                    );
                  }}
                >
                  <option value="target">目标工况测试</option>
                  <option value="source">源工况留出验证</option>
                </select>
              </label>
            )}
            <label>
              {report.synthetic ? "模型" : "种子"}
              <select
                aria-label="报告种子"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
              >
                {seeds.map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            </label>
          </div>
          {protocol in report.selected && (
            <p role="status">
              {report.selected[protocol] === null
                ? "源工况准入：无候选通过"
                : `源工况候选：${armNames[report.selected[protocol]!] ?? report.selected[protocol]}`}
            </p>
          )}
          {report.scope && (
            <details>
              <summary>实验适用范围</summary>
              <p>{report.scope}</p>
            </details>
          )}
          {report.classNames && (
            <details>
              <summary>{report.classNames.length}类诊断标签</summary>
              <p>{report.classNames.join(" · ")}</p>
            </details>
          )}
          <div className="table-wrap">
            <table className="report-table">
              <thead>
                <tr>
                  <th>表示 / 判定</th>
                  <th>
                    {report.classNames?.[0] === "非黏滞"
                      ? "非黏滞 / 黏滞"
                      : "正常 / 故障"}
                  </th>
                  <th>误报</th>
                  <th>漏报</th>
                  <th>{report.synthetic ? "原标签误报率" : "正常误报率"}</th>
                  <th>故障召回</th>
                  <th>宏F1</th>
                  <th>支持 / 全部记录</th>
                </tr>
              </thead>
              <tbody>
                {rows
                  .filter((row) => row.seed === seed)
                  .map(({ arm, label, kind, metric: m, loopRates }) => (
                    <tr key={arm + kind}>
                      <td className="report-arm">
                        {label ?? armNames[arm] ?? arm}
                        <small className="block">{kind}</small>
                        {loopRates && (
                          <small className="block">
                            回路等权：误报 {percent(loopRates.normal_fpr, 1)} ·
                            召回 {percent(loopRates.fault_recall, 1)} · F1{" "}
                            {loopRates.macro_f1.toFixed(4)}
                          </small>
                        )}
                      </td>
                      <td
                        data-label={
                          report.classNames?.[0] === "非黏滞"
                            ? "非黏滞 / 黏滞"
                            : "正常 / 故障"
                        }
                      >
                        {m.normal} / {m.fault}
                      </td>
                      <td data-label="误报">{m.false_positives}</td>
                      <td data-label="漏报">{m.false_negatives}</td>
                      <td
                        data-label={
                          report.synthetic ? "原标签误报率" : "正常误报率"
                        }
                      >
                        {percent(m.false_positives, m.normal)}
                      </td>
                      <td data-label="故障召回">
                        {percent(m.fault - m.false_negatives, m.fault)}
                      </td>
                      <td data-label="宏F1">{m.macro_f1.toFixed(4)}</td>
                      <td data-label="支持 / 全部记录">
                        {m.records === undefined
                          ? "—"
                          : `${m.supported_records} / ${m.records}`}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        !error && <p role="status">尚未选择实验报告</p>
      )}
    </section>
  );
}
