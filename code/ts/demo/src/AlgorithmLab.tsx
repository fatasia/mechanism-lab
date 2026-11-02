import { useEffect, useMemo, useState } from "react";
import { Chart, lines, palette } from "./Chart";
import { request } from "./domain";
import { ReferenceLab } from "./ReferenceLab";
import { LifeLab, VisionLab } from "./DomainLabs";
import { ConditioningLab } from "./ConditioningLab";
import { ReportLab } from "./ReportLab";

type Metric = {
  confusion: number[][];
  macro_f1: number;
  accuracy: number;
  records: number;
  false_positives: number;
  false_negatives: number;
  candidate_set_coverage?: number;
  singleton_fraction?: number;
  empty_fraction?: number;
  errors: { record: number; truth: number; predicted: number }[];
};
type Arm = {
  seeds: Record<string, { validation: Metric; test: Metric }>;
  test_f1: { mean: number; min: number; max: number };
};
type Study = {
  run_id: string;
  records: number;
  cycles: number;
  classes: string[];
  protocols: Record<
    string,
    {
      counts: Record<string, number>;
      selected: { arm: string };
      arms: Record<string, Arm>;
    }
  >;
};
type RecordData = {
  record: {
    id: number;
    label: number;
    file: string;
    key: string;
    load: string;
    profile: string;
  };
  run_id: string;
  partition: string;
  predicted: number;
  model: string;
  scores: number[];
  values: number[][];
  sample_rate_hz: number;
  channels: string[];
  reference_alignment: boolean;
  mechanism: Record<string, number>;
  candidate_set: boolean[] | null;
};
const arms: Record<string, string> = {
  mechanism_rf: "机理特征 · 随机森林",
  current_only_rf: "电流＋指令参考",
  error_only_rf: "误差＋指令参考",
  random_kernels: "随机多尺度卷积＋岭分类",
  multiscale_fusion: "多尺度残差网络＋幅值融合",
  multiscale_ensemble: "三种子网络集成＋记录级校准",
};
const classNames = ["正常", "间隙", "润滑不足", "剥落"];
const percent = (value: number) => (value * 100).toFixed(1) + "%";

function ActuatorLab() {
  const [kind, setKind] = useState("actuator-ensemble-study"),
    [protocol, setProtocol] = useState("record-holdout"),
    [study, setStudy] = useState<Study | null>(null),
    [record, setRecord] = useState(0),
    [cycle, setCycle] = useState(0),
    [channel, setChannel] = useState(1),
    [arm, setArm] = useState(""),
    [example, setExample] = useState<RecordData | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    const ctrl = new AbortController();
    setStudy(null);
    setExample(null);
    setError("");
    request<Study | null>(`/api/benchmarks/${kind}`, { signal: ctrl.signal })
      .then(setStudy)
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(String(e));
      });
    return () => ctrl.abort();
  }, [kind]);
  const group = study?.protocols[protocol];
  useEffect(() => {
    if (group) {
      setArm(group.selected.arm);
      setRecord(
        group.arms[group.selected.arm].seeds["42"].test.errors[0]?.record ?? 0,
      );
    }
  }, [group]);
  useEffect(() => {
    if (!study || !arm) return;
    const ctrl = new AbortController();
    setExample(null);
    setError("");
    request<RecordData>(
      `/api/benchmarks/${kind}/record/${record}?protocol=${protocol}&cycle=${cycle}&arm=${arm}&run_id=${study.run_id}`,
      { signal: ctrl.signal },
    )
      .then(setExample)
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(String(e));
      });
    return () => ctrl.abort();
  }, [study, kind, record, protocol, cycle, arm]);
  const option = useMemo(
    () =>
      example
        ? lines(
            example.values.map((_, i) => i / 25),
            [
              {
                name: example.channels[channel],
                data: example.values.map((row) => row[channel]),
                color: palette().primary,
              },
            ],
            undefined,
            "周期内时间 / 秒",
            example.channels[channel].split(" / ")[1],
          )
        : {},
    [example, channel],
  );
  return (
    <>
      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>执行器 · 诊断与适用域</h2>
          </div>
          {study && (
            <a
              className="primary"
              href={`/api/benchmarks/${kind}/download?run_id=${study.run_id}`}
            >
              下载模型与协议 ↓
            </a>
          )}
        </div>
        <div className="lab-controls">
          <label>
            评估路线
            <select
              aria-label="参考策略"
              value={kind}
              onChange={(e) => {
                setProtocol(
                  e.target.value.includes("ensemble") ||
                    e.target.value.includes("release")
                    ? "record-holdout"
                    : "held-out-load",
                );
                setKind(e.target.value);
              }}
            >
              <option value="actuator-ensemble-study">
                覆盖工况 · 集成与校准
              </option>
              <option value="actuator-release-study">
                覆盖工况 · 三种子消融
              </option>
              <option value="actuator-transfer-study">
                分工况健康投运参考
              </option>
              <option value="actuator-study">无目标工况参考</option>
            </select>
          </label>
          <label>
            留出协议
            <select
              aria-label="留出协议"
              value={protocol}
              onChange={(e) => setProtocol(e.target.value)}
            >
              {kind.includes("ensemble") || kind.includes("release") ? (
                <option value="record-holdout">
                  完整记录留出 · 已覆盖工况
                </option>
              ) : (
                <>
                  <option value="held-out-load">留出 −40 kgf 负载</option>
                  <option value="held-out-profile">留出正弦运动轨迹</option>
                </>
              )}
            </select>
          </label>
        </div>
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
        {!study ? (
          <p role="status">实验尚未完成或正在读取。</p>
        ) : (
          group && (
            <>
              <p>
                {study.records} 份记录 · {study.cycles} 个周期 · 留出{" "}
                {group.counts.test} 份
              </p>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>方法</th>
                      <th>验证宏 F1</th>
                      <th>留出宏 F1 · 均值</th>
                      <th>单次集成 / 三种子范围</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(group.arms).map(([key, a]) => (
                      <tr key={key}>
                        <td>
                          <button
                            className="text-button"
                            onClick={() => setArm(key)}
                          >
                            {arms[key]}
                            {group.selected.arm === key
                              ? key === "multiscale_ensemble"
                                ? " · 固定集成"
                                : " · 验证集选定"
                              : ""}
                          </button>
                        </td>
                        <td>
                          {percent(
                            Object.values(a.seeds).reduce(
                              (s, x) => s + x.validation.macro_f1,
                              0,
                            ) / Object.keys(a.seeds).length,
                          )}
                        </td>
                        <td
                          className={
                            a.test_f1.mean < 0.9 ? "metric-warning" : ""
                          }
                        >
                          {percent(a.test_f1.mean)}
                        </td>
                        <td>
                          {key === "multiscale_ensemble"
                            ? "固定三种子集成"
                            : `${percent(a.test_f1.min)}～${percent(a.test_f1.max)}`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {group.arms.multiscale_ensemble && (
                <>
                  <p className="metric-warning">
                    正常留出{" "}
                    {group.arms.multiscale_ensemble.seeds[
                      "42"
                    ].test.confusion[0].reduce((sum, n) => sum + n, 0)}{" "}
                    条 · 误报{" "}
                    {
                      group.arms.multiscale_ensemble.seeds["42"].test
                        .false_positives
                    }{" "}
                    条
                  </p>
                  <p>
                    候选集合覆盖{" "}
                    {percent(
                      group.arms.multiscale_ensemble.seeds["42"].test
                        .candidate_set_coverage!,
                    )}{" "}
                    · 单一候选{" "}
                    {percent(
                      group.arms.multiscale_ensemble.seeds["42"].test
                        .singleton_fraction!,
                    )}
                  </p>
                </>
              )}
            </>
          )
        )}
      </section>
      {study && group && (
        <section className="panel">
          <div className="section-heading">
            <div>
              <h2>逐记录诊断与误判定位</h2>
            </div>
            <button
              onClick={() => {
                setRecord(
                  group.arms[arm]?.seeds["42"].test.errors[0]?.record ?? 0,
                );
                setCycle(0);
              }}
            >
              定位首个误判
            </button>
          </div>
          <div className="lab-controls">
            <label>
              诊断方法
              <select
                aria-label="诊断方法"
                value={arm}
                onChange={(e) => setArm(e.target.value)}
              >
                {Object.keys(group.arms).map((id) => (
                  <option key={id} value={id}>
                    {arms[id]}
                  </option>
                ))}
              </select>
            </label>
            <label>
              记录编号
              <input
                aria-label="记录编号"
                type="number"
                min={0}
                max={study.records - 1}
                value={record}
                onChange={(e) => setRecord(Number(e.target.value))}
              />
            </label>
            <label>
              动作周期
              <select
                aria-label="动作周期"
                value={cycle}
                onChange={(e) => setCycle(Number(e.target.value))}
              >
                {[0, 1, 2, 3, 4].map((i) => (
                  <option key={i} value={i}>
                    第 {i + 1} 周期
                  </option>
                ))}
              </select>
            </label>
            <label>
              信号
              <select
                aria-label="执行器信号"
                value={channel}
                onChange={(e) => setChannel(Number(e.target.value))}
              >
                {["指令位置", "跟随误差", "电流"].map((n, i) => (
                  <option key={n} value={i}>
                    {n}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {example ? (
            <>
              <div className="evidence-strip">
                <strong>
                  {example.candidate_set
                    ? `候选：${
                        example.candidate_set
                          .map((v, i) => (v ? classNames[i] : null))
                          .filter(Boolean)
                          .join(" / ") || "空集合 · 需复核"
                      }`
                    : `首位候选：${classNames[example.predicted]}`}
                </strong>
                <span>数据标签：{classNames[example.record.label]}</span>
                <span>
                  {example.partition} · {example.record.file} ·{" "}
                  {example.record.key}
                </span>
              </div>
              <Chart label="执行器实测动作信号" option={option} />
              <details>
                <summary>原单位机理特征与分类分数</summary>
                <div className="table-wrap">
                  <table>
                    <tbody>
                      {Object.entries(example.mechanism).map(([n, v]) => (
                        <tr key={n}>
                          <th>{n}</th>
                          <td>{v.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p>
                  {example.scores
                    .map((s, i) => `${classNames[i]} ${s.toFixed(3)}`)
                    .join(" · ")}
                </p>
              </details>
            </>
          ) : (
            <p role="status">读取记录…</p>
          )}
        </section>
      )}
    </>
  );
}

interface Rotating {
  run_id: string;
  records: number;
  top_candidate_matches: number;
  duplicate_drive_end_signals: string[][];
  examples: {
    file: string;
    expected: string;
    top_candidate: string;
    rpm: number;
    frequency_hz: number[];
    envelope_psd: number[];
    candidates: {
      candidate: string;
      frequency_hz: number;
      evidence_score: number;
      peaks: { observed_hz: number }[];
    }[];
  }[];
}
const bearingNames: Record<string, string> = {
  inner_ring: "内圈",
  outer_ring: "外圈",
  rolling_element: "滚动体",
  cage: "保持架",
};
function RotatingLab() {
  const [study, setStudy] = useState<Rotating | null>(null),
    [index, setIndex] = useState(0),
    [error, setError] = useState("");
  useEffect(() => {
    request<Rotating | null>("/api/benchmarks/rotating-study")
      .then(setStudy)
      .catch((e) => setError(String(e)));
  }, []);
  const example = study?.examples[index];
  const option = useMemo(
    () =>
      example
        ? lines(
            example.frequency_hz,
            [
              {
                name: "包络谱",
                data: example.envelope_psd,
                color: palette().primary,
              },
            ],
            undefined,
            "频率 / Hz",
            "功率谱密度",
          )
        : {},
    [example],
  );
  return (
    <section className="panel">
      <div className="section-heading">
        <div>
          <h2>轴承 · 包络频谱证据</h2>
        </div>
        {study && (
          <a
            href={`/api/benchmarks/rotating-study/download?run_id=${study.run_id}`}
          >
            下载分析记录 ↓
          </a>
        )}
      </div>
      {error && <p role="alert">{error}</p>}
      {study && example ? (
        <>
          <p>
            {study.records} 份故障记录 · 首位候选匹配{" "}
            {study.top_candidate_matches} 份
          </p>
          <label>
            实测文件
            <select
              aria-label="轴承文件"
              value={index}
              onChange={(e) => setIndex(Number(e.target.value))}
            >
              {study.examples.map((e, i) => (
                <option key={e.file} value={i}>
                  {e.file} · 标签 {bearingNames[e.expected]}
                </option>
              ))}
            </select>
          </label>
          <Chart label="轴承包络功率谱" option={option} />
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>候选部位</th>
                  <th>参考频率 / Hz</th>
                  <th>一阶邻域峰 / Hz</th>
                  <th>谐波证据分数</th>
                </tr>
              </thead>
              <tbody>
                {example.candidates.map((c) => (
                  <tr key={c.candidate}>
                    <td>{bearingNames[c.candidate]}</td>
                    <td>{c.frequency_hz.toFixed(2)}</td>
                    <td>{c.peaks[0]?.observed_hz.toFixed(2)}</td>
                    <td>{c.evidence_score.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>转速 {example.rpm} rpm</p>
          <p className="metric-warning">
            重复驱动端信号：
            {study.duplicate_drive_end_signals
              .map((pair) => pair.join(" / "))
              .join("，")}
          </p>
        </>
      ) : (
        <p role="status">读取频谱实验…</p>
      )}
    </section>
  );
}

interface Metro {
  run_id: string;
  source_rows: number;
  median_interval_seconds: number;
  quality_excluded: number[];
  models: Record<
    string,
    {
      detected_events: number;
      reported_events: number;
      alarm_events: number;
      unmatched_events: number;
      unmatched_per_1000_logged_hours: number;
      failures: {
        start: string;
        detected: boolean;
        lead_hours: number | null;
      }[];
    }
  >;
  timeline: {
    end: number[];
    reservoir_bar: number[];
    ae_scores: number[];
    ae_end: number[];
  };
}
function MetroLab() {
  const [study, setStudy] = useState<Metro | null>(null),
    [layer, setLayer] = useState("pressure"),
    [error, setError] = useState("");
  useEffect(() => {
    request<Metro | null>("/api/benchmarks/metro-study")
      .then(setStudy)
      .catch((e) => setError(String(e)));
  }, []);
  const option = useMemo(
    () =>
      study
        ? lines(
            layer === "pressure" ? study.timeline.end : study.timeline.ae_end,
            [
              {
                name: layer === "pressure" ? "储气压力" : "序列 AE 分数",
                data:
                  layer === "pressure"
                    ? study.timeline.reservoir_bar
                    : study.timeline.ae_scores,
                color: palette().primary,
              },
            ],
            undefined,
            "源记录日期",
            layer === "pressure" ? "bar" : "异常分数",
            (v) => new Date(Number(v) * 1000).toISOString().slice(5, 10),
          )
        : {},
    [study, layer],
  );
  return (
    <section className="panel">
      <div className="section-heading">
        <div>
          <h2>压缩机 · 在役时间外回测</h2>
        </div>
        {study && (
          <a
            href={`/api/benchmarks/metro-study/download?run_id=${study.run_id}`}
          >
            下载模型与回测 ↓
          </a>
        )}
      </div>
      {error && <p role="alert">{error}</p>}
      {study ? (
        <>
          <p>
            {study.source_rows.toLocaleString()} 条实测记录 · 时间戳中位间隔{" "}
            {study.median_interval_seconds} 秒 · 质量排除{" "}
            {study.quality_excluded.reduce((a, b) => a + b, 0)} 个十分钟窗口
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>模型</th>
                  <th>报告事件检出</th>
                  <th>告警段</th>
                  <th>未匹配告警</th>
                  <th>每千记录小时未匹配</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(study.models).map(([n, m]) => (
                  <tr key={n}>
                    <td>
                      {n
                        .replace("sequence_ae", "序列 AE")
                        .replace("iforest", "孤立森林")
                        .replace("robust", "稳健统计")}
                    </td>
                    <td>
                      {m.detected_events}/{m.reported_events}
                    </td>
                    <td>{m.alarm_events}</td>
                    <td>{m.unmatched_events}</td>
                    <td>{m.unmatched_per_1000_logged_hours.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <label>
            时间图层
            <select
              aria-label="压缩机图层"
              value={layer}
              onChange={(e) => setLayer(e.target.value)}
            >
              <option value="pressure">储气压力</option>
              <option value="score">序列 AE · 种子42</option>
            </select>
          </label>
          <Chart label="压缩机历史趋势" option={option} />
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>维修报告起点</th>
                  <th>序列 AE 检出</th>
                  <th>提前量 / 小时（负数为延后）</th>
                </tr>
              </thead>
              <tbody>
                {study.models.sequence_ae_42.failures.map((f) => (
                  <tr key={f.start}>
                    <td>{f.start.replace("T", " ")}</td>
                    <td>{f.detected ? "匹配到告警" : "未检出"}</td>
                    <td>
                      {f.lead_hours === null ? "—" : f.lead_hours.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <p role="status">回测尚未完成或正在读取。</p>
      )}
    </section>
  );
}

export function AlgorithmLab() {
  const [section, setSection] = useState("执行器");
  return (
    <>
      <div className="lab-tabs">
        {[
          "执行器",
          "漏报与迁移专项",
          "轴承频谱",
          "在役回测",
          "视觉检测",
          "剩余寿命",
          "强基线与导出",
          "独立报告",
        ].map((s) => (
          <button
            className={section === s ? "selected" : ""}
            aria-pressed={section === s}
            key={s}
            onClick={() => setSection(s)}
          >
            {s}
          </button>
        ))}
      </div>
      {section === "执行器" ? (
        <ActuatorLab />
      ) : section === "漏报与迁移专项" ? (
        <ConditioningLab />
      ) : section === "轴承频谱" ? (
        <RotatingLab />
      ) : section === "在役回测" ? (
        <MetroLab />
      ) : section === "视觉检测" ? (
        <VisionLab />
      ) : section === "剩余寿命" ? (
        <LifeLab />
      ) : section === "独立报告" ? (
        <ReportLab />
      ) : (
        <ReferenceLab />
      )}
    </>
  );
}
