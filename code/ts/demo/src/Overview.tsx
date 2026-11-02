import { useMemo, useState } from "react";
import { Chart, lines, palette } from "./Chart";
import { Experiment, Model, Replay, algorithms } from "./domain";

export function Overview({
  experiment,
  model,
  replay,
  onSelect,
}: {
  experiment: Experiment;
  model: Model;
  replay: Replay | null;
  onSelect: (cycle: number) => void;
}) {
  const [filter, setFilter] = useState("all"),
    [offset, setOffset] = useState(0);
  const metric = experiment.models[model];
  const option = useMemo(
    () =>
      replay
        ? lines(
            replay.scores.map((_, i) => i),
            [
              {
                name: "异常分数",
                data: replay.scores.map((v, i) =>
                  filter === "all" || replay.partitions[i] === filter
                    ? v
                    : null,
                ),
                color: palette().primary,
              },
            ],
            replay.threshold,
          )
        : {},
    [replay, filter],
  );
  const events = (replay?.events ?? []).filter(
    (e) =>
      filter === "all" ||
      replay?.partitions.slice(e.start, e.end + 1).includes(filter),
  );
  return (
    <>
      <div className="kpis">
        <div>
          <span>健康训练周期</span>
          <strong>
            {experiment.train_healthy_cycles}
            <small>独立校准 {experiment.calibration_healthy_cycles} 周期</small>
          </strong>
        </div>
        <div>
          <span>测试集平均精确率 AP</span>
          <strong>
            {metric.test_pr_auc?.toFixed(3) ?? "—"}
            <small>
              {model === "ae" && experiment.architecture === "dense"
                ? "全局瓶颈 AE"
                : algorithms[model]}
            </small>
          </strong>
        </div>
        <div>
          <span>测试集召回率</span>
          <strong>
            {metric.recall === null ? "—" : (metric.recall * 100).toFixed(1)}
            <small>%</small>
          </strong>
        </div>
      </div>
      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>周期异常分数</h2>
          </div>
          <select
            aria-label="数据分区"
            value={filter}
            onChange={(e) => {
              setFilter(e.target.value);
              setOffset(0);
            }}
          >
            <option value="all">全部周期</option>
            <option value="test">只看测试集</option>
            <option value="train">只看训练集</option>
            <option value="calibration">只看校准集</option>
          </select>
        </div>
        {replay ? (
          <Chart label="周期异常分数趋势" option={option} onSelect={onSelect} />
        ) : (
          <p className="loading">正在读取实验分数…</p>
        )}
      </section>
      <div className="two-col">
        <section className="panel">
          <h2>测试集判别结果</h2>
          <div className="matrix" aria-label="混淆矩阵">
            <span>
              正常 → 正常<strong>{metric.confusion[0][0]}</strong>
            </span>
            <span className="mistake">
              正常 → 告警<strong>{metric.confusion[0][1]}</strong>
            </span>
            <span className="mistake">
              异常 → 未报<strong>{metric.confusion[1][0]}</strong>
            </span>
            <span>
              异常 → 告警<strong>{metric.confusion[1][1]}</strong>
            </span>
          </div>
          <dl>
            <dt>测试集精确率</dt>
            <dd>
              {metric.precision === null
                ? "—"
                : (metric.precision * 100).toFixed(1) + "%"}
            </dd>
            <dt>每千正常周期误报</dt>
            <dd>
              {metric.false_positive_per_1000?.toFixed(1) ?? "旧版未记录"}
            </dd>
            <dt>阈值</dt>
            <dd>{metric.threshold.toPrecision(5)}</dd>
          </dl>
        </section>
        <section className="panel">
          <div className="section-heading">
            <h2>持续告警片段</h2>
            <small>{events.length} 条</small>
          </div>
          <p>3/5 超限触发</p>
          {events.slice(offset, offset + 5).map((e) => (
            <button
              className="event"
              key={e.start}
              onClick={() => onSelect(e.start)}
            >
              <span className="signal">●</span>周期 {e.start} — {e.end}
              <span>{e.reason === "recovery" ? "恢复" : "段末截断"} ↗</span>
            </button>
          ))}
          {!events.length && <p>当前范围没有持续告警。</p>}
          <div className="pager">
            <button
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - 5))}
            >
              上一页
            </button>
            <button
              disabled={offset + 5 >= events.length}
              onClick={() => setOffset(offset + 5)}
            >
              下一页
            </button>
          </div>
        </section>
      </div>
    </>
  );
}
