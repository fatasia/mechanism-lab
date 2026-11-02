import { useMemo, useState } from "react";
import { Chart, lines, palette } from "./Chart";
import { Experiment, Model, algorithms } from "./domain";
import { ValveStudy } from "./ValveStudy";

export function Experiments({
  experiment,
  model,
  onSelect,
  onTrain,
  running,
}: {
  experiment: Experiment;
  model: Model;
  onSelect: (cycle: number) => void;
  onTrain: (profile: string) => void;
  running: boolean;
}) {
  const [profile, setProfile] = useState(
    experiment.task === "valve" ? "valve-switch-100hz" : "process-10hz",
  );
  const option = useMemo(
    () =>
      lines(
        experiment.history.map((h) => h.epoch),
        [
          {
            name: "训练 Huber",
            data: experiment.history.map((h) => h.train),
            color: palette().primary,
          },
          {
            name: "健康验证 Huber",
            data: experiment.history.map((h) => h.validation),
            color: palette().secondary,
          },
        ],
        undefined,
        "训练轮数",
      ),
    [experiment.history],
  );
  const metric = experiment.models[model];
  return (
    <>
      {experiment.task === "valve" && <ValveStudy />}
      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>固定协议 · 四模型对照</h2>
          </div>
          <a
            className="primary"
            href={`/api/download/${experiment.task}?run_id=${experiment.run_id}`}
          >
            下载本版本模型包 ↓
          </a>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>算法</th>
                <th>测试 AP</th>
                <th>精确率</th>
                <th>召回率</th>
                <th>误报 / 千正常周期</th>
                <th>ONNX 数值差异</th>
              </tr>
            </thead>
            <tbody>
              {(
                Object.entries(experiment.models) as [
                  Model,
                  Experiment["models"][Model],
                ][]
              ).map(([k, m]) => (
                <tr key={k}>
                  <td>
                    {k === "ae" && experiment.architecture === "dense"
                      ? "全局瓶颈 AE"
                      : algorithms[k]}
                  </td>
                  <td>{m.test_pr_auc?.toFixed(3) ?? "不可计算"}</td>
                  <td>
                    {m.precision === null
                      ? "—"
                      : (m.precision * 100).toFixed(1) + "%"}
                  </td>
                  <td>
                    {m.recall === null
                      ? "—"
                      : (m.recall * 100).toFixed(1) + "%"}
                  </td>
                  <td>{m.false_positive_per_1000?.toFixed(1) ?? "—"}</td>
                  <td>
                    {m.onnx_max_abs_error?.toExponential(2) ?? "图外统计"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="panel">
        <h2>{algorithms[model]} · 测试错误定位</h2>
        <div className="two-col">
          <div>
            <p>误报 {metric.confusion[0][1]} 个周期</p>
            <div className="case-links">
              {metric.false_positive_ids?.slice(0, 16).map((i) => (
                <button key={i} onClick={() => onSelect(i)}>
                  #{i}
                </button>
              ))}
            </div>
          </div>
          <div>
            <p>漏报 {metric.confusion[1][0]} 个周期</p>
            <div className="case-links">
              {metric.false_negative_ids?.slice(0, 16).map((i) => (
                <button key={i} onClick={() => onSelect(i)}>
                  #{i}
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>
      <section className="panel">
        <h2>AE 训练与早停验证</h2>
        <Chart label="自编码器训练损失" option={option} />
      </section>
      <section className="panel provenance">
        <h2>来源与复现</h2>
        <p>{experiment.source}</p>
        <dl>
          <dt>数据版本</dt>
          <dd className="hash">{experiment.data_sha256}</dd>
          <dt>代码指纹</dt>
          <dd className="hash">{experiment.source_hash ?? "旧版未记录"}</dd>
          <dt>信号配置</dt>
          <dd>
            {experiment.profile} · {experiment.sample_rate_hz} Hz
          </dd>
        </dl>
        <div className="training-controls">
          <select
            aria-label="训练信号配置"
            value={profile}
            onChange={(e) => setProfile(e.target.value)}
          >
            <option value="process-10hz">压力 / 功率 / 流量 · 10 Hz</option>
            {experiment.task === "valve" && (
              <option value="valve-switch-100hz">
                阀门换向压力 · 100 Hz · 全局瓶颈
              </option>
            )}
            <option value="thermal-1hz">温度 / 低频振动 · 1 Hz</option>
          </select>
          <button disabled={running} onClick={() => onTrain(profile)}>
            新建固定实验
          </button>
        </div>
      </section>
    </>
  );
}
