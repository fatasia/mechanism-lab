import { useEffect, useMemo, useState } from "react";
import { Chart, lines, palette } from "./Chart";
import { Cycle, Experiment, Replay, request, partitions } from "./domain";

export function Analysis({
  experiment,
  replay,
  cycle,
  setCycle,
}: {
  experiment: Experiment;
  replay: Replay | null;
  cycle: number;
  setCycle: (cycle: number) => void;
}) {
  const [channel, setChannel] = useState(0),
    [wave, setWave] = useState<Cycle | null>(null),
    [playing, setPlaying] = useState(false),
    [attempt, setAttempt] = useState(0),
    [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    setWave(null);
    setError("");
    request<Cycle>(
      `/api/cycle/${experiment.task}/${cycle}?channel=${channel}&run_id=${experiment.run_id}`,
      { signal: controller.signal },
    )
      .then(setWave)
      .catch((e) => {
        if (!controller.signal.aborted) setError(String(e));
      });
    return () => controller.abort();
  }, [experiment.task, experiment.run_id, cycle, channel, attempt]);
  // 等待上一周期数据实际到达，避免慢请求下不断清空图表。
  useEffect(() => {
    if (!playing || !wave) return;
    const timer = setTimeout(() => {
      if (cycle >= experiment.cycle_count - 1) setPlaying(false);
      else setCycle(cycle + 1);
    }, 400);
    return () => clearTimeout(timer);
  }, [playing, wave, cycle, experiment.cycle_count, setCycle]);
  const original = useMemo(
    () =>
      wave
        ? lines(
            wave.actual.map(
              (_, i) =>
                (experiment.window_start_seconds ?? 0) +
                i / experiment.sample_rate_hz,
            ),
            [
              { name: "原始信号", data: wave.actual, color: palette().primary },
              {
                name: "AE 重构",
                data: wave.reconstruction,
                color: palette().secondary,
              },
            ],
            undefined,
            "周期内时间 / 秒",
            experiment.units[channel],
          )
        : {},
    [
      wave,
      experiment.sample_rate_hz,
      experiment.window_start_seconds,
      experiment.units,
      channel,
    ],
  );
  const residual = useMemo(
    () =>
      wave
        ? lines(
            wave.actual.map(
              (_, i) =>
                (experiment.window_start_seconds ?? 0) +
                i / experiment.sample_rate_hz,
            ),
            [
              {
                name: "绝对残差",
                data: wave.actual.map((v, i) =>
                  Math.abs(v - wave.reconstruction[i]),
                ),
                color: palette().danger,
              },
            ],
            undefined,
            "周期内时间 / 秒",
            experiment.units[channel],
          )
        : {},
    [
      wave,
      experiment.sample_rate_hz,
      experiment.window_start_seconds,
      experiment.units,
      channel,
    ],
  );
  return (
    <>
      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>
              周期 {cycle}{" "}
              <span className="partition">
                {partitions[replay?.partitions[cycle] ?? "excluded"]}
              </span>
            </h2>
            <p>
              {wave
                ? wave.healthy
                  ? "部件标签：正常"
                  : "部件标签：异常"
                : error
                  ? "读取失败"
                  : "正在读取信号…"}
            </p>
          </div>
          <select
            aria-label="通道"
            value={channel}
            onChange={(e) => {
              setPlaying(false);
              setChannel(Number(e.target.value));
            }}
          >
            {experiment.channels.map((c, i) => (
              <option value={i} key={c}>
                {c} ·{" "}
                {experiment.units[i] === "degC" ? "°C" : experiment.units[i]}
              </option>
            ))}
          </select>
        </div>
        {error && (
          <p role="alert" className="error">
            {error}
            <button onClick={() => setAttempt((v) => v + 1)}>重试读取</button>
          </p>
        )}
        <Chart label="原始信号与自编码器重构" option={original} />
        <div className="playback">
          <button
            className="primary"
            disabled={!wave}
            onClick={() => setPlaying(!playing)}
          >
            {playing ? "暂停回放" : "播放回放"}
          </button>
          <input
            aria-label="回放周期"
            type="range"
            min={0}
            max={experiment.cycle_count - 1}
            value={cycle}
            onChange={(e) => {
              setPlaying(false);
              setCycle(Number(e.target.value));
            }}
          />
          <input
            className="cycle-number"
            aria-label="周期编号"
            type="number"
            min={0}
            max={experiment.cycle_count - 1}
            value={cycle}
            onChange={(e) => {
              const value = Number(e.target.value);
              if (
                Number.isInteger(value) &&
                value >= 0 &&
                value < experiment.cycle_count
              ) {
                setPlaying(false);
                setCycle(value);
              }
            }}
          />
        </div>
      </section>
      {wave?.mechanism && (
        <section className="panel">
          <h2>换向机理证据</h2>
          <p>
            {wave.mechanism.flagged
              ? "压力响应时序超出参考范围"
              : "压力响应时序在参考范围内"}
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>时序特征</th>
                  <th>本周期 / 秒</th>
                  <th>健康中心 / 秒</th>
                  <th>参考范围 / 秒</th>
                </tr>
              </thead>
              <tbody>
                {wave.mechanism.items
                  .filter((_, i) => [1, 4, 6, 7, 8].includes(i))
                  .map((item) => (
                    <tr key={item.name}>
                      <td>
                        {
                          (
                            {
                              PS1_t50_s: "入口压力下降 50%",
                              PS2_t50_s: "出口压力上升 50%",
                              PS1_transition_s: "入口响应时长",
                              PS2_transition_s: "出口响应时长",
                              handover_gap_s: "进出口交接时间差",
                            } as Record<string, string>
                          )[item.name]
                        }
                      </td>
                      <td>{item.value.toFixed(3)}</td>
                      <td>{item.center.toFixed(3)}</td>
                      <td>
                        {item.lower === null
                          ? "≤ "
                          : item.lower.toFixed(3) + " ～ "}
                        {item.upper.toFixed(3)}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <p>
            规则分数 {wave.mechanism.score.toFixed(2)} / 阈值{" "}
            {wave.mechanism.threshold.toFixed(2)}
          </p>
        </section>
      )}
      <section className="panel">
        <h2>重构残差 · {experiment.channels[channel]}</h2>
        <Chart label="通道绝对重构残差" option={residual} />
      </section>
    </>
  );
}
