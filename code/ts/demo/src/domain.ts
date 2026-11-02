export type Task = "valve" | "cooler" | "pump" | "accumulator";
export type Model = "ae" | "iforest" | "mad" | "rf";
export interface Metric {
  threshold: number;
  test_pr_auc: number | null;
  precision: number | null;
  recall: number | null;
  test_positive_cycles: number;
  test_alert_cycles: number;
  onnx_max_abs_error?: number;
  false_positive_per_1000?: number | null;
  false_positive_ids?: number[];
  false_negative_ids?: number[];
  confusion: number[][];
}
export interface Experiment {
  task: Task;
  run_id: string;
  profile: string;
  units: string[];
  sample_rate_hz: number;
  window_start_seconds?: number;
  window_seconds?: number;
  architecture?: string;
  normal_policy?: string;
  cycle_count: number;
  source: string;
  protocol: string;
  scope: string;
  channels: string[];
  train_healthy_cycles: number;
  calibration_healthy_cycles: number;
  models: Record<Model, Metric>;
  history: { epoch: number; train: number; validation: number }[];
  source_hash?: string;
  split_hash?: string;
  data_sha256: string;
}
export interface Replay {
  run_id: string;
  scores: number[];
  threshold: number;
  healthy: boolean[];
  partitions: string[];
  events: { start: number; end: number; score: number; reason: string }[];
  splits: Record<string, number[]>;
}
export interface Cycle {
  actual: number[];
  reconstruction: number[];
  healthy: boolean;
  mechanism?: {
    score: number;
    threshold: number;
    flagged: boolean;
    items: {
      name: string;
      value: number;
      center: number;
      upper: number;
      lower: number | null;
      unit: string;
    }[];
  };
}
export interface Job {
  id?: string;
  stale?: boolean;
  worker_status?: string;
  cancel_requested?: boolean;
  status: string;
  task: Task | null;
  log: string;
}
export const names: Record<Task, string> = {
  valve: "换向阀",
  cooler: "冷却器",
  pump: "液压泵",
  accumulator: "蓄能器",
};
export const algorithms: Record<Model, string> = {
  ae: "自编码器",
  iforest: "孤立森林",
  mad: "稳健统计",
  rf: "随机森林",
};
export const partitions: Record<string, string> = {
  train: "训练集",
  validation: "验证集",
  calibration: "校准集",
  test: "测试集",
  excluded: "隔离周期",
};
export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    signal: init?.signal
      ? AbortSignal.any([init.signal, AbortSignal.timeout(15000)])
      : AbortSignal.timeout(15000),
  });
  if (!response.ok) {
    const content = response.headers.get("content-type")?.includes("json")
      ? await response.json()
      : null;
    throw new Error(
      typeof content?.detail === "string"
        ? content.detail
        : `请求失败 (${response.status})，请检查本地服务和实验版本`,
    );
  }
  return response.json();
}
