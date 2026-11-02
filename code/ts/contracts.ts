/** 平台无关契约；异常分数不默认解释为故障概率。 */
export type ScoreKind = "anomaly-score" | "class-probability";

export interface ModelManifest {
  schemaVersion: 1;
  bundleKind?: "valve-fusion-runtime";
  sync?: { required: boolean; expected_lead_s: number; tolerance_samples: number };
  task: "valve" | "cooler" | "pump" | "accumulator";
  source: string;
  data_sha256: string;
  channels: string[];
  units: string[];
  sampleRateHz: number;
  windowSeconds: number;
  versions: Record<string, string>;
  models: Record<
    "ae" | "iforest" | "rf",
    {
      file: string;
      sha256: string;
      inputName: string;
      shape: (number | null)[];
      normalizationEmbedded: boolean;
      outputIndex: number;
      transform: "identity" | "negate" | "column1";
      threshold: number;
      scoreKind: ScoreKind;
    }
  >;
}

export interface ScoreRecord {
  assetId: string;
  componentId: string;
  modelVersion: string;
  windowEnd: string;
  score: number;
  kind: ScoreKind;
}

/** 实际测得的同步信息；期望偏移和容差由模型包提供，不由请求覆盖。 */
export interface Synchronization {
  command_edge_s?: number | null;
  window_start_s?: number | null;
  channel_skew_s?: number | null;
}

/** 第二版统一原始窗口契约，支持部件异常评分和多类诊断。 */
export interface PortableModelManifest {
  schemaVersion: 2;
  modelId: string;
  version: string;
  origin: Record<string, string>;
  graph: { file: string; sha256: string };
  input: {
    name: string;
    shape: [null, number, number];
    channels: string[];
    units: string[];
    sampleRateHz: number;
    layout: "BTC";
    dtype: "float32";
    preprocessingEmbedded: true;
  };
  outputs: {
    index: number;
    role: "anomaly-score" | "class-probability" | "uncalibrated-class-score" | "prediction-set";
    transform?: "identity" | "negate" | "column1";
    classes?: string[];
    threshold?: number;
  }[];
  execution: { provider: "CPUExecutionProvider"; maxBatch: number };
  scope: string;
  validation: { referenceWindows: number; tolerance: number; maxAbsErrors: Record<string,number> };
}
