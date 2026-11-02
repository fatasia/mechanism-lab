import { useState } from "react";

type Part = Record<string, unknown> & {
  part_id: string;
  name: string;
  quantity: number;
  unit: string;
};
type Details = Record<string, unknown> & {
  description?: string;
  bom?: Part[];
  failure_modes?: string[];
};

function structured(value: unknown): value is Details {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const row = value as Details;
  return (
    (row.description === undefined || typeof row.description === "string") &&
    (row.failure_modes === undefined ||
      (Array.isArray(row.failure_modes) &&
        row.failure_modes.every((m) => typeof m === "string"))) &&
    (row.bom === undefined ||
      (Array.isArray(row.bom) &&
        row.bom.every(
          (part) =>
            part &&
            typeof part === "object" &&
            typeof part.part_id === "string" &&
            typeof part.name === "string" &&
            typeof part.quantity === "number" &&
            Number.isFinite(part.quantity) &&
            part.quantity > 0 &&
            typeof part.unit === "string",
        )))
  );
}

/** 只编辑档案属性；备件数量不是仓库库存，不触发领用或库存更新。 */
export function AssetDetails({ initial }: { initial: string }) {
  const [details, setDetails] = useState<Details>(() => JSON.parse(initial));
  const [json, setJson] = useState(initial);
  const [mode, setMode] = useState(() =>
    structured(JSON.parse(initial)) ? "form" : "json",
  );
  const [error, setError] = useState("");
  const parts = details.bom ?? [];
  const modes = details.failure_modes ?? [];

  function switchMode() {
    setError("");
    if (mode === "form") {
      setJson(JSON.stringify(details, null, 2));
      setMode("json");
      return;
    }
    try {
      const next: unknown = JSON.parse(json);
      if (!structured(next))
        throw Error(
          "属性不符合表单格式，请保留JSON编辑。BOM项需包含part_id、name、正数quantity和unit；failure_modes为字符串列表。",
        );
      setDetails(next);
      setMode("form");
    } catch (e) {
      setError(String(e));
    }
  }

  function updatePart(index: number, value: Partial<Part>) {
    setDetails({
      ...details,
      bom: parts.map((part, i) => (i === index ? { ...part, ...value } : part)),
    });
  }

  return (
    <div className="asset-details">
      <div className="section-heading">
        <h3>部件清单与失效模式</h3>
        <button type="button" onClick={switchMode}>
          {mode === "form" ? "编辑属性JSON" : "使用结构化表单"}
        </button>
      </div>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {mode === "json" ? (
        <label>
          BOM、失效模式与档案属性 JSON
          <textarea
            name="details"
            aria-label="BOM、失效模式与档案属性 JSON"
            rows={8}
            value={json}
            onChange={(event) => setJson(event.target.value)}
          />
        </label>
      ) : (
        <>
          <input type="hidden" name="details" value={JSON.stringify(details)} />
          <label>
            档案说明
            <textarea
              rows={2}
              aria-label="档案说明"
              value={details.description ?? ""}
              onChange={(event) =>
                setDetails({ ...details, description: event.target.value })
              }
            />
          </label>
          {parts.map((part, index) => (
            <fieldset key={index} className="asset-part">
              <legend>部件 {index + 1}</legend>
              <div className="editor-grid">
                <label>
                  部件编码
                  <input
                    required
                    value={part.part_id}
                    onChange={(event) =>
                      updatePart(index, { part_id: event.target.value })
                    }
                  />
                </label>
                <label>
                  部件名称
                  <input
                    required
                    value={part.name}
                    onChange={(event) =>
                      updatePart(index, { name: event.target.value })
                    }
                  />
                </label>
                <label>
                  装配数量
                  <input
                    type="number"
                    required
                    min="0.000001"
                    step="any"
                    value={Number.isFinite(part.quantity) ? part.quantity : ""}
                    onChange={(event) =>
                      updatePart(index, {
                        quantity: event.target.valueAsNumber,
                      })
                    }
                  />
                </label>
                <label>
                  计量单位
                  <input
                    required
                    value={part.unit}
                    onChange={(event) =>
                      updatePart(index, { unit: event.target.value })
                    }
                  />
                </label>
              </div>
              <button
                type="button"
                aria-label={`移除部件 ${index + 1}`}
                onClick={() =>
                  setDetails({
                    ...details,
                    bom: parts.filter((_, i) => i !== index),
                  })
                }
              >
                移除部件
              </button>
            </fieldset>
          ))}
          <button
            type="button"
            onClick={() =>
              setDetails({
                ...details,
                bom: [
                  ...parts,
                  { part_id: "", name: "", quantity: 1, unit: "件" },
                ],
              })
            }
          >
            添加部件
          </button>
          <h4>失效模式</h4>
          {modes.map((value, index) => (
            <div className="asset-mode" key={index}>
              <label>
                失效模式 {index + 1}
                <input
                  required
                  value={value}
                  onChange={(event) =>
                    setDetails({
                      ...details,
                      failure_modes: modes.map((m, i) =>
                        i === index ? event.target.value : m,
                      ),
                    })
                  }
                />
              </label>
              <button
                type="button"
                aria-label={`移除失效模式 ${index + 1}`}
                onClick={() =>
                  setDetails({
                    ...details,
                    failure_modes: modes.filter((_, i) => i !== index),
                  })
                }
              >
                移除
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={() =>
              setDetails({ ...details, failure_modes: [...modes, ""] })
            }
          >
            添加失效模式
          </button>
        </>
      )}
    </div>
  );
}
