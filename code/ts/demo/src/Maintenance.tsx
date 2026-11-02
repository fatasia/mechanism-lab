import { FormEvent, ReactNode, useEffect, useState } from "react";
import { request } from "./domain";
import { AssetDetails } from "./AssetDetails";

type Asset = {
  id: string;
  name: string;
  kind: string;
  parent_id: string | null;
  details: string;
};
type Order = {
  id: number;
  asset_id: string;
  title: string;
  kind: string;
  operator: string;
  status: string;
  revision: number;
  notes: string;
  cause: string;
  verification: string;
};
type Plan = {
  id: number;
  asset_id: string;
  name: string;
  interval_days: number;
  steps: string;
  next_due: string;
};
type Row = Record<string, string | number | null>;
type Snapshot = {
  assets: Asset[];
  work_orders: Order[];
  cases: Row[];
  parts: { id: string; name: string; quantity: number }[];
  stock_moves: Row[];
  people: { id: string; name: string; skills: string; shift: string }[];
  plans: Plan[];
  checks: Row[];
  audit: Row[];
  performance: Row[];
};
const states: Record<string, string> = {
  open: "待处理",
  working: "处理中",
  completed: "待复核",
  closed: "已关闭",
  cancelled: "已取消",
};
const assetKinds: Record<string, string> = {
  line: "产线",
  equipment: "设备",
  station: "工位",
  component: "部件",
};

function Field({
  label,
  name,
  type = "text",
  value,
  required = true,
  children,
}: {
  label: string;
  name: string;
  type?: string;
  value?: string | number;
  required?: boolean;
  children?: ReactNode;
}) {
  return (
    <label>
      {label}
      {children ? (
        <select name={name} required={required} defaultValue={value}>
          {children}
        </select>
      ) : (
        <input
          name={name}
          type={type}
          required={required}
          defaultValue={value}
        />
      )}
    </label>
  );
}
function DataTable({
  rows,
  columns,
}: {
  rows: Row[];
  columns: [string, string][];
}) {
  return !rows.length ? (
    <p>暂无记录。</p>
  ) : (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map(([id, name]) => (
              <th key={id}>{name}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={String(row.id ?? i)}>
              {columns.map(([id]) => (
                <td key={id}>
                  {typeof row[id] === "number" && !Number.isInteger(row[id])
                    ? Number(row[id]).toFixed(3)
                    : String(row[id] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Maintenance() {
  const [tab, setTab] = useState("设备"),
    [data, setData] = useState<Snapshot | null>(null),
    [selected, setSelected] = useState<Asset | null>(null),
    [order, setOrder] = useState<Order | null>(null),
    [planId, setPlanId] = useState(""),
    [error, setError] = useState(""),
    [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false);
  const [stockRequestId, setStockRequestId] = useState(() =>
    crypto.randomUUID(),
  );
  const [assetEdit, setAssetEdit] = useState(0);
  const [assetQuery, setAssetQuery] = useState("");
  async function refresh(signal?: AbortSignal) {
    const next = await request<Snapshot>("/api/maintenance", {
      signal: AbortSignal.any([
        ...(signal ? [signal] : []),
        AbortSignal.timeout(10_000),
      ]),
    });
    setData(next);
    setOrder((current) =>
      current
        ? (next.work_orders.find((o) => o.id === current.id) ?? null)
        : null,
    );
    setError("");
  }
  useEffect(() => {
    const controller = new AbortController();
    void refresh(controller.signal).catch((e) => {
      if (!controller.signal.aborted) setError(String(e));
    });
    return () => controller.abort();
  }, []);
  async function send(path: string, body: unknown, method = "POST") {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const result = await request<{ order_id?: number }>(
        `/api/maintenance${path}`,
        {
          method,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );
      await refresh();
      if (path.endsWith("/stock")) setStockRequestId(crypto.randomUUID());
      setMessage(
        result.order_id
          ? `已保存，点检未通过项已创建工单 ${result.order_id}`
          : "已保存",
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  function submit(
    event: FormEvent<HTMLFormElement>,
    fn: (form: FormData) => [string, unknown, string?],
  ) {
    event.preventDefault();
    try {
      const [path, body, method] = fn(new FormData(event.currentTarget));
      void send(path, body, method);
    } catch (e) {
      setError(String(e));
    }
  }
  const assetOptions = (
    <>
      <option value="">选择设备或部件</option>
      {data?.assets.map((a) => (
        <option key={a.id} value={a.id}>
          {a.name} · {a.id}
        </option>
      ))}
    </>
  );
  const plan = data?.plans.find((p) => p.id === Number(planId));
  const visibleAssets =
    data?.assets.filter((asset) =>
      `${asset.id} ${asset.name}`
        .toLowerCase()
        .includes(assetQuery.trim().toLowerCase()),
    ) ?? [];
  return (
    <>
      <div className="lab-tabs">
        {["设备", "维修工单", "备件", "点检保养", "人员", "案例与审计"].map(
          (t) => (
            <button
              key={t}
              className={tab === t ? "selected" : ""}
              aria-pressed={tab === t}
              onClick={() => {
                setTab(t);
                setError("");
                setMessage("");
              }}
            >
              {t}
            </button>
          ),
        )}
        <button
          disabled={busy}
          onClick={() => void refresh().catch((e) => setError(String(e)))}
        >
          更新运维记录
        </button>
      </div>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {message && <p role="status">{message}</p>}
      {!data ? (
        <section className="panel">读取本地运维记录…</section>
      ) : (
        <>
          {tab === "设备" && (
            <section className="panel">
              <div className="section-heading">
                <div>
                  <h2>设备档案与可维护单元</h2>
                </div>
                <button
                  onClick={() => {
                    setSelected(null);
                    setAssetEdit((v) => v + 1);
                  }}
                >
                  新建档案
                </button>
              </div>
              <div className="lab-controls">
                <label>
                  查找档案
                  <input
                    type="search"
                    placeholder="名称或编号"
                    value={assetQuery}
                    onChange={(event) => setAssetQuery(event.target.value)}
                  />
                </label>
              </div>
              <div className="table-wrap">
                <table className="asset-table">
                  <thead>
                    <tr>
                      <th>编号</th>
                      <th>名称</th>
                      <th>层级</th>
                      <th>父级</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleAssets.map((a) => (
                      <tr key={a.id}>
                        <td data-label="编号">{a.id}</td>
                        <td data-label="名称">{a.name}</td>
                        <td data-label="层级">
                          {assetKinds[a.kind] ?? a.kind}
                        </td>
                        <td data-label="父级">{a.parent_id ?? "—"}</td>
                        <td data-label="操作">
                          <button
                            onClick={() => {
                              setSelected(a);
                              setAssetEdit((v) => v + 1);
                            }}
                          >
                            编辑档案
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {!visibleAssets.length && (
                  <p role="status">
                    {assetQuery.trim() ? "没有匹配的档案。" : "尚无设备档案。"}
                  </p>
                )}
              </div>
              <form
                className="editor"
                key={`${selected?.id ?? "new"}/${assetEdit}`}
                onSubmit={(e) =>
                  submit(e, (f) => [
                    "/assets",
                    {
                      id: f.get("id"),
                      name: f.get("name"),
                      kind: f.get("kind"),
                      parent_id: f.get("parent") || null,
                      details: JSON.parse(String(f.get("details") || "{}")),
                    },
                    "PUT",
                  ])
                }
              >
                <div className="editor-grid">
                  <Field label="设备编号" name="id" value={selected?.id} />
                  <Field label="名称" name="name" value={selected?.name} />
                  <Field
                    label="层级"
                    name="kind"
                    value={selected?.kind ?? "equipment"}
                  >
                    <option value="line">产线</option>
                    <option value="equipment">设备</option>
                    <option value="station">工位</option>
                    <option value="component">部件</option>
                  </Field>
                  <Field
                    label="父级"
                    name="parent"
                    required={false}
                    value={selected?.parent_id ?? ""}
                  >
                    <option value="">无父级</option>
                    {data.assets
                      .filter((a) => a.id !== selected?.id)
                      .map((a) => (
                        <option key={a.id} value={a.id}>
                          {a.name}
                        </option>
                      ))}
                  </Field>
                </div>
                <AssetDetails
                  initial={
                    selected?.details ??
                    '{"bom":[],"failure_modes":[],"description":""}'
                  }
                />
                <button className="primary" disabled={busy}>
                  保存档案
                </button>
              </form>
            </section>
          )}
          {tab === "维修工单" && (
            <>
              <section className="panel">
                <h2>维修与维护工单</h2>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>编号</th>
                        <th>设备</th>
                        <th>事项</th>
                        <th>状态</th>
                        <th>处理人</th>
                        <th></th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.work_orders.map((o) => (
                        <tr key={o.id}>
                          <td>{o.id}</td>
                          <td>{o.asset_id}</td>
                          <td>{o.title}</td>
                          <td>{states[o.status]}</td>
                          <td>{o.operator}</td>
                          <td>
                            <button onClick={() => setOrder(o)}>
                              处理工单
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <form
                  className="editor"
                  onSubmit={(e) =>
                    submit(e, (f) => [
                      "/orders",
                      {
                        asset_id: f.get("asset"),
                        title: f.get("title"),
                        operator: f.get("operator"),
                        kind: f.get("kind"),
                        event_id: f.get("event")
                          ? Number(f.get("event"))
                          : null,
                      },
                    ])
                  }
                >
                  <div className="editor-grid">
                    <Field label="设备 / 部件" name="asset">
                      {assetOptions}
                    </Field>
                    <Field label="事项" name="title" />
                    <Field label="处理人" name="operator" />
                    <Field label="类别" name="kind">
                      <option value="repair">维修</option>
                      <option value="inspection">点检</option>
                      <option value="service">保养</option>
                    </Field>
                    <Field
                      label="关联告警编号（可选）"
                      name="event"
                      type="number"
                      required={false}
                    />
                  </div>
                  <button className="primary" disabled={busy}>
                    创建工单
                  </button>
                </form>
              </section>
              {order && (
                <section className="panel">
                  <h2>
                    工单 {order.id} · {states[order.status]}
                  </h2>
                  <form
                    key={`${order.id}/${order.revision}`}
                    className="editor"
                    onSubmit={(e) => {
                      const target = e.nativeEvent as SubmitEvent;
                      const next = (target.submitter as HTMLButtonElement)
                        .value;
                      submit(e, (f) => [
                        `/orders/${order.id}/transition`,
                        {
                          status: next,
                          revision: order.revision,
                          operator: f.get("operator"),
                          notes: f.get("notes"),
                          cause: f.get("cause"),
                          verification: f.get("verification"),
                        },
                      ]);
                    }}
                  >
                    <Field
                      label="本次操作人"
                      name="operator"
                      value={order.operator}
                    />
                    <label>
                      实际处理内容
                      <textarea name="notes" defaultValue={order.notes} />
                    </label>
                    <label>
                      原因与失效模式（未确认可留空）
                      <input name="cause" defaultValue={order.cause} />
                    </label>
                    <label>
                      维修后复核结果
                      <textarea
                        name="verification"
                        defaultValue={order.verification}
                      />
                    </label>
                    <div className="inline-actions">
                      {order.status === "open" && (
                        <button
                          className="primary"
                          value="working"
                          disabled={busy}
                        >
                          开始处理
                        </button>
                      )}
                      {order.status === "working" && (
                        <button
                          className="primary"
                          value="completed"
                          disabled={busy}
                        >
                          提交完成
                        </button>
                      )}
                      {order.status === "completed" && (
                        <>
                          <button
                            className="primary"
                            value="closed"
                            disabled={busy}
                          >
                            复核关闭
                          </button>
                          <button value="working" disabled={busy}>
                            退回处理
                          </button>
                        </>
                      )}
                      {["open", "working"].includes(order.status) && (
                        <button value="cancelled" disabled={busy}>
                          取消工单
                        </button>
                      )}
                    </div>
                  </form>
                </section>
              )}
            </>
          )}
          {tab === "备件" && (
            <section className="panel">
              <h2>备件库存与领退流水</h2>
              <DataTable
                rows={data.parts}
                columns={[
                  ["id", "编号"],
                  ["name", "备件"],
                  ["quantity", "当前库存"],
                ]}
              />
              <div className="editor-grid">
                <form
                  className="editor"
                  onSubmit={(e) =>
                    submit(e, (f) => [
                      "/parts",
                      { id: f.get("id"), name: f.get("name") },
                      "PUT",
                    ])
                  }
                >
                  <h3>备件档案</h3>
                  <Field label="备件编号" name="id" />
                  <Field label="名称" name="name" />
                  <button disabled={busy}>保存备件</button>
                </form>
                <form
                  className="editor"
                  onSubmit={(e) =>
                    submit(e, (f) => {
                      const reason = String(f.get("reason"));
                      return [
                        `/parts/${f.get("part")}/stock`,
                        {
                          delta:
                            Number(f.get("quantity")) *
                            (reason === "issue" ? -1 : 1),
                          reason,
                          request_id: stockRequestId,
                          operator: f.get("operator"),
                          order_id: f.get("order")
                            ? Number(f.get("order"))
                            : null,
                        },
                      ];
                    })
                  }
                >
                  <h3>出入库</h3>
                  <Field label="备件" name="part">
                    <option value="">选择备件</option>
                    {data.parts.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.name}
                      </option>
                    ))}
                  </Field>
                  <Field label="动作" name="reason">
                    <option value="receive">入库</option>
                    <option value="issue">领用</option>
                    <option value="return">退库</option>
                  </Field>
                  <Field label="数量" name="quantity" type="number" />
                  <Field label="操作人" name="operator" />
                  <Field
                    label="工单（领用 / 退库必选）"
                    name="order"
                    required={false}
                  >
                    <option value="">未关联</option>
                    {data.work_orders.map((o) => (
                      <option key={o.id} value={o.id}>
                        {o.id} · {o.title}
                      </option>
                    ))}
                  </Field>
                  <button className="primary" disabled={busy}>
                    记录库存变动
                  </button>
                </form>
              </div>
              <details>
                <summary>库存流水</summary>
                <DataTable
                  rows={data.stock_moves}
                  columns={[
                    ["part_id", "备件"],
                    ["order_id", "工单"],
                    ["delta", "数量变动"],
                    ["reason", "动作"],
                    ["operator", "操作人"],
                    ["at", "时间"],
                  ]}
                />
              </details>
            </section>
          )}
          {tab === "点检保养" && (
            <section className="panel">
              <h2>点检与周期保养</h2>
              <DataTable
                rows={data.plans as unknown as Row[]}
                columns={[
                  ["id", "计划"],
                  ["asset_id", "设备"],
                  ["name", "名称"],
                  ["interval_days", "周期 / 天"],
                  ["next_due", "下次计划"],
                ]}
              />
              <div className="editor-grid">
                <form
                  className="editor"
                  onSubmit={(e) =>
                    submit(e, (f) => [
                      "/plans",
                      {
                        asset_id: f.get("asset"),
                        name: f.get("name"),
                        interval_days: Number(f.get("days")),
                        steps: String(f.get("steps"))
                          .split("\n")
                          .map((s) => s.trim())
                          .filter(Boolean),
                        next_due: f.get("due"),
                      },
                    ])
                  }
                >
                  <h3>创建计划</h3>
                  <Field label="设备" name="asset">
                    {assetOptions}
                  </Field>
                  <Field label="计划名称" name="name" />
                  <Field
                    label="周期 / 天"
                    name="days"
                    type="number"
                    value={7}
                  />
                  <Field label="下次计划日期" name="due" type="date" />
                  <label>
                    检查项（每行一项）
                    <textarea name="steps" required rows={4} />
                  </label>
                  <button disabled={busy}>保存计划</button>
                </form>
                <form
                  className="editor"
                  onSubmit={(e) =>
                    submit(e, (f) => [
                      `/plans/${planId}/check`,
                      {
                        operator: f.get("operator"),
                        notes: f.get("notes"),
                        results: Object.fromEntries(
                          (JSON.parse(plan!.steps) as string[]).map((step) => [
                            step,
                            f.get(`step:${step}`) === "on",
                          ]),
                        ),
                      },
                    ])
                  }
                >
                  <h3>记录执行结果</h3>
                  <label>
                    检查计划
                    <select
                      required
                      value={planId}
                      onChange={(e) => setPlanId(e.target.value)}
                    >
                      <option value="">选择计划</option>
                      {data.plans.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.name} · {p.asset_id}
                        </option>
                      ))}
                    </select>
                  </label>
                  <Field label="执行人" name="operator" />
                  {plan &&
                    (JSON.parse(plan.steps) as string[]).map((step) => (
                      <label className="check-label" key={`${plan.id}/${step}`}>
                        <input type="checkbox" name={`step:${step}`} />
                        {step}：通过
                      </label>
                    ))}
                  <label>
                    执行记录 / 未通过项
                    <textarea name="notes" />
                  </label>
                  <button className="primary" disabled={busy || !plan}>
                    保存点检结果
                  </button>
                </form>
              </div>
              <details>
                <summary>点检履历</summary>
                <DataTable
                  rows={data.checks}
                  columns={[
                    ["plan_id", "计划"],
                    ["operator", "执行人"],
                    ["result", "结果"],
                    ["notes", "记录"],
                    ["at", "时间"],
                  ]}
                />
              </details>
            </section>
          )}
          {tab === "人员" && (
            <section className="panel">
              <h2>人员、技能与作业统计</h2>
              <DataTable
                rows={data.people}
                columns={[
                  ["id", "人员编号"],
                  ["name", "姓名"],
                  ["skills", "技能"],
                  ["shift", "班次"],
                ]}
              />
              <form
                className="editor"
                onSubmit={(e) =>
                  submit(e, (f) => [
                    "/people",
                    {
                      id: f.get("id"),
                      name: f.get("name"),
                      skills: String(f.get("skills"))
                        .split(",")
                        .map((s) => s.trim())
                        .filter(Boolean),
                      shift: f.get("shift"),
                    },
                    "PUT",
                  ])
                }
              >
                <div className="editor-grid">
                  <Field label="人员编号（相同编号更新）" name="id" />
                  <Field label="姓名" name="name" />
                  <Field label="技能（逗号分隔）" name="skills" />
                  <Field label="班次" name="shift" />
                </div>
                <button disabled={busy}>保存人员档案</button>
              </form>
              <h3>已关闭工单统计</h3>
              <DataTable
                rows={data.performance}
                columns={[
                  ["operator", "关闭操作人"],
                  ["closed_orders", "关闭工单数"],
                  ["order_cycle_hours", "平均工单周期 / 小时"],
                ]}
              />
            </section>
          )}
          {tab === "案例与审计" && (
            <section className="panel">
              <h2>维修复核案例</h2>
              <DataTable
                rows={data.cases}
                columns={[
                  ["order_id", "工单"],
                  ["asset_id", "设备"],
                  ["cause", "报告原因"],
                  ["action", "处理内容"],
                  ["verification", "复核结果"],
                  ["confirmed_by", "复核人"],
                ]}
              />
              <details>
                <summary>本地操作审计</summary>
                <DataTable
                  rows={data.audit}
                  columns={[
                    ["at", "时间"],
                    ["action", "动作"],
                    ["entity", "对象"],
                    ["detail", "内容"],
                  ]}
                />
              </details>
            </section>
          )}
        </>
      )}
    </>
  );
}
