// 8766隔离库预置无工作者的排队任务；测试旧ID冲突及真实取消请求。
async function check(page) {
  if (new URL(page.url()).port !== "8766") throw Error("只能运行于隔离QA服务");
  const oldId = "0".repeat(32);
  await page.route("**/api/job", (route) =>
    route.fulfill({
      json: {
        id: oldId,
        status: "running",
        task: "valve",
        log: "QA stale task view",
      },
    }),
  );
  try {
    await page.reload();
    await page.locator(".job summary").click();
    const rejected = page.waitForResponse((response) =>
      response.url().endsWith("/api/job/cancel"),
    );
    await page.getByRole("button", { name: "停止训练", exact: true }).click();
    const response = await rejected;
    if (response.status() !== 409) throw Error("旧任务ID未被拒绝");
    if (response.request().postDataJSON().job_id !== oldId)
      throw Error("前端没有提交显示中的任务ID");
    const current = await (
      await page.request.get("http://127.0.0.1:8766/api/job")
    ).json();
    if (current.cancel_requested) throw Error("旧请求误取消新任务");
  } finally {
    await page.unroute("**/api/job");
  }
  await page.reload();
  await page.locator(".job summary").click();
  const accepted = page.waitForResponse((response) =>
    response.url().endsWith("/api/job/cancel"),
  );
  await page.getByRole("button", { name: "停止训练", exact: true }).click();
  if ((await accepted).status() !== 200) throw Error("当前任务取消失败");
  const pending = page.getByRole("button", { name: "正在停止…", exact: true });
  await pending.waitFor();
  if (await pending.isEnabled()) throw Error("取消请求后仍可重复点击");
  return {
    checks: [
      "stale_id_conflict",
      "current_task_not_cancelled_by_stale_request",
      "explicit_id_cancel",
      "cancel_pending_state",
    ],
  };
}
