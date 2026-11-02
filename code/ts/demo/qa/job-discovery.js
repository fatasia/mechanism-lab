// 注入任务状态验证跨窗口发现与状态文案，不启动训练、不产生模型结果。
async function check(page) {
  let state = { id: null, status: "idle", task: null, log: "" };
  let runReads = 0;
  const observe = (request) => {
    if (request.url().includes("/api/runs/valve")) runReads++;
  };
  page.on("request", observe);
  await page.route("**/api/job", (route) => route.fulfill({ json: state }));
  try {
    await page.reload();
    await page.getByRole("heading", { name: "设备健康总览" }).waitFor();
    const initialReads = runReads;
    const refreshed = page.waitForResponse(
      (response) => response.url().includes("/api/runs/valve") && response.ok(),
    );
    state = {
      id: "qa-completed-job",
      status: "complete",
      task: "valve",
      log: "UI test",
    };
    await page
      .locator(".job summary")
      .filter({ hasText: "完成" })
      .waitFor({ timeout: 16000 });
    await refreshed;
    if (runReads <= initialReads) throw Error("新完成任务未刷新实验列表");
    state = {
      id: "qa-running-job",
      status: "running",
      task: "valve",
      log: "UI test",
      stale: true,
    };
    await page
      .locator(".job summary")
      .filter({ hasText: "心跳中断" })
      .waitFor({ timeout: 16000 });
    state = { ...state, stale: false, worker_status: "queued" };
    await page.locator(".job summary").filter({ hasText: "排队中" }).waitFor();
    return {
      checks: [
        "idle_discovers_completed_job",
        "refreshes_experiment_list",
        "stale_and_queued_states",
      ],
      runReads,
    };
  } finally {
    page.off("request", observe);
    await page.unroute("**/api/job");
    await page.reload();
  }
}
