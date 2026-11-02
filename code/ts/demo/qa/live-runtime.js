// 使用隔离QA服务；预置lab/slow序号10、lab/fast序号20的有效历史。
// playwright-cli -s=live-ui run-code --filename ../code/ts/demo/qa/live-runtime.js
async function check(page) {
  const checks = [];
  await page.getByRole("button", { name: "05运行与告警", exact: true }).click();
  await page
    .getByRole("heading", { name: "告警投递队列", exact: true })
    .waitFor();
  const entity = page.getByLabel("回放实体", { exact: true });
  const sequence = page.getByLabel("提交序号", { exact: true });
  const live = page.getByRole("checkbox", {
    name: "自动刷新（3秒）",
    exact: true,
  });
  await entity.fill("lab/fast");
  await page.waitForFunction(
    () => document.querySelector('[aria-label="提交序号"]').value === "21",
  );
  let reads = 0;
  const observe = (req) => {
    if (req.url().includes("/api/runtime/status")) reads++;
  };
  page.on("request", observe);
  await sequence.fill("999");
  await page.waitForResponse(
    (r) => r.url().includes("/api/runtime/status") && r.ok(),
  );
  await page.waitForResponse(
    (r) => r.url().includes("/api/runtime/status") && r.ok(),
  );
  if ((await sequence.inputValue()) !== "999") throw Error("轮询覆盖手动序号");
  checks.push("auto_refresh_preserves_manual_sequence");
  await live.uncheck();
  const pausedReads = reads;
  await new Promise((resolve) => setTimeout(resolve, 3600));
  if (reads !== pausedReads) throw Error("暂停后仍有新轮询");
  checks.push("pause_stops_polling");
  await page.route(
    "**/api/runtime/history?entity=lab%2Fslow",
    async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 700));
      await route.continue();
    },
  );
  await entity.fill("lab/slow");
  await entity.fill("lab/fast");
  await page.waitForFunction(
    () => document.querySelector('[aria-label="提交序号"]').value === "21",
  );
  await new Promise((resolve) => setTimeout(resolve, 1000));
  if ((await sequence.inputValue()) !== "21")
    throw Error("旧实体响应污染新实体");
  await page.unroute("**/api/runtime/history?entity=lab%2Fslow");
  checks.push("entity_switch_rejects_stale_response");
  await live.check();
  try {
    await page.context().setOffline(true);
    await page.getByRole("status").filter({ hasText: "连接中断" }).waitFor();
    await page.screenshot({ path: "live-ui-offline.png", fullPage: true });
  } finally {
    await page.context().setOffline(false);
  }
  await page.getByRole("status").filter({ hasText: "更新于" }).waitFor();
  checks.push("offline_state_and_auto_recovery");
  page.off("request", observe);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: "live-ui-mobile.png", fullPage: true });
  if (
    await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
  )
    throw Error("窄屏整页溢出");
  checks.push("mobile_no_page_overflow");
  return { checks };
}
