// 首次状态读取超过10秒超时；后续读取恢复真实隔离QA服务。
async function check(page) {
  await page.reload();
  let calls = 0;
  await page.route("**/api/runtime/status", async (route) => {
    calls++;
    if (calls === 1) {
      await new Promise((resolve) => setTimeout(resolve, 12000));
      await route.abort();
    } else await route.continue();
  });
  try {
    await page
      .getByRole("button", { name: "05运行与告警", exact: true })
      .click();
    await page
      .getByRole("status")
      .filter({ hasText: "连接中断 · 显示最后成功读取的数据" })
      .waitFor({ timeout: 16000 });
    await page
      .getByRole("status")
      .filter({ hasText: "更新于" })
      .waitFor({ timeout: 10000 });
    if (calls < 2) throw Error("读取超时后未重试");
    return { check: "runtime_read_timeout_recovers", calls };
  } finally {
    await page.unroute("**/api/runtime/status");
  }
}
