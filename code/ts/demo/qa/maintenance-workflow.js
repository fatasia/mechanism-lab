// 仅在8766隔离QA数据库创建带QA标识的资产/工单；不代表现场维修。
async function check(page) {
  if (new URL(page.url()).port !== "8766") throw Error("只能运行于隔离QA服务");
  await page.reload();
  let calls = 0;
  await page.route("**/api/maintenance", async (route) => {
    if (++calls === 1) await route.abort("internetdisconnected");
    else await route.continue();
  });
  await page.getByRole("button", { name: "06运维工作台", exact: true }).click();
  await page.getByRole("alert").waitFor();
  await page.getByRole("button", { name: "更新运维记录", exact: true }).click();
  await page.getByRole("heading", { name: "设备档案与可维护单元" }).waitFor();
  await page.unroute("**/api/maintenance");
  const id = `qa-ui-${Date.now()}`;
  await page.getByLabel("设备编号", { exact: true }).fill(id);
  await page.getByLabel("名称", { exact: true }).fill("QA流程测试阀门");
  await page.getByRole("button", { name: "保存档案", exact: true }).click();
  await page.getByRole("cell", { name: id, exact: true }).waitFor();
  await page.getByRole("button", { name: "维修工单", exact: true }).click();
  await page
    .getByRole("combobox", { name: "设备 / 部件", exact: true })
    .selectOption(id);
  await page.getByLabel("事项", { exact: true }).fill(id);
  await page.getByLabel("处理人", { exact: true }).fill("QA测试员");
  await page.getByRole("button", { name: "创建工单", exact: true }).click();
  const row = page
    .locator("tbody tr")
    .filter({ has: page.getByRole("cell", { name: id, exact: true }) });
  await row.getByRole("button", { name: "处理工单" }).click();
  await page.getByRole("button", { name: "开始处理", exact: true }).click();
  await page.getByRole("button", { name: "提交完成", exact: true }).click();
  await page
    .getByRole("alert")
    .filter({ hasText: "必须填写实际处理内容" })
    .waitFor();
  await page
    .getByLabel("实际处理内容", { exact: true })
    .fill("QA界面流程测试，不是实际设备维修");
  await page.getByRole("button", { name: "提交完成", exact: true }).click();
  await page.getByRole("button", { name: "复核关闭", exact: true }).click();
  await page
    .getByRole("alert")
    .filter({ hasText: "必须填写复核结果" })
    .waitFor();
  await page.getByLabel("本次操作人", { exact: true }).fill("QA复核员");
  await page
    .getByLabel("维修后复核结果", { exact: true })
    .fill(`QA状态转换核验通过 ${id}`);
  await page.getByRole("button", { name: "复核关闭", exact: true }).click();
  await row.getByRole("cell", { name: "已关闭", exact: true }).waitFor();
  await page.getByRole("button", { name: "案例与审计", exact: true }).click();
  await page
    .getByRole("cell", { name: `QA状态转换核验通过 ${id}`, exact: true })
    .waitFor();
  await page.screenshot({ path: "maintenance-workflow.png", fullPage: true });
  return {
    asset: id,
    checks: [
      "initial_read_retry",
      "create_asset",
      "create_order",
      "require_processing_notes",
      "require_verification",
      "close_order_creates_case",
    ],
  };
}
