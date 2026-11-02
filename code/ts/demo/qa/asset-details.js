// 隔离库验证结构化属性、未知字段保留和库存隔离；不写生产资产。
async function check(page) {
  if (new URL(page.url()).port !== "8766") throw Error("只能运行于隔离QA服务");
  await page.reload();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.getByRole("button", { name: "06运维工作台", exact: true }).click();
  await page.getByRole("heading", { name: "设备档案与可维护单元" }).waitFor();
  const before = await (
    await page.request.get(new URL("/api/maintenance", page.url()).href)
  ).json();
  const id = `qa-bom-${Date.now()}`;
  await page.getByLabel("设备编号", { exact: true }).fill(id);
  await page.getByLabel("名称", { exact: true }).fill("QA结构化档案");
  await page.getByRole("button", { name: "编辑属性JSON", exact: true }).click();
  await page.getByLabel("BOM、失效模式与档案属性 JSON", { exact: true }).fill(
    JSON.stringify({
      bom: [],
      failure_modes: [],
      custom: { preserve: true },
    }),
  );
  await page
    .getByRole("button", { name: "使用结构化表单", exact: true })
    .click();
  await page
    .getByLabel("档案说明", { exact: true })
    .fill("QA装配清单，不是库存变更");
  await page.getByRole("button", { name: "添加部件", exact: true }).click();
  await page.getByLabel("部件编码", { exact: true }).fill("QA-SEAL");
  await page.getByLabel("部件名称", { exact: true }).fill("测试密封件");
  await page.getByLabel("装配数量", { exact: true }).fill("0");
  await page.getByRole("button", { name: "保存档案", exact: true }).click();
  const invalid = await (
    await page.request.get(new URL("/api/maintenance", page.url()).href)
  ).json();
  if (invalid.assets.some((a) => a.id === id))
    throw Error("无效装配数量仍然保存");
  await page.getByLabel("装配数量", { exact: true }).fill("2.5");
  await page.getByLabel("计量单位", { exact: true }).fill("米");
  await page.getByRole("button", { name: "添加失效模式", exact: true }).click();
  await page.getByLabel("失效模式 1", { exact: true }).fill("密封磨损");
  await page.getByRole("button", { name: "保存档案", exact: true }).click();
  const row = page
    .locator("tbody tr")
    .filter({ has: page.getByRole("cell", { name: id, exact: true }) });
  await row.waitFor();
  await page.getByLabel("查找档案", { exact: true }).fill(id);
  if ((await page.locator("tbody tr").count()) !== 1) throw Error("档案检索未筛选");
  await page.getByLabel("查找档案", { exact: true }).fill("QA-NO-MATCH-FIXTURE");
  await page.getByRole("status").filter({ hasText: "没有匹配的档案" }).waitFor();
  await page.getByLabel("查找档案", { exact: true }).fill(id);
  await row.getByRole("button", { name: "编辑档案", exact: true }).click();
  if (
    (await page.getByLabel("装配数量", { exact: true }).inputValue()) !== "2.5"
  )
    throw Error("重读装配数量不一致");
  await page.setViewportSize({ width: 390, height: 900 });
  if (
    await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
  )
    throw Error("BOM窄屏溢出");
  await page.screenshot({ path: "asset-details-mobile.png", fullPage: true });
  await page.getByRole("button", { name: "编辑属性JSON", exact: true }).click();
  const detailInput = page.getByLabel("BOM、失效模式与档案属性 JSON", {
    exact: true,
  });
  const details = JSON.parse(await detailInput.inputValue());
  if (
    !details.custom.preserve ||
    details.bom[0].quantity !== 2.5 ||
    details.failure_modes[0] !== "密封磨损"
  )
    throw Error("属性切换丢失内容");
  // 旧版自定义BOM结构不强制转换，不丢弃未知字段。
  const legacy = JSON.stringify({
    bom: [{ legacy_code: "A" }],
    custom: { preserve: true },
  });
  await detailInput.fill(legacy);
  await page
    .getByRole("button", { name: "使用结构化表单", exact: true })
    .click();
  await page
    .getByRole("alert")
    .filter({ hasText: "属性不符合表单格式" })
    .waitFor();
  if ((await detailInput.inputValue()) !== legacy) throw Error("未知BOM被覆盖");
  await detailInput.fill(JSON.stringify(details));
  await page
    .getByRole("button", { name: "使用结构化表单", exact: true })
    .click();
  await page.getByRole("button", { name: "移除部件 1", exact: true }).click();
  await page
    .getByRole("button", { name: "移除失效模式 1", exact: true })
    .click();
  await page.getByRole("button", { name: "保存档案", exact: true }).click();
  await page.getByRole("status").filter({ hasText: "已保存" }).waitFor();
  const after = await (
    await page.request.get(new URL("/api/maintenance", page.url()).href)
  ).json();
  const saved = JSON.parse(after.assets.find((a) => a.id === id).details);
  if (saved.bom.length || saved.failure_modes.length || !saved.custom.preserve)
    throw Error("移除项目或未知字段保存失败");
  if (
    JSON.stringify(before.parts) !== JSON.stringify(after.parts) ||
    JSON.stringify(before.stock_moves) !== JSON.stringify(after.stock_moves)
  )
    throw Error("档案编辑意外改变库存");
  await page.setViewportSize({ width: 1440, height: 1000 });
  return {
    asset: id,
    checks: [
      "structured_bom",
      "invalid_quantity",
      "reload",
      "json_roundtrip",
      "unknown_fields_preserved",
      "legacy_bom_retained",
      "remove_entries",
      "stock_unchanged",
      "mobile_layout",
      "asset_search_and_empty_state",
    ],
  };
}
