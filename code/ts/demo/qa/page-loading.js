// 六页实际加载、窄屏溢出和未捕获异常检查，不提交业务写入。
async function check(page) {
  const errors = [];
  const observe = (error) => errors.push(error.message);
  page.on("pageerror", observe);
  const pages = [
    ["01总览", "周期异常分数"],
    ["02部件分析", "换向机理证据 · 独立于 AE 告警"],
    ["03实验与模型", "固定协议 · 四模型对照"],
    ["04算法基准", "执行器 · 诊断与适用域"],
    ["05运行与告警", "告警投递队列"],
    ["06运维工作台", "设备档案与可维护单元"],
  ];
  const checks = [];
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 900 });
    for (const [navigation, heading] of pages) {
      await page.getByRole("button", { name: navigation, exact: true }).click();
      await page.getByRole("heading", { name: heading, exact: true }).waitFor();
      if (
        await page.evaluate(
          () => document.documentElement.scrollWidth > innerWidth,
        )
      ) {
        throw Error(`${navigation}在${width}px整页溢出`);
      }
      checks.push({ navigation, width });
    }
  }
  page.off("pageerror", observe);
  if (errors.length) throw Error(errors.join("\n"));
  return { checks, uncaught_errors: errors };
}
