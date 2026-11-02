// 加载已完成真实实验报告；坏格式仅作为界面错误处理用例。
async function check(page) {
  await page.reload();
  await page.getByRole("button", { name: "04算法基准", exact: true }).click();
  await page.getByRole("button", { name: "独立报告", exact: true }).click();
  const input = page.getByLabel("选择实验报告", { exact: true });
  await input.setInputFiles(
    "../outputs/phase-alignment-study/runs/20261031T190809Z-8676888a/metrics.json",
  );
  await page.getByText("20261031T190809Z-8676888a", { exact: true }).waitFor();
  const rows = page.locator("tbody tr");
  if ((await rows.count()) !== 3) throw Error("分相位报告行数错误");
  if ((await rows.nth(2).getByRole("cell").nth(3).innerText()) !== "6")
    throw Error("漏报指标不匹配");
  await page.getByLabel("报告种子", { exact: true }).selectOption("44");
  if ((await rows.nth(2).getByRole("cell").nth(3).innerText()) !== "5")
    throw Error("种子切换未生效");
  await input.setInputFiles(
    "../outputs/context-gate-study/runs/20261031T190412Z-5e76302c/metrics.json",
  );
  await page.getByText("20261031T190412Z-5e76302c", { exact: true }).waitFor();
  if ((await rows.count()) !== 6) throw Error("门槛对照报告行数错误");
  await page
    .getByLabel("报告留出协议", { exact: true })
    .selectOption("held-out-profile");
  if ((await rows.nth(2).getByRole("cell").nth(3).innerText()) !== "133")
    throw Error("协议切换结果不匹配");
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.screenshot({ path: "report-import-desktop.png", fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  if (
    await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
  )
    throw Error("报告窄屏整页溢出");
  await page.screenshot({ path: "report-import-mobile.png", fullPage: true });
  await input.setInputFiles(
    "../outputs/valve-fusion-study/runs/20261031T191451Z-1892de96/metrics.json",
  );
  await page.getByText("20261031T191451Z-1892de96", { exact: true }).waitFor();
  if ((await rows.count()) !== 4) throw Error("阀门报告行数错误");
  if ((await rows.nth(1).getByRole("cell").nth(3).innerText()) !== "0")
    throw Error("阀门漏报数不匹配");
  if ((await rows.nth(1).getByRole("cell").nth(6).innerText()) !== "1.0000")
    throw Error("阀门二分类宏F1不匹配");
  await page.getByLabel("报告留出协议", { exact: true }).selectOption("验证集");
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.screenshot({ path: "valve-fusion-report.png", fullPage: true });
  await input.setInputFiles(
    "../outputs/phase-coral-study/runs/20261031T192736Z-9afdb0b5/metrics.json",
  );
  await page.getByText("20261031T192736Z-9afdb0b5", { exact: true }).waitFor();
  if ((await rows.count()) !== 2) throw Error("深度工况对照行数错误");
  if ((await rows.nth(0).getByRole("cell").nth(3).innerText()) !== "93")
    throw Error("ERM漏报数不匹配");
  if ((await rows.nth(1).getByRole("cell").nth(3).innerText()) !== "74")
    throw Error("工况对齐漏报数不匹配");
  await input.setInputFiles(
    "../outputs/valve-multiseed-study/runs/20261031T204038Z-a0dab190/metrics.json",
  );
  await page.getByText("20261031T204038Z-a0dab190", { exact: true }).waitFor();
  await page
    .getByLabel("报告留出协议", { exact: true })
    .selectOption("historical-test");
  await page.getByLabel("报告种子", { exact: true }).selectOption("44");
  if ((await rows.count()) !== 9) throw Error("阀门多种子报告支路不完整");
  if ((await rows.nth(0).getByRole("cell").nth(3).innerText()) !== "12")
    throw Error("原AE漏报数不匹配");
  if ((await rows.nth(6).getByRole("cell").nth(3).innerText()) !== "0")
    throw Error("双确认漏报数不匹配");
  await input.setInputFiles(
    "../outputs/motor-cnn-study/runs/20261031T203355Z-d5109655/metrics.json",
  );
  await page.getByText("20261031T203355Z-d5109655", { exact: true }).waitFor();
  await page.getByRole("status").filter({ hasText: "源工况候选" }).waitFor();
  await page.getByLabel("报告评价分区", { exact: true }).selectOption("source");
  await rows.getByText("源工况留出验证", { exact: true }).waitFor();
  await page
    .getByLabel("报告留出协议", { exact: true })
    .selectOption("held-out-torque-profile");
  if (
    (await page.getByLabel("报告评价分区", { exact: true }).inputValue()) !==
    "target"
  )
    throw Error("切换协议没有复位测试分区");
  await page.getByRole("status").filter({ hasText: "无候选通过" }).waitFor();
  await page.screenshot({ path: "motor-source-selection.png", fullPage: true });
  await input.setInputFiles(
    "../outputs/valve-stress-study/runs/20261031T204218Z-227e3ce8/metrics.json",
  );
  await page.getByRole("note").filter({ hasText: "合成扰动报告" }).waitFor();
  await page
    .getByLabel("报告留出协议", { exact: true })
    .selectOption("时间平移");
  const shifted = rows.filter({ hasText: "PS1＋PS2 -50 ms" });
  if ((await shifted.getByRole("cell").nth(3).innerText()) !== "44")
    throw Error("时间平移漏报数未显示");
  await page.screenshot({ path: "valve-stress-report.png", fullPage: true });
  await input.setInputFiles(
    "../outputs/stiction-study/runs/20261031T205149Z-5b867fad/metrics.json",
  );
  await page.getByText("20261031T205149Z-5b867fad", { exact: true }).waitFor();
  if ((await rows.count()) !== 4) throw Error("黏滞对照支路不完整");
  await page.getByLabel("报告种子", { exact: true }).selectOption("44");
  if ((await rows.count()) !== 3) throw Error("未训练种子错误显示逻辑回归结果");
  await rows
    .nth(1)
    .getByText(/回路等权：.*66.21%/)
    .waitFor();
  if ((await rows.nth(1).getByRole("cell").nth(7).innerText()) !== "72 / 80")
    throw Error("黏滞支持范围错误");
  await page.setViewportSize({ width: 390, height: 844 });
  if (
    await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)
  )
    throw Error("黏滞加权结果窄屏溢出");
  await page.screenshot({ path: "stiction-report-mobile.png", fullPage: true });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await input.setInputFiles(
    "../outputs/actuator-mechanism-study/runs/20261031T213509Z-630e85e4/metrics.json",
  );
  await page.getByText("20261031T213509Z-630e85e4", { exact: true }).waitFor();
  if ((await rows.count()) !== 4) throw Error("运动回归对照行数错误");
  await rows
    .getByRole("cell", { name: "健康运动回归残差 直接分类", exact: true })
    .waitFor();
  if ((await rows.nth(2).getByRole("cell").nth(3).innerText()) !== "26")
    throw Error("运动回归漏报数不匹配");
  await input.setInputFiles(
    "../outputs/commissioning-budget-study/runs/20261031T213932Z-e027a651/metrics.json",
  );
  await page.getByText("20261031T213932Z-e027a651", { exact: true }).waitFor();
  if ((await rows.count()) !== 6) throw Error("健康参考预算行数错误");
  await rows
    .getByRole("cell", {
      name: "工况尺度收缩 · 6条健康参考 直接分类",
      exact: true,
    })
    .waitFor();
  if ((await rows.nth(5).getByRole("cell").nth(1).innerText()) !== "8 / 240")
    throw Error("固定测试分母不匹配");
  await input.setInputFiles(
    "../outputs/motor-binary-study/runs/20261031T211915Z-ed950cdf/metrics.json",
  );
  await page.getByText("20261031T211915Z-ed950cdf", { exact: true }).waitFor();
  await rows
    .getByRole("cell", { name: "无量纲机理 · 二分类RF 直接分类", exact: true })
    .waitFor();
  if ((await rows.nth(1).getByRole("cell").nth(2).innerText()) !== "4")
    throw Error("电机二分类误报数不匹配");
  // 下列小对象仅测试覆盖率展示，不属于实验结果。
  const coverageFixture = {
    run_id: "ui-coverage-fixture",
    classes: ["健康", "故障"],
    protocols: {
      fixture: {
        target: {
          spectral_cnn: {
            42: {
              normal: 1,
              fault: 1,
              false_positives: 0,
              false_negatives: 0,
              macro_f1: 1,
              records: 3,
              supported_records: 2,
            },
          },
        },
      },
    },
  };
  await input.setInputFiles({
    name: "coverage-fixture.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(coverageFixture)),
  });
  await page.getByText("ui-coverage-fixture", { exact: true }).waitFor();
  if ((await rows.nth(0).getByRole("cell").nth(7).innerText()) !== "2 / 3")
    throw Error("不支持记录没有进入覆盖率分母");
  await page.getByText("2类诊断标签", { exact: true }).waitFor();
  coverageFixture.protocols.fixture.target.spectral_cnn[
    "42"
  ].supported_records = 3;
  await input.setInputFiles({
    name: "invalid-coverage.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(coverageFixture)),
  });
  await page
    .getByRole("alert")
    .filter({ hasText: "记录覆盖计数不一致" })
    .waitFor();
  await input.setInputFiles({
    name: "invalid.json",
    mimeType: "application/json",
    buffer: Buffer.from('{"run_id":"bad","protocols":{}}'),
  });
  await page
    .getByRole("alert")
    .filter({ hasText: "报告没有工况协议" })
    .waitFor();
  if (await rows.count()) throw Error("坏报告仍显示旧结果");
  return {
    checks: [
      "phase_metrics",
      "seed_switch",
      "gate_comparison",
      "protocol_switch",
      "mobile_layout",
      "valve_binary_metrics",
      "deep_alignment_metrics",
      "valve_training_seeds",
      "motor_source_selection_and_partitions",
      "synthetic_stress_separated_from_real_evidence",
      "stiction_loop_weighting_and_coverage",
      "actuator_mechanical_residual_metrics",
      "commissioning_fixed_test_denominator",
      "motor_binary_metrics",
      "coverage_and_class_labels",
      "inconsistent_coverage_rejected",
      "invalid_report_state",
    ],
  };
}
