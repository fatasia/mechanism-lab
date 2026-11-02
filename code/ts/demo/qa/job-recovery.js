// QA服务读取测试，不启动训练。首次请求断网后应自动恢复真实任务状态。
async function check(page) {
  let calls = 0;
  await page.route("**/api/job", async (route) => {
    calls++;
    if (calls === 1) await route.abort("internetdisconnected");
    else await route.continue();
  });
  await page.reload();
  const warning = page
    .getByRole("status")
    .filter({ hasText: "训练状态连接中断" });
  await warning.waitFor();
  await warning.waitFor({ state: "hidden" });
  if (calls < 2) throw Error("任务状态未重试");
  await page.unroute("**/api/job");
  return { check: "job_status_recovers_after_read_failure", calls };
}
