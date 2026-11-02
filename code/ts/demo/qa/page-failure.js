// 首次按需模块加载失败的恢复流程；不提交业务数据。
async function check(page) {
  await page.route("**/assets/Analysis-*.js", route => route.abort("internetdisconnected"));
  await page.reload();
  await page.getByRole("heading", {name:"周期异常分数",exact:true}).waitFor();
  await page.getByRole("button", {name:"02部件分析",exact:true}).click();
  await page.getByRole("heading", {name:"页面暂时不可用",exact:true}).waitFor();
  await page.screenshot({path:"page-module-failure.png",fullPage:true});
  await page.getByRole("button", {name:"01总览",exact:true}).click();
  await page.getByRole("heading", {name:"周期异常分数",exact:true}).waitFor();
  await page.getByRole("button", {name:"02部件分析",exact:true}).click();
  await page.getByRole("heading", {name:"页面暂时不可用",exact:true}).waitFor();
  await page.unroute("**/assets/Analysis-*.js");
  await page.getByRole("button", {name:"重新加载应用",exact:true}).click();
  await page.getByRole("heading", {name:"周期异常分数",exact:true}).waitFor();
  await page.getByRole("button", {name:"02部件分析",exact:true}).click();
  await page.getByRole("heading", {name:"换向机理证据 · 独立于 AE 告警",exact:true}).waitFor();
  return {checks:["failed_chunk_has_recovery_ui","navigation_survives","reload_restores_failed_page"]};
}
