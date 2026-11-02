import { Component, type ReactNode } from "react";

// 页面失败不卸载导航和训练状态；失败的ES模块需要重新加载应用才能再次导入。
export class PageBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (this.state.failed)
      return (
        <section className="panel" role="alert">
          <h2>页面暂时不可用</h2>
          <p>页面发生错误，请重新加载应用。</p>
          <button onClick={() => window.location.reload()}>重新加载应用</button>
        </section>
      );
    return this.props.children;
  }
}
