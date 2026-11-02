import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { LineChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  MarkLineComponent,
} from "echarts/components";
import { SVGRenderer } from "echarts/renderers";
import type { EChartsOption } from "echarts";
echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  MarkLineComponent,
  SVGRenderer,
]);

/** 图表色板从 CSS 令牌读取，随主题切换而变化。 */
export function palette() {
  const style = getComputedStyle(document.documentElement);
  const read = (name: string) => style.getPropertyValue(name).trim();
  return {
    primary: read("--accent"),
    secondary: read("--warning"),
    danger: read("--danger"),
    info: read("--info"),
    muted: read("--text-muted"),
    grid: read("--line"),
  };
}

/** 图表只在挂载时创建，数据更新不销毁缩放和交互状态。 */
export function Chart({
  option,
  onSelect,
  label,
}: {
  option: EChartsOption;
  onSelect?: (index: number) => void;
  label: string;
}) {
  const element = useRef<HTMLDivElement>(null),
    chart = useRef<echarts.ECharts | null>(null),
    select = useRef(onSelect);
  select.current = onSelect;
  useEffect(() => {
    const instance = echarts.init(element.current!, undefined, {
      renderer: "svg",
    });
    chart.current = instance;
    instance.on("click", (p) => select.current?.(p.dataIndex));
    const observer = new ResizeObserver(() => instance.resize());
    observer.observe(element.current!);
    return () => {
      observer.disconnect();
      instance.dispose();
      chart.current = null;
    };
  }, []);
  useEffect(() => {
    if (Object.keys(option).length === 0) {
      chart.current?.clear();
      return;
    }
    chart.current?.setOption(option, { lazyUpdate: true });
  }, [option]);
  return <div role="img" aria-label={label} className="chart" ref={element} />;
}
export function lines(
  labels: number[],
  series: { name: string; data: (number | null)[]; color: string }[],
  threshold?: number,
  axis = "周期编号",
  unit = "",
  formatLabel?: (value: string) => string,
): EChartsOption {
  const colors = palette();
  return {
    animation: false,
    textStyle: { fontFamily: "inherit" },
    tooltip: {
      trigger: "axis",
      renderMode: "richText",
      formatter: (params) => {
        const list = Array.isArray(params) ? params : [params];
        const name = String(list[0].name);
        return [
          formatLabel ? formatLabel(name) : name,
          ...list.map(
            (p) =>
              `${p.seriesName}  ${typeof p.value === "number" ? Number(p.value.toPrecision(5)) : String(p.value ?? "—")}`,
          ),
        ].join("\n");
      },
    },
    legend: { bottom: 0, textStyle: { color: colors.muted } },
    grid: { left: 65, right: 70, top: 35, bottom: 70 },
    xAxis: {
      type: "category",
      name: axis,
      data: labels,
      boundaryGap: false,
      nameLocation: "middle",
      nameGap: 28,
      nameTextStyle: { color: colors.muted },
      axisLine: { lineStyle: { color: colors.grid } },
      axisLabel: {
        color: colors.muted,
        hideOverlap: true,
        // 采样时刻由浮点加法生成，显示到 0.01 秒以免长尾挤占窄屏坐标轴。
        formatter:
          formatLabel ??
          ((value: string) => String(Number(Number(value).toFixed(2)))),
      },
    },
    yAxis: {
      type: "value",
      name: unit,
      nameTextStyle: { color: colors.muted },
      axisLabel: { color: colors.muted },
      scale: true,
      splitLine: { lineStyle: { color: colors.grid } },
    },
    dataZoom: [{ id: "inside", type: "inside" }],
    series: series.map((s, i) => ({
      id: s.name,
      name: s.name,
      type: "line",
      data: s.data,
      showSymbol: false,
      lineStyle: { width: 1.6, color: s.color },
      itemStyle: { color: s.color },
      markLine:
        i === 0 && threshold !== undefined
          ? {
              symbol: "none",
              data: [{ yAxis: threshold }],
              lineStyle: { color: colors.danger },
              label: {
                formatter: "阈值 {c}",
                color: colors.danger,
                position: "end",
              },
            }
          : undefined,
    })),
  };
}
