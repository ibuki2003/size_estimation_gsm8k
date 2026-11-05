import * as echarts from 'echarts/core';
import { ScatterChart, LineChart } from 'echarts/charts';
import { GridComponent, LegendComponent, TooltipComponent, ToolboxComponent, DataZoomComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import './style.css';

echarts.use([ScatterChart, LineChart, GridComponent, LegendComponent, TooltipComponent, ToolboxComponent, DataZoomComponent, CanvasRenderer]);

const formatSize = value => {
  for (const [unit, scale] of [['T', 1e12], ['B', 1e9], ['M', 1e6], ['K', 1e3]]) {
    if (value >= scale) return `${Number((value / scale).toPrecision(3))}${unit}`;
  }
  return Number(value.toPrecision(3)).toString();
};

async function main() {
  const response = await fetch(`${import.meta.env.BASE_URL}scaling-data.json`);
  if (!response.ok) throw new Error(`データを読み込めませんでした (${response.status})`);
  const data = await response.json();
  document.querySelector('#count').textContent = `${data.models.length} measurements`;
  for (const [key, title, subtitle] of [
    ['parameters', 'Total parameters', '総パラメータ数'],
    ['active_parameters', 'Active parameters', '有効パラメータ数'],
  ]) {
    const plot = data.plots[key];
    const card = document.createElement('article');
    card.innerHTML = `<div class="card-header"><div><h2>${title}</h2><p>${subtitle}</p></div><button>ズームをリセット</button></div><div class="stats"></div><div class="chart"></div><p class="note"></p>`;
    document.querySelector('#plots').append(card);
    const regression = plot.regression;
    card.querySelector('.stats').textContent = regression
      ? `既知 ${plot.knownCount} 件　·　accuracy = ${regression.slope.toFixed(3)} × log₁₀(N) ${regression.intercept < 0 ? '−' : '+'} ${Math.abs(regression.intercept).toFixed(3)}　·　R² ${regression.r2?.toFixed(3) ?? '—'}`
      : `既知 ${plot.knownCount} 件　·　回帰にはサイズが異なる既知データが2件以上必要です`;
    const known = data.models.filter(model => model[key] !== null).map(model => ({
      value: [Math.log10(model[key]), model.accuracy], model, estimated: false,
    }));
    const estimated = (plot.estimates ?? []).filter(model => model.estimatedSize !== null).map(model => ({
      value: [Math.log10(model.estimatedSize), model.accuracy], model, estimated: true,
    }));
    const unknownCount = data.models.length - known.length;
    card.querySelector('.note').textContent = `${unknownCount} 件がサイズ不明${regression ? ` / ${estimated.length} 件を回帰から推定` : '（回帰未成立のため未描画）'}。ドラッグで範囲を選択してズーム。下部スライダーでも範囲を調整できます。`;
    const sizes = [...known, ...estimated].map(point => point.value[0]);
    const min = sizes.length ? Math.min(...sizes) - Math.log10(1.8) : 9;
    const max = sizes.length ? Math.max(...sizes) + Math.log10(1.8) : 12;
    const line = regression ? Array.from({ length: 100 }, (_, i) => {
      const x = min + (max - min) * i / 99;
      return [x, regression.slope * x + regression.intercept];
    }) : [];
    const chart = echarts.init(card.querySelector('.chart'));
    chart.setOption({
      color: ['#197e74', '#c17a37', '#8a969b'],
      animation: false,
      grid: { left: 65, right: 35, top: 55, bottom: 90 },
      legend: { top: 10, data: ['実測サイズ', '推定サイズ', '回帰直線'] },
      tooltip: {
        trigger: 'item',
        renderMode: 'html',
        formatter: params => {
          const point = params.data;
          if (!point.model) return '';
          const model = point.model;
          const element = document.createElement('div');
          element.style.whiteSpace = 'pre-line';
          element.textContent = `${model.model}\n${point.estimated ? '推定' : '実測'}サイズ: ${formatSize(point.estimated ? model.estimatedSize : model[key])}\n正答率: ${model.accuracy.toFixed(2)}% (${model.correct}/${data.total})\n提供: ${model.provider || '—'} / 作者: ${model.author || '—'}\n${model.timestamp || ''}`;
          return element;
        },
      },
      xAxis: { type: 'value', min, max, name: 'Parameters', nameLocation: 'middle', nameGap: 32, axisLabel: { formatter: value => formatSize(10 ** value) }, splitLine: { lineStyle: { color: '#edf0ed' } } },
      yAxis: { type: 'value', min: 0, max: 100, name: 'Accuracy (%)', splitLine: { lineStyle: { color: '#edf0ed' } } },
      toolbox: { right: 5, feature: { dataZoom: { title: { zoom: '領域ズーム', back: '戻る' } } } },
      dataZoom: [
        { type: 'slider', xAxisIndex: 0, bottom: 12, height: 22, labelFormatter: value => formatSize(10 ** value) },
        { type: 'inside', xAxisIndex: 0, zoomOnMouseWheel: false, moveOnMouseMove: false },
        { type: 'inside', yAxisIndex: 0, zoomOnMouseWheel: false, moveOnMouseMove: false },
      ],
      series: [
        { name: '実測サイズ', type: 'scatter', symbolSize: 12, data: known },
        { name: '推定サイズ', type: 'scatter', symbol: 'diamond', symbolSize: 14, itemStyle: { color: '#fff', borderColor: '#c17a37', borderWidth: 2 }, data: estimated },
        { name: '回帰直線', type: 'line', showSymbol: false, silent: true, lineStyle: { type: 'dashed', width: 1.5 }, data: line },
      ],
    });
    chart.dispatchAction({ type: 'takeGlobalCursor', key: 'dataZoomSelect', dataZoomSelectActive: true });
    card.querySelector('button').addEventListener('click', () => chart.dispatchAction({
      type: 'dataZoom', batch: [{ dataZoomIndex: 0, start: 0, end: 100 }, { dataZoomIndex: 1, start: 0, end: 100 }, { dataZoomIndex: 2, start: 0, end: 100 }],
    }));
    new ResizeObserver(() => chart.resize()).observe(card);
  }
}

main().catch(error => {
  document.querySelector('#plots').textContent = error.message;
});
