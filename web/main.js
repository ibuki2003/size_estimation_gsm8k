import * as echarts from 'echarts/core';
import { ScatterChart, LineChart, CustomChart } from 'echarts/charts';
import { GridComponent, LegendComponent, TooltipComponent, ToolboxComponent, DataZoomComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import './style.css';

echarts.use([ScatterChart, LineChart, CustomChart, GridComponent, LegendComponent, TooltipComponent, ToolboxComponent, DataZoomComponent, CanvasRenderer]);

const authorColors = new Map(Object.entries({
  openai: '#10a37f',
  anthropic: '#c07858',
  google: '#4285f4',
  deepseek: '#4d6bfe',
  zai: '#303030',
  moonshotai: '#6850d8',
  qwen: '#7c3aed',
  xiaomi: '#ff6900',
}));

function authorColor(author) {
  const key = author?.trim().toLowerCase() || '';
  if (!key) return '#8a969b';
  if (authorColors.has(key)) return authorColors.get(key);
  let hash = 0;
  for (const char of key) hash = (Math.imul(hash, 31) + char.codePointAt(0)) | 0;
  return `hsl(${(hash >>> 0) % 360}, 60%, 43%)`;
}

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
      itemStyle: { color: authorColor(model.author) },
    }));
    const estimated = (plot.estimates ?? []).filter(model => model.estimatedSize !== null).map(model => ({
      value: [Math.log10(model.estimatedSize), model.accuracy], model, estimated: true,
      itemStyle: { color: '#fff', borderColor: authorColor(model.author), borderWidth: 2 },
    }));
    const unknownCount = data.models.length - known.length;
    card.querySelector('.note').textContent = `${unknownCount} 件がサイズ不明${regression ? ` / ${estimated.length} 件を回帰から推定` : '（回帰未成立のため未描画）'}。ドラッグで範囲を選択してズーム。下部スライダーでも範囲を調整できます。`;
    const prediction = regression?.prediction;
    card.querySelector('.note').textContent += prediction
      ? ' 帯は個別モデルの95%予測区間です。逆算したサイズ推定区間は推定点のツールチップに表示します。既知モデルと同じ関係・等分散の正規誤差を仮定しています。'
      : ' 95%予測区間の計算には、サイズが異なる既知データが3件以上必要です。';
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
          if (point.estimated) {
            element.textContent += !prediction ? '\n95%サイズ推定区間: 算出不可'
              : model.logSizeInterval
                ? `\n95%サイズ推定区間: ${model.logSizeInterval.map(x => formatSize(10 ** x)).join(' ～ ')}`
                : '\n95%サイズ推定区間: 有限区間に絞れません';
          }
          return element;
        },
      },
      xAxis: { type: 'value', min, max, name: 'Parameters', nameLocation: 'middle', nameGap: 32, axisLabel: { formatter: value => formatSize(10 ** value) }, splitLine: { lineStyle: { color: '#edf0ed' } } },
      yAxis: { type: 'value', min: 0, max: 100, name: 'Accuracy (%)', splitLine: { lineStyle: { color: '#edf0ed' } } },
      toolbox: { right: 5, feature: { dataZoom: { title: { zoom: '領域ズーム', back: '戻る' } } } },
      dataZoom: [
        { type: 'slider', xAxisIndex: 0, filterMode: 'weakFilter', bottom: 12, height: 22, labelFormatter: value => formatSize(10 ** value) },
        { type: 'inside', xAxisIndex: 0, filterMode: 'weakFilter', zoomOnMouseWheel: false, moveOnMouseMove: false },
        { type: 'inside', yAxisIndex: 0, zoomOnMouseWheel: false, moveOnMouseMove: false },
      ],
      series: [
        ...(prediction ? [{
          name: '95%予測区間', type: 'custom', silent: true, clip: true,
          data: [[min, max]], encode: { x: [0, 1], y: [] }, z: 0,
          renderItem: (params, api) => {
            const edges = [-1, 1].map(sign => line.map(([x, y]) => api.coord([
              x, y + sign * Math.sqrt(prediction.squaredMargin
                * (1 + 1 / prediction.count + (x - prediction.meanX) ** 2 / prediction.variance)),
            ])));
            return { type: 'polygon', shape: { points: [...edges[0], ...edges[1].reverse()] }, style: { fill: 'rgba(25,126,116,0.12)' } };
          },
        }] : []),
        { name: '実測サイズ', type: 'scatter', symbolSize: 12, data: known },
        { name: '推定サイズ', type: 'scatter', symbol: 'diamond', symbolSize: 14, itemStyle: { color: '#fff', borderColor: '#8a969b', borderWidth: 2 }, data: estimated },
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
