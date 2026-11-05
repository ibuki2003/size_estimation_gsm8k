import { parse } from 'csv-parse/sync';

const TOTAL = 1319;

// With t = sqrt(df) * tan(theta), the t density is proportional to cos(theta)^(df - 1).
function studentT975(df) {
  const integral = end => {
    const steps = 2048;
    const h = end / steps;
    let sum = 0;
    for (let i = 0; i <= steps; i++) {
      sum += (i === 0 || i === steps ? 1 : i % 2 ? 4 : 2)
        * Math.cos(i * h) ** (df - 1);
    }
    return sum * h / 3;
  };
  const target = integral(Math.PI / 2) * 0.95;
  let low = 0;
  let high = Math.PI / 2;
  for (let i = 0; i < 50; i++) {
    const mid = (low + high) / 2;
    if (integral(mid) < target) low = mid;
    else high = mid;
  }
  return Math.sqrt(df) * Math.tan((low + high) / 2);
}

// Bare numbers are parameter counts; K/M/B/T suffixes are also accepted.
function parameters(value, row, column) {
  if (!value.trim()) return null;
  const match = value.trim().match(/^(\d+(?:\.\d+)?|\.\d+)\s*([kmbt]?)$/i);
  const result = match
    ? Number(match[1]) * 10 ** ({ '': 0, k: 3, m: 6, b: 9, t: 12 }[match[2].toLowerCase()])
    : NaN;
  if (!Number.isFinite(result) || result <= 0) {
    throw new Error(`CSV row ${row}: invalid ${column}: ${value}`);
  }
  return result;
}

export function buildData(csv) {
  const records = parse(csv, {
    columns: true,
    skip_empty_lines: true,
    bom: true,
    relax_column_count_less: true,
  });
  const models = records.flatMap((record, index) => {
    if (record.effort?.trim() !== 'none' || !record.correct?.trim()) return [];
    const correct = Number(record.correct);
    if (!Number.isFinite(correct) || correct < 0) return [];
    if (!Number.isInteger(correct) || correct > TOTAL) {
      throw new Error(`CSV row ${index + 2}: invalid correct: ${record.correct}`);
    }
    return [{
      model: record.model,
      author: record.model_author,
      provider: record.provider,
      timestamp: record.timestamp,
      correct,
      accuracy: correct / TOTAL * 100,
      parameters: parameters(record.parameters ?? '', index + 2, 'parameters'),
      active_parameters: parameters(record.active_parameters ?? '', index + 2, 'active_parameters'),
    }];
  });
  return {
    total: TOTAL,
    models,
    plots: Object.fromEntries(['parameters', 'active_parameters'].map(key => [key, fit(models, key)])),
  };
}

function fit(models, key) {
  const known = models.filter(model => model[key] !== null);
  if (known.length < 2) return { knownCount: known.length, regression: null };
  const xs = known.map(model => Math.log10(model[key]));
  const meanX = xs.reduce((a, b) => a + b, 0) / xs.length;
  const meanY = known.reduce((sum, model) => sum + model.accuracy, 0) / known.length;
  const variance = xs.reduce((sum, x) => sum + (x - meanX) ** 2, 0);
  if (variance === 0) return { knownCount: known.length, regression: null };
  const slope = xs.reduce((sum, x, i) => sum + (x - meanX) * (known[i].accuracy - meanY), 0) / variance;
  const intercept = meanY - slope * meanX;
  const residual = known.reduce((sum, model, i) =>
    sum + (model.accuracy - (slope * xs[i] + intercept)) ** 2, 0);
  const total = known.reduce((sum, model) => sum + (model.accuracy - meanY) ** 2, 0);
  const prediction = known.length > 2 ? {
    meanX,
    meanY,
    variance,
    count: known.length,
    squaredMargin: studentT975(known.length - 2) ** 2 * residual / (known.length - 2),
  } : null;
  const estimates = models.filter(model => model[key] === null).map(model => {
    const value = slope === 0 ? NaN : 10 ** ((model.accuracy - intercept) / slope);
    let logSizeInterval = null;
    if (prediction) {
      // Invert (y - fitted(x))² <= t² s² (1 + 1/n + (x - meanX)² / Sxx).
      const k = prediction.squaredMargin;
      const a = slope ** 2 - k / variance;
      const delta = model.accuracy - meanY;
      if (a > 0) {
        const center = slope * delta / a;
        const halfWidth = Math.sqrt(k * (delta ** 2 / variance + a * (1 + 1 / known.length))) / a;
        logSizeInterval = [meanX + center - halfWidth, meanX + center + halfWidth];
      }
    }
    return { ...model, estimatedSize: Number.isFinite(value) && value > 0 ? value : null, logSizeInterval };
  });
  return {
    knownCount: known.length,
    regression: { slope, intercept, r2: total === 0 ? null : 1 - residual / total, prediction },
    estimates,
  };
}
