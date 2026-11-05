import { parse } from 'csv-parse/sync';

const TOTAL = 1319;

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
  const estimates = models.filter(model => model[key] === null).map(model => {
    const value = slope === 0 ? NaN : 10 ** ((model.accuracy - intercept) / slope);
    return { ...model, estimatedSize: Number.isFinite(value) && value > 0 ? value : null };
  });
  return {
    knownCount: known.length,
    regression: { slope, intercept, r2: total === 0 ? null : 1 - residual / total },
    estimates,
  };
}
