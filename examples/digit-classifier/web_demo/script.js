const ARTIFACT_CONFIG = {
  artifactRoot: document.body.dataset.artifactRoot || '.',
  metricsFile: document.body.dataset.metricsFile || 'metrics.json',
  predictionsFile: document.body.dataset.predictionsFile || 'predictions.json',
  lossCurvesFile: document.body.dataset.lossCurvesFile || 'loss_curves.json',
  metadataFile: document.body.dataset.metadataFile || 'run_metadata.json',
  maxPredictionRows: parseInt(document.body.dataset.predictionRows || '25', 10),
};

async function fetchJson(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

function formatNumber(value, fractionDigits = 4) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '–';
  }
  if (Math.abs(value) >= 1) {
    return value.toFixed(Math.min(fractionDigits, 2));
  }
  return value.toFixed(fractionDigits);
}

function formatPercentage(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '–';
  }
  return `${(value * 100).toFixed(2)}%`;
}

function setStatus(message, level = 'info') {
  const banner = document.querySelector('[data-role="status-banner"]');
  if (!banner) return;
  banner.textContent = message;
  banner.dataset.level = level;
}

function fillMetricElements(metrics) {
  document.querySelectorAll('[data-metric-key]').forEach((el) => {
    const key = el.getAttribute('data-metric-key');
    if (!key) return;
    const raw = metrics[key];
    el.textContent = typeof raw === 'number' ? formatNumber(raw) : String(raw ?? '–');
  });
}

function renderMetricsTable(metrics) {
  const tableBody = document.querySelector('[data-role="metrics-table"]');
  if (!tableBody) return;

  tableBody.innerHTML = '';
  const entries = Object.entries(metrics || {});
  if (!entries.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 2;
    cell.textContent = 'Metric summary unavailable.';
    row.appendChild(cell);
    tableBody.appendChild(row);
    return;
  }
  entries.sort(([aKey], [bKey]) => aKey.localeCompare(bKey));

  for (const [key, value] of entries) {
    const row = document.createElement('tr');
    const keyCell = document.createElement('th');
    keyCell.scope = 'row';
    keyCell.textContent = key;
    const valueCell = document.createElement('td');
    valueCell.textContent = typeof value === 'number' ? formatNumber(value, 6) : String(value);
    row.appendChild(keyCell);
    row.appendChild(valueCell);
    tableBody.appendChild(row);
  }
}

function renderLossCurves(curves) {
  const tableBody = document.querySelector('[data-role="loss-table"]');
  if (!tableBody) return;

  tableBody.innerHTML = '';
  const epochs = Array.isArray(curves?.epochs) ? curves.epochs : [];
  if (!epochs.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 5;
    cell.textContent = 'Loss curve data unavailable.';
    row.appendChild(cell);
    tableBody.appendChild(row);
    return;
  }
  for (let i = 0; i < epochs.length; i += 1) {
    const row = document.createElement('tr');

    const epochCell = document.createElement('th');
    epochCell.scope = 'row';
    epochCell.textContent = epochs[i];
    row.appendChild(epochCell);

    const columns = ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'];
    columns.forEach((column) => {
      const cell = document.createElement('td');
      const series = curves[column];
      const value = Array.isArray(series) ? series[i] : undefined;
      cell.textContent = column.includes('accuracy') ? formatPercentage(value) : formatNumber(value, 6);
      row.appendChild(cell);
    });

    tableBody.appendChild(row);
  }
}

function renderRunMetadata(metadata) {
  const container = document.querySelector('[data-role="metadata-json"]');
  if (container) {
    container.textContent = JSON.stringify(metadata, null, 2);
  }

  document.querySelectorAll('[data-metadata-key]').forEach((el) => {
    const key = el.getAttribute('data-metadata-key');
    if (!key) return;

    const segments = key.split('.');
    let cursor = metadata;
    for (const segment of segments) {
      if (cursor && typeof cursor === 'object' && segment in cursor) {
        cursor = cursor[segment];
      } else {
        cursor = undefined;
        break;
      }
    }

    if (typeof cursor === 'object') {
      el.textContent = JSON.stringify(cursor, null, 2);
    } else if (cursor !== undefined) {
      el.textContent = String(cursor);
    }
  });
}

function renderPredictions(predictions) {
  const tableBody = document.querySelector('[data-role="predictions-table"]');
  if (!tableBody) return;

  tableBody.innerHTML = '';
  const samples = Array.isArray(predictions?.samples) ? predictions.samples : [];
  const limit = Math.max(1, ARTIFACT_CONFIG.maxPredictionRows || 25);
  if (!samples.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.colSpan = 6;
    cell.textContent = 'No prediction samples were found in predictions.json.';
    row.appendChild(cell);
    tableBody.appendChild(row);
    return;
  }

  samples.slice(0, limit).forEach((sample) => {
    const row = document.createElement('tr');
    const indexCell = document.createElement('th');
    indexCell.scope = 'row';
    indexCell.textContent = sample.index;

    const datasetCell = document.createElement('td');
    datasetCell.textContent =
      sample.dataset_index !== undefined ? sample.dataset_index : '–';

    const trueCell = document.createElement('td');
    trueCell.textContent = sample.true_label;

    const predictedCell = document.createElement('td');
    predictedCell.textContent = sample.predicted_label;

    const confidenceCell = document.createElement('td');
    const probs = Array.isArray(sample.probabilities) ? sample.probabilities : [];
    const predictedIdx = typeof sample.predicted_label === 'number' ? sample.predicted_label : probs.indexOf(Math.max(...probs));
    const confidence = predictedIdx >= 0 && probs[predictedIdx] !== undefined ? probs[predictedIdx] : undefined;
    confidenceCell.textContent = formatPercentage(confidence);

    const sparklineCell = document.createElement('td');
    if (probs.length) {
      const span = document.createElement('span');
      span.classList.add('probabilities');
      span.textContent = probs.map((value, idx) => `${idx}:${formatPercentage(value)}`).join(' ');
      sparklineCell.appendChild(span);
    } else {
      sparklineCell.textContent = '–';
    }

    row.appendChild(indexCell);
    row.appendChild(datasetCell);
    row.appendChild(trueCell);
    row.appendChild(predictedCell);
    row.appendChild(confidenceCell);
    row.appendChild(sparklineCell);

    if (sample.true_label !== sample.predicted_label) {
      row.classList.add('is-mismatch');
    }

    tableBody.appendChild(row);
  });

  if (samples.length > limit) {
    const remainder = samples.length - limit;
    const summaryRow = document.createElement('tr');
    const summaryCell = document.createElement('td');
    summaryCell.colSpan = 6;
    summaryCell.classList.add('table-note');
    summaryCell.textContent = `${remainder} additional sample${remainder === 1 ? '' : 's'} not shown. Adjust data-prediction-rows to see more.`;
    summaryRow.appendChild(summaryCell);
    tableBody.appendChild(summaryRow);
  }
}

async function loadArtifacts() {
  const base = ARTIFACT_CONFIG.artifactRoot.replace(/\/$/, '');
  const paths = {
    metrics: `${base}/${ARTIFACT_CONFIG.metricsFile}`,
    predictions: `${base}/${ARTIFACT_CONFIG.predictionsFile}`,
    lossCurves: `${base}/${ARTIFACT_CONFIG.lossCurvesFile}`,
    metadata: `${base}/${ARTIFACT_CONFIG.metadataFile}`,
  };

  const [metrics, predictions, lossCurves, metadata] = await Promise.all([
    fetchJson(paths.metrics),
    fetchJson(paths.predictions),
    fetchJson(paths.lossCurves),
    fetchJson(paths.metadata),
  ]);

  return { metrics, predictions, lossCurves, metadata };
}

async function initialiseDashboard() {
  try {
    setStatus('Loading artifacts…', 'info');
    const artifacts = await loadArtifacts();
    setStatus('Artifacts loaded', 'success');

    fillMetricElements(artifacts.metrics);
    renderMetricsTable(artifacts.metrics);
    renderLossCurves(artifacts.lossCurves);
    renderRunMetadata(artifacts.metadata);
    renderPredictions(artifacts.predictions);
  } catch (error) {
    console.error(error);
    setStatus(`Unable to load artifacts: ${error.message}`, 'error');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initialiseDashboard();
});
