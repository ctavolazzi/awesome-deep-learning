const timelineTemplate = [
  {
    title: '01 · Data loading',
    description: 'The digits dataset ships with scikit-learn and is flattened to 64 features.',
    file: 'examples/digit-classifier/run_demo.py',
    hint: 'Swap in your own loader, preserve the JSON outputs, and the UI keeps working.'
  },
  {
    title: '02 · Train / validation split',
    description: 'A stratified split maintains class balance between training and evaluation sets.',
    file: 'examples/digit-classifier/run_demo.py',
    hint: 'Experiment with different test sizes when comparing generalization.'
  },
  {
    title: '03 · Model training',
    description: 'The pipeline scales inputs with StandardScaler before fitting MLPClassifier.',
    file: 'examples/digit-classifier/run_demo.py',
    hint: 'Tweak hidden layer widths or learning rate and rerun to refresh this dashboard.'
  },
  {
    title: '04 · Evaluation',
    description: 'Metrics, confusion matrix, and representative samples are exported as JSON.',
    file: 'examples/digit-classifier/run_demo.py',
    hint: 'Wire these artifacts into MLFlow or wandb for longer-running projects.'
  },
  {
    title: '05 · Reporting',
    description: 'This dashboard reads the artifacts to build a shareable status report.',
    file: 'examples/digit-classifier/web_demo/',
    hint: 'Host the folder behind a static server or publish via GitHub Pages.'
  }
];

let dashboardData = null;
let runIndexData = null;
let artifactRoot = null;
let activeRunPath = null;
let activeRunId = null;

const DEFAULT_ARTIFACT_ROOT = './artifacts';
const DEFAULT_ARTIFACT_PATH = `${DEFAULT_ARTIFACT_ROOT}/latest`;
const RUN_INDEX_FILE = 'index.json';

function determineArtifactBasePath() {
  const url = new URL(window.location.href);
  const queryOverride = url.searchParams.get('artifacts');
  if (queryOverride) {
    return queryOverride.replace(/\/$/, '');
  }

  const attr = document.body.dataset.artifactPath;
  if (attr) {
    return attr.replace(/\/$/, '');
  }

  return DEFAULT_ARTIFACT_PATH;
}

function deriveArtifactRoot(basePath) {
  const normalized = (basePath || DEFAULT_ARTIFACT_PATH).replace(/\/$/, '');
  const parts = normalized.split('/');
  if (parts.length <= 1) {
    return DEFAULT_ARTIFACT_ROOT;
  }
  parts.pop();
  const joined = parts.join('/');
  return joined || DEFAULT_ARTIFACT_ROOT;
}

function buildRunPath(artifactDir) {
  if (!artifactDir) {
    return null;
  }
  const root = (artifactRoot || DEFAULT_ARTIFACT_ROOT).replace(/\/$/, '');
  return `${root}/${artifactDir}`;
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

function formatHiddenLayers(layers) {
  if (!Array.isArray(layers) || layers.length === 0) {
    return '—';
  }
  return layers.join(', ');
}

function annotateSample(sample) {
  const probabilities = Array.isArray(sample.probabilities)
    ? sample.probabilities.map((value) => Number.parseFloat(value))
    : [];
  const entries = probabilities
    .map((value, label) => ({ label, value }))
    .sort((a, b) => b.value - a.value);
  const topAlternate = entries.find((entry) => entry.label !== sample.predicted_label);
  const confidencePct = (sample.confidence * 100).toFixed(1);
  const baseDescription =
    sample.predicted_label === sample.true_label
      ? `Prediction matches the label with ${confidencePct}% confidence.`
      : `Prediction differs from the label with ${confidencePct}% confidence.`;
  const context = topAlternate
    ? `Next strongest class: ${topAlternate.label} (${(topAlternate.value * 100).toFixed(1)}%).`
    : 'No alternate class scores available.';

  return {
    id: sample.id,
    label: sample.true_label,
    predictedLabel: sample.predicted_label,
    confidence: sample.confidence,
    description: baseDescription,
    context,
    pixels: sample.pixels,
    probabilities,
  };
}

function deriveObservations(metrics, confusionMatrix, samples, summary) {
  const accuracy = typeof metrics.accuracy === 'number' ? metrics.accuracy : 0;
  const macroF1 =
    typeof metrics.macro_f1 === 'number'
      ? metrics.macro_f1
      : typeof metrics.macroF1 === 'number'
      ? metrics.macroF1
      : 0;
  const observations = [];
  observations.push(
    `Hold-out accuracy reached ${(accuracy * 100).toFixed(2)}% with macro F1 ${(macroF1 * 100).toFixed(2)}%.`
  );

  let maxConfusion = 0;
  let confusionPair = null;
  (confusionMatrix || []).forEach((row, trueLabel) => {
    row.forEach((value, predictedLabel) => {
      if (trueLabel !== predictedLabel && value > maxConfusion) {
        maxConfusion = value;
        confusionPair = { trueLabel, predictedLabel, value };
      }
    });
  });

  if (confusionPair && confusionPair.value > 0) {
    observations.push(
      `Largest confusion: ${confusionPair.trueLabel} → ${confusionPair.predictedLabel} (${confusionPair.value} samples).`
    );
  } else {
    observations.push('Confusion matrix is perfectly diagonal for this run.');
  }

  const misclassified = samples.filter((sample) => sample.true_label !== sample.predicted_label).length;
  const total = samples.length;
  if (total > 0) {
    const rate = ((misclassified / total) * 100).toFixed(1);
    observations.push(
      `${misclassified} of ${total} dashboard samples are misclassified (${rate}%); drill into them via the inspector.`
    );
  }

  if (typeof summary.train_accuracy === 'number' && typeof summary.test_accuracy === 'number') {
    const gap = (summary.train_accuracy - summary.test_accuracy) * 100;
    observations.push(
      `Train/test gap: ${gap >= 0 ? '+' : ''}${gap.toFixed(2)} percentage points.`
    );
  }

  return observations;
}

function buildConfiguration(summary) {
  const hiddenLayers = formatHiddenLayers(summary.hidden_layer_sizes);
  return [
    { label: 'Scaler', value: 'StandardScaler' },
    { label: 'Hidden layers', value: hiddenLayers },
    { label: 'Activation', value: summary.activation || '—' },
    { label: 'Solver', value: summary.solver || '—' },
    {
      label: 'Learning rate',
      value: summary.learning_rate_init ? summary.learning_rate_init.toString() : 'adaptive',
    },
    {
      label: 'Train / test split',
      value: `${summary.dataset.train_samples} / ${summary.dataset.test_samples}`,
    },
    { label: 'Random state', value: String(summary.random_state) },
  ];
}

function assembleDashboardData(basePath, summary, metrics, samplePayload, curvePayload) {
  const hiddenLayers = formatHiddenLayers(summary.hidden_layer_sizes);
  const parameters = [
    summary.activation && `activation=${summary.activation}`,
    summary.solver && `solver=${summary.solver}`,
    summary.alpha !== undefined && `alpha=${summary.alpha}`,
    summary.max_iter !== undefined && `max_iter=${summary.max_iter}`,
  ]
    .filter(Boolean)
    .join(' · ');

  const loss = Array.isArray(curvePayload.loss) ? curvePayload.loss : [];
  const samples = Array.isArray(samplePayload.samples) ? samplePayload.samples : [];
  const annotatedSamples = samples
    .map((sample) => ({ order: sample.order ?? 0, value: annotateSample(sample) }))
    .sort((a, b) => a.order - b.order)
    .map((entry) => entry.value);

  return {
    artifactBasePath: basePath,
    runId: summary.run_id || summary.artifact_dir || basePath,
    runName: summary.run_name || summary.run_id || summary.artifact_dir || basePath,
    artifactDir: summary.artifact_dir || basePath.split('/').pop(),
    sampleDigits: annotatedSamples,
    trainingSummary: {
      model: `${summary.model} (${hiddenLayers})`,
      parameters: parameters || 'See run_summary.json for full configuration.',
      metrics: {
        accuracy: metrics.accuracy,
        macroF1: metrics.macro_f1,
        trainTimeSeconds: summary.train_time_seconds,
      },
      lossCurve: loss,
      lastRunAt: summary.generated_at,
      iterations: curvePayload.n_iterations,
      converged: curvePayload.converged,
    },
    dataset: summary.dataset || null,
    sampleGalleryImage: samplePayload.gallery_image || null,
    confusionMatrix: metrics.confusion_matrix,
    observations: deriveObservations(metrics, metrics.confusion_matrix, samples, summary),
    configuration: buildConfiguration(summary),
    timeline: timelineTemplate,
  };
}

async function loadDashboardData(basePath) {
  const [summary, metrics, samples, curve] = await Promise.all([
    fetchJson(`${basePath}/run_summary.json`),
    fetchJson(`${basePath}/metrics.json`),
    fetchJson(`${basePath}/predictions.json`),
    fetchJson(`${basePath}/loss_curve.json`),
  ]);

  return assembleDashboardData(basePath, summary, metrics, samples, curve);
}

async function loadRunIndex(rootPath) {
  try {
    const index = await fetchJson(`${rootPath.replace(/\/$/, '')}/${RUN_INDEX_FILE}`);
    if (!index || !Array.isArray(index.runs)) {
      return null;
    }
    return index;
  } catch (error) {
    console.warn('Run index unavailable', error);
    return null;
  }
}

function findRunEntryById(runId) {
  if (!runId || !runIndexData || !Array.isArray(runIndexData.runs)) {
    return null;
  }
  return runIndexData.runs.find((run) => run.run_id === runId) || null;
}

function findRunEntryByPath(path) {
  if (!path || !runIndexData || !Array.isArray(runIndexData.runs)) {
    return null;
  }
  const normalized = path.replace(/\/$/, '');
  const parts = normalized.split('/');
  const last = parts[parts.length - 1];
  return (
    runIndexData.runs.find(
      (run) => run.artifact_dir === last || run.run_id === last || run.run_name === last
    ) || null
  );
}

async function switchRun(descriptor) {
  const targetPath = descriptor?.path || determineArtifactBasePath();
  const runId = descriptor?.runId;

  try {
    const data = await loadDashboardData(targetPath);
    dashboardData = data;
    activeRunPath = targetPath;
    activeRunId = runId || data.runId || null;
    viewDigits = [...dashboardData.sampleDigits];
    inspectorDescriptionEl.textContent =
      'Choose a digit to inspect per-pixel activations and contextual notes captured during evaluation.';
    renderDashboard();
    cycleHeroDigit();
    setupEventListeners();
  } catch (error) {
    renderErrorState(targetPath, error);
  }
}

const root = document.documentElement;
const heroModelEl = document.getElementById('hero-model');
const heroRunDateEl = document.getElementById('hero-run-date');
const heroDigitCanvas = document.getElementById('hero-digit');
const digitGridEl = document.getElementById('digit-grid');
const filterEl = document.getElementById('digit-filter');
const misToggleEl = document.getElementById('misclassified-toggle');
const randomButton = document.getElementById('random-button');
const inspectorTitleEl = document.getElementById('inspector-title');
const inspectorCanvas = document.getElementById('inspector-canvas');
const inspectorLabelEl = document.getElementById('inspector-label');
const inspectorPredictionEl = document.getElementById('inspector-prediction');
const inspectorConfidenceEl = document.getElementById('inspector-confidence');
const inspectorDescriptionEl = document.getElementById('inspector-description');
const downloadButton = document.getElementById('download-button');
const modelNameEl = document.getElementById('model-name');
const modelParamsEl = document.getElementById('model-params');
const accuracyEl = document.getElementById('metric-accuracy');
const f1El = document.getElementById('metric-f1');
const trainTimeEl = document.getElementById('metric-train-time');
const lossCanvas = document.getElementById('loss-chart');
const lossRefreshButton = document.getElementById('loss-refresh');
const matrixEl = document.getElementById('confusion-matrix');
const observationsEl = document.getElementById('insight-observations');
const configListEl = document.getElementById('config-list');
const runTableBody = document.getElementById('run-table-body');
const timelineEl = document.getElementById('timeline');
const toggleThemeButton = document.getElementById('toggle-theme');
const compareButton = document.getElementById('compare-button');
const runSelectorEl = document.getElementById('run-selector');

let heroIntervalId;
let lossAnimationFrame;
let currentDigitId = null;
let viewDigits = [];
const cardRefs = new Map();
let listenersRegistered = false;

function syncThemeToggle() {
  const dark = root.classList.contains('dark');
  toggleThemeButton.setAttribute('aria-pressed', String(dark));
  const label = toggleThemeButton.querySelector('.btn__label');
  if (label) {
    label.textContent = dark ? 'Light' : 'Dark';
  }
}

function formatPercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return '—';
  }
  return `${(value * 100).toFixed(1)}%`;
}

function formatSeconds(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return '—';
  }
  return `${value.toFixed(1)} s`;
}

function formatDate(isoString) {
  if (!isoString) {
    return '—';
  }
  const date = new Date(isoString);
  return date.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  });
}

function getColorForValue(value) {
  const maxValue = 16;
  const intensity = value / maxValue;
  const hue = 220 - intensity * 120;
  const lightness = 15 + intensity * 70;
  return `hsl(${hue}, 80%, ${lightness}%)`;
}

function drawDigit(canvas, pixels) {
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext('2d');
  const source = Array.isArray(pixels) ? pixels : [];
  const size = Math.sqrt(source.length || 64);
  const scale = canvas.width / size;
  const radius = scale * 0.35;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const value = source[y * size + x] || 0;
      const color = getColorForValue(value);
      const centerX = x * scale + scale / 2;
      const centerY = y * scale + scale / 2;

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function downloadCanvas(canvas, filename) {
  const link = document.createElement('a');
  link.href = canvas.toDataURL('image/png');
  link.download = filename;
  link.click();
}

function shuffleDigits() {
  if (!dashboardData) {
    return;
  }
  viewDigits = [...dashboardData.sampleDigits].sort(() => Math.random() - 0.5);
  renderDigits();
}

function populateFilterOptions() {
  filterEl.innerHTML = '';
  const allOption = document.createElement('option');
  allOption.value = 'all';
  allOption.textContent = 'All digits';
  filterEl.appendChild(allOption);

  if (!dashboardData) {
    filterEl.disabled = true;
    return;
  }

  filterEl.disabled = false;
  const labels = Array.from(new Set(dashboardData.sampleDigits.map((digit) => digit.label))).sort(
    (a, b) => a - b
  );
  labels.forEach((label) => {
    const option = document.createElement('option');
    option.value = label;
    option.textContent = `Digit ${label}`;
    filterEl.appendChild(option);
  });

  filterEl.value = 'all';
}

function updateInspector(digit) {
  if (!digit) {
    inspectorTitleEl.textContent = 'Select a sample';
    inspectorLabelEl.textContent = '—';
    inspectorPredictionEl.textContent = '—';
    inspectorConfidenceEl.textContent = '—';
    inspectorDescriptionEl.textContent =
      'Choose a digit to inspect per-pixel activations and contextual notes captured during evaluation.';
    downloadButton.disabled = true;
    drawDigit(inspectorCanvas, new Array(64).fill(0));
    currentDigitId = null;
    highlightCards();
    highlightMatrix();
    return;
  }

  inspectorTitleEl.textContent = `Sample ${digit.id.toUpperCase()}`;
  inspectorLabelEl.textContent = digit.label;
  inspectorPredictionEl.textContent = `${digit.predictedLabel} (${digit.predictedLabel === digit.label ? 'match' : 'mismatch'})`;
  inspectorConfidenceEl.textContent = `${Math.round(digit.confidence * 100)}%`;
  inspectorDescriptionEl.textContent = `${digit.description} ${digit.context}`;
  downloadButton.disabled = false;
  currentDigitId = digit.id;
  drawDigit(inspectorCanvas, digit.pixels);
  highlightCards(digit.id);
  highlightMatrix(digit.label, digit.predictedLabel);
}

function createDigitCard(digit) {
  const card = document.createElement('article');
  card.className = 'digit-card';
  card.setAttribute('data-digit-id', digit.id);
  card.setAttribute('data-true-label', digit.label);
  card.setAttribute('data-predicted-label', digit.predictedLabel);
  card.tabIndex = 0;

  const heading = document.createElement('header');
  heading.className = 'digit-card__header';
  heading.innerHTML = `<span class="digit-label">Label ${digit.label}</span><span class="digit-confidence">${Math.round(
    digit.confidence * 100
  )}%</span>`;

  const canvas = document.createElement('canvas');
  canvas.width = 160;
  canvas.height = 160;
  drawDigit(canvas, digit.pixels);

  const footer = document.createElement('footer');
  footer.className = 'digit-card__footer';
  footer.innerHTML = `
    <span class="prediction">Predicted ${digit.predictedLabel}</span>
    <span class="status ${digit.label === digit.predictedLabel ? 'status--ok' : 'status--warn'}">
      ${digit.label === digit.predictedLabel ? 'Correct' : 'Misclassified'}
    </span>
  `;

  const description = document.createElement('p');
  description.textContent = `${digit.description} ${digit.context}`.trim();

  card.appendChild(heading);
  card.appendChild(canvas);
  card.appendChild(footer);
  card.appendChild(description);

  card.addEventListener('click', () => updateInspector(digit));
  card.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      updateInspector(digit);
    }
  });

  cardRefs.set(digit.id, card);
  return card;
}

function highlightCards(activeId) {
  cardRefs.forEach((card, id) => {
    if (activeId && id === activeId) {
      card.classList.add('digit-card--active');
    } else {
      card.classList.remove('digit-card--active');
    }
  });
}

function populateRunSelector() {
  if (!runSelectorEl) {
    return;
  }

  runSelectorEl.innerHTML = '';

  const runs = runIndexData?.runs || [];
  if (!runs.length) {
    const option = document.createElement('option');
    option.value = activeRunPath || determineArtifactBasePath();
    option.textContent = 'Latest artifacts';
    runSelectorEl.appendChild(option);
    runSelectorEl.disabled = true;
    return;
  }

  runs.forEach((run) => {
    const option = document.createElement('option');
    option.value = run.run_id;
    const accuracy =
      typeof run.test_accuracy === 'number' ? ` · ${(run.test_accuracy * 100).toFixed(1)}% acc` : '';
    option.textContent = `${run.run_name || run.run_id}${accuracy}`;
    if (run.run_id === activeRunId) {
      option.selected = true;
    }
    runSelectorEl.appendChild(option);
  });

  runSelectorEl.disabled = runs.length <= 1;
}

function highlightMatrix(trueLabel, predictedLabel) {
  matrixEl.querySelectorAll('.matrix__cell').forEach((cell) => {
    const matches =
      Number(cell.dataset.trueLabel) === trueLabel && Number(cell.dataset.predictedLabel) === predictedLabel;
    cell.classList.toggle('matrix__cell--active', Boolean(matches));
  });
}

function renderDigits() {
  if (!dashboardData) {
    digitGridEl.innerHTML = '';
    const empty = document.createElement('p');
    empty.className = 'empty-state';
    empty.textContent = 'Run the Python demo to generate artifacts for this dashboard.';
    digitGridEl.appendChild(empty);
    updateInspector(null);
    misToggleEl.disabled = true;
    randomButton.disabled = true;
    return;
  }

  misToggleEl.disabled = false;
  randomButton.disabled = false;
  digitGridEl.innerHTML = '';
  cardRefs.clear();

  const labelFilter = filterEl.value;
  const onlyMisclassified = misToggleEl.checked;

  const filtered = viewDigits.filter((digit) => {
    const matchesLabel = labelFilter === 'all' || digit.label === Number.parseInt(labelFilter, 10);
    const matchesMisclassification = !onlyMisclassified || digit.label !== digit.predictedLabel;
    return matchesLabel && matchesMisclassification;
  });

  filtered.forEach((digit) => {
    const card = createDigitCard(digit);
    digitGridEl.appendChild(card);
  });

  if (filtered.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'empty-state';
    empty.textContent = 'No samples match the current filters. Try relaxing the criteria.';
    digitGridEl.appendChild(empty);
    updateInspector(null);
    return;
  }

  const selected = filtered.find((digit) => digit.id === currentDigitId) || filtered[0];
  updateInspector(selected);
}

function populateMetrics() {
  if (!dashboardData) {
    heroModelEl.textContent = '—';
    heroRunDateEl.textContent = '—';
    modelNameEl.textContent = 'Awaiting artifacts';
    modelParamsEl.textContent = 'Run the Python demo to populate metrics.';
    accuracyEl.textContent = '—';
    f1El.textContent = '—';
    trainTimeEl.textContent = '—';
    return;
  }

  const summary = dashboardData.trainingSummary;
  heroModelEl.textContent = summary.model;
  heroRunDateEl.textContent = formatDate(summary.lastRunAt);
  modelNameEl.textContent = summary.model;
  modelParamsEl.textContent = summary.parameters;
  accuracyEl.textContent = formatPercent(summary.metrics.accuracy);
  f1El.textContent = formatPercent(summary.metrics.macroF1);
  trainTimeEl.textContent = formatSeconds(summary.metrics.trainTimeSeconds);
}

function drawLossCurve({ animate = false } = {}) {
  const ctx = lossCanvas.getContext('2d');
  const padding = 40;
  const width = lossCanvas.width;
  const height = lossCanvas.height;
  const values = dashboardData ? dashboardData.trainingSummary.lossCurve : [];
  ctx.clearRect(0, 0, width, height);

  if (!values.length) {
    lossRefreshButton.disabled = true;
    ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--muted');
    ctx.font = '14px Inter, sans-serif';
    ctx.fillText('Loss curve will appear after running the demo.', padding, height / 2);
    return;
  }

  lossRefreshButton.disabled = false;
  const maxLoss = Math.max(...values);
  const minLoss = Math.min(...values);
  const xStep = (width - padding * 2) / (values.length - 1);

  ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--border');
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, height - padding);
  ctx.lineTo(width - padding, height - padding);
  ctx.stroke();

  let progress = animate ? 0 : values.length - 1;

  const drawSeries = () => {
    ctx.save();
    ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue('--accent');
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let index = 0; index <= progress; index += 1) {
      const loss = values[index];
      const x = padding + index * xStep;
      const normalized = (loss - minLoss) / (maxLoss - minLoss || 1);
      const y = height - padding - normalized * (height - padding * 2);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
    ctx.restore();

    if (progress < values.length - 1 && animate) {
      progress += 0.25;
      lossAnimationFrame = requestAnimationFrame(drawSeries);
    }
  };

  drawSeries();

  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue('--muted');
  ctx.font = '12px Inter, sans-serif';
  ctx.fillText('Epochs', width / 2 - 20, height - padding + 24);
  ctx.save();
  ctx.translate(padding - 24, height / 2 + 20);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Loss', 0, 0);
  ctx.restore();
}

function renderConfusionMatrix() {
  matrixEl.innerHTML = '';
  const matrix = dashboardData ? dashboardData.confusionMatrix : [];
  if (!matrix.length) {
    const message = document.createElement('p');
    message.className = 'empty-state';
    message.textContent = 'Confusion matrix will render after artifacts are generated.';
    matrixEl.appendChild(message);
    return;
  }

  const labels = Array.from({ length: matrix.length }, (_, index) => index);

  const headerRow = document.createElement('div');
  headerRow.className = 'matrix__row matrix__row--header';
  headerRow.appendChild(document.createElement('span'));
  labels.forEach((label) => {
    const cell = document.createElement('span');
    cell.textContent = label;
    cell.className = 'matrix__header';
    headerRow.appendChild(cell);
  });
  matrixEl.appendChild(headerRow);

  matrix.forEach((row, trueLabel) => {
    const rowEl = document.createElement('div');
    rowEl.className = 'matrix__row';

    const rowHeader = document.createElement('span');
    rowHeader.textContent = trueLabel;
    rowHeader.className = 'matrix__header';
    rowHeader.setAttribute('aria-hidden', 'true');
    rowEl.appendChild(rowHeader);

    row.forEach((value, predictedLabel) => {
      const cell = document.createElement('button');
      cell.type = 'button';
      cell.className = 'matrix__cell';
      cell.textContent = value;
      cell.dataset.trueLabel = String(trueLabel);
      cell.dataset.predictedLabel = String(predictedLabel);
      const isDiagonal = trueLabel === predictedLabel;
      if (isDiagonal) {
        cell.classList.add('matrix__cell--diagonal');
      }

      cell.addEventListener('mouseenter', () => {
        highlightMatrix(trueLabel, predictedLabel);
        highlightCardsByMatrix(trueLabel, predictedLabel);
      });
      cell.addEventListener('focus', () => {
        highlightMatrix(trueLabel, predictedLabel);
        highlightCardsByMatrix(trueLabel, predictedLabel);
      });
      cell.addEventListener('mouseleave', () => {
        highlightMatrix(
          cardRefs.get(currentDigitId)?.dataset.trueLabel
            ? Number(cardRefs.get(currentDigitId).dataset.trueLabel)
            : undefined,
          cardRefs.get(currentDigitId)?.dataset.predictedLabel
            ? Number(cardRefs.get(currentDigitId).dataset.predictedLabel)
            : undefined
        );
        highlightCards(currentDigitId || undefined);
      });
      cell.addEventListener('blur', () => {
        highlightMatrix();
        highlightCards(currentDigitId || undefined);
      });
      cell.addEventListener('click', () => {
        const matchingSample = dashboardData.sampleDigits.find(
          (digit) => digit.label === trueLabel && digit.predictedLabel === predictedLabel
        );
        if (matchingSample) {
          updateInspector(matchingSample);
        }
      });

      rowEl.appendChild(cell);
    });

    matrixEl.appendChild(rowEl);
  });
}

function highlightCardsByMatrix(trueLabel, predictedLabel) {
  cardRefs.forEach((card) => {
    const matches =
      Number(card.dataset.trueLabel) === trueLabel && Number(card.dataset.predictedLabel) === predictedLabel;
    card.classList.toggle('digit-card--matrix', Boolean(matches));
  });
}

function renderObservations() {
  observationsEl.innerHTML = '';
  const observations = dashboardData ? dashboardData.observations : [];
  if (!observations.length) {
    const item = document.createElement('li');
    item.textContent = 'Run the demo to populate quantitative observations.';
    observationsEl.appendChild(item);
    return;
  }

  observations.forEach((observation) => {
    const item = document.createElement('li');
    item.textContent = observation;
    observationsEl.appendChild(item);
  });
}

function renderConfiguration() {
  configListEl.innerHTML = '';
  const configuration = dashboardData ? dashboardData.configuration : [];
  if (!configuration.length) {
    const row = document.createElement('div');
    row.className = 'config-row';
    const term = document.createElement('dt');
    term.textContent = 'Status';
    const definition = document.createElement('dd');
    definition.textContent = 'Artifacts missing. Run the Python demo to see configuration details.';
    row.appendChild(term);
    row.appendChild(definition);
    configListEl.appendChild(row);
    return;
  }

  configuration.forEach((entry) => {
    const row = document.createElement('div');
    row.className = 'config-row';

    const term = document.createElement('dt');
    term.textContent = entry.label;

    const definition = document.createElement('dd');
    definition.textContent = entry.value;

    row.appendChild(term);
    row.appendChild(definition);
    configListEl.appendChild(row);
  });
}

function renderRuns() {
  runTableBody.innerHTML = '';
  const runs = runIndexData ? runIndexData.runs || [] : [];
  if (!runs.length) {
    const row = document.createElement('tr');
    row.innerHTML = '
      <td colspan="6">Run the demo to capture experiment history.</td>
    ';
    runTableBody.appendChild(row);
    compareButton.disabled = true;
    compareButton.textContent = 'Compare experiments';
    return;
  }

  runs.forEach((run) => {
    const row = document.createElement('tr');
    if (run.run_id === activeRunId) {
      row.classList.add('run-table__row--active');
    }
    row.innerHTML = `
      <th scope="row">${run.run_name || run.run_id}</th>
      <td>${formatHiddenLayers(run.hidden_layer_sizes || [])}</td>
      <td>${run.activation || '—'}</td>
      <td>${formatPercent(run.test_accuracy)}</td>
      <td>${formatPercent(run.macro_f1)}</td>
      <td>${run.run_id === activeRunId ? 'Active bundle' : 'Saved run'}</td>
    `;
    runTableBody.appendChild(row);
  });

  compareButton.disabled = runs.length < 2;
  compareButton.textContent = 'Compare experiments';
}

function renderTimeline() {
  timelineEl.innerHTML = '';
  const entries = dashboardData ? dashboardData.timeline : timelineTemplate;
  entries.forEach((entry) => {
    const item = document.createElement('li');
    item.className = 'timeline__item';
    item.innerHTML = `
      <h3>${entry.title}</h3>
      <p>${entry.description}</p>
      <p class="timeline__hint"><strong>Touchpoint:</strong> ${entry.file}</p>
      <p class="timeline__hint">${entry.hint}</p>
    `;
    timelineEl.appendChild(item);
  });
}

function cycleHeroDigit() {
  if (heroIntervalId) {
    clearInterval(heroIntervalId);
  }

  if (!dashboardData || !dashboardData.sampleDigits.length) {
    drawDigit(heroDigitCanvas, []);
    return;
  }

  let index = 0;
  const renderNext = () => {
    const digit = dashboardData.sampleDigits[index];
    drawDigit(heroDigitCanvas, digit.pixels);
    index = (index + 1) % dashboardData.sampleDigits.length;
  };

  renderNext();
  heroIntervalId = setInterval(renderNext, 3200);
}

function toggleTheme() {
  const isDark = root.classList.toggle('dark');
  root.classList.toggle('light', !isDark);
  syncThemeToggle();
  drawLossCurve();
  if (currentDigitId) {
    const digit = dashboardData.sampleDigits.find((sample) => sample.id === currentDigitId);
    if (digit) {
      drawDigit(inspectorCanvas, digit.pixels);
    }
  }
}

function setupEventListeners() {
  if (listenersRegistered) {
    return;
  }
  filterEl.addEventListener('change', () => {
    renderDigits();
  });

  misToggleEl.addEventListener('change', () => {
    renderDigits();
  });

  randomButton.addEventListener('click', () => {
    shuffleDigits();
  });

  downloadButton.addEventListener('click', () => {
    if (!dashboardData || !currentDigitId) {
      return;
    }
    const digit = dashboardData.sampleDigits.find((sample) => sample.id === currentDigitId);
    if (digit) {
      drawDigit(inspectorCanvas, digit.pixels);
      downloadCanvas(inspectorCanvas, `${digit.id}.png`);
    }
  });

  toggleThemeButton.addEventListener('click', () => {
    toggleTheme();
  });

  lossRefreshButton.addEventListener('click', () => {
    if (lossAnimationFrame) {
      cancelAnimationFrame(lossAnimationFrame);
    }
    drawLossCurve({ animate: true });
  });

  compareButton.addEventListener('click', () => {
    const runs = runIndexData ? runIndexData.runs || [] : [];
    if (runs.length < 2) {
      const toast = document.createElement('div');
      toast.className = 'toast';
      toast.textContent = 'Add another run to compare results.';
      document.body.appendChild(toast);
      setTimeout(() => {
        toast.classList.add('toast--visible');
      }, 16);
      setTimeout(() => {
        toast.classList.remove('toast--visible');
        setTimeout(() => {
          toast.remove();
        }, 300);
      }, 1600);
      return;
    }

    const sortable = runs.filter((run) => typeof run.test_accuracy === 'number');
    if (sortable.length < 2) {
      return;
    }
    const sorted = [...sortable].sort((a, b) => b.test_accuracy - a.test_accuracy);
    const best = sorted[0];
    const challenger = sorted[1];
    const delta = (best.test_accuracy - challenger.test_accuracy) * 100;
    const message = `Top run ${best.run_name || best.run_id} leads ${
      challenger.run_name || challenger.run_id
    } by ${delta.toFixed(2)} percentage points in accuracy.`;
    compareButton.textContent = 'Comparison noted';
    compareButton.disabled = true;
    compareButton.setAttribute('aria-live', 'polite');
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
      toast.classList.add('toast--visible');
    }, 16);
    setTimeout(() => {
    toast.classList.remove('toast--visible');
      setTimeout(() => {
        toast.remove();
      }, 300);
    }, 3200);
  });

  if (runSelectorEl) {
    runSelectorEl.addEventListener('change', (event) => {
      const selectedRunId = event.target.value;
      const entry = findRunEntryById(selectedRunId);
      if (!entry) {
        return;
      }
      const path = buildRunPath(entry.artifact_dir);
      if (path) {
        switchRun({ path, runId: entry.run_id });
      }
    });
  }

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      clearInterval(heroIntervalId);
    } else {
      cycleHeroDigit();
    }
  });

  listenersRegistered = true;
}

function applyTheme() {
  if (!root.classList.contains('dark')) {
    root.classList.add('dark');
  }
  root.classList.remove('light');
  syncThemeToggle();
}

function renderDashboard() {
  populateFilterOptions();
  populateMetrics();
  renderDigits();
  drawLossCurve();
  renderConfusionMatrix();
  renderObservations();
  renderConfiguration();
  populateRunSelector();
  renderRuns();
  renderTimeline();
}

function renderErrorState(basePath, error) {
  console.error('Failed to load dashboard artifacts', error);
  dashboardData = null;
  currentDigitId = null;
  viewDigits = [];
  inspectorDescriptionEl.textContent = `Artifacts not found at ${basePath}. Run the demo and refresh.`;
  downloadButton.disabled = true;
  compareButton.disabled = true;
  if (runSelectorEl) {
    runSelectorEl.disabled = true;
  }
  renderDashboard();
  cycleHeroDigit();
  setupEventListeners();
}

async function bootstrap() {
  applyTheme();
  const requestedPath = determineArtifactBasePath();
  artifactRoot = deriveArtifactRoot(requestedPath);
  runIndexData = await loadRunIndex(artifactRoot);

  let descriptor = { path: requestedPath, runId: null };
  if (runIndexData && Array.isArray(runIndexData.runs) && runIndexData.runs.length) {
    const matching = findRunEntryByPath(requestedPath);
    if (matching) {
      descriptor = { path: buildRunPath(matching.artifact_dir), runId: matching.run_id };
    } else {
      const latest =
        findRunEntryById(runIndexData.latest_run_id) ||
        (runIndexData.runs.length ? runIndexData.runs[0] : null);
      if (latest) {
        descriptor = { path: buildRunPath(latest.artifact_dir), runId: latest.run_id };
      }
    }
  }

  await switchRun(descriptor);
}

bootstrap();
