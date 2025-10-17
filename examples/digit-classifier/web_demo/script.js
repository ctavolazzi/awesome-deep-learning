const ARTIFACT_ROOT = "./artifacts";
const METRICS_FILE = "metrics.json";
const TRAINING_DYNAMICS_FILE = "training_dynamics.json";
const PREDICTIONS_FILE = "predictions.json";
const CHART_JS_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js";

const PREDICTION_ROWS_PER_PAGE = 25;
const MAX_PREDICTION_ROWS = 500;

const state = {
  predictions: [],
  currentPage: 0,
  rowsPerPage: PREDICTION_ROWS_PER_PAGE,
  truncated: false,
  chart: null,
};

function byId(id) {
  return document.getElementById(id);
}

function getSection(id) {
  return byId(id) || document.querySelector(`[data-section="${id}"]`);
}

function ensureStatusElement(section) {
  if (!section) return null;
  const existing = section.querySelector(".status-message");
  if (existing) return existing;
  const el = document.createElement("div");
  el.className = "status-message";
  el.setAttribute("role", "status");
  el.style.marginBottom = "0.75rem";
  section.prepend(el);
  return el;
}

function setStatus(section, type, message) {
  const status = ensureStatusElement(section);
  if (!status) return;
  status.dataset.status = type;
  status.textContent = message;
  status.hidden = !message;
}

async function fetchJsonArtifact(filename) {
  const url = `${ARTIFACT_ROOT}/${filename}`;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to fetch ${filename}: ${response.status} ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    throw new Error(`Unable to load ${filename}: ${error.message}`);
  }
}

function formatMetricValue(value) {
  if (typeof value === "number") {
    if (Number.isInteger(value)) {
      return value.toString();
    }
    return value.toFixed(4);
  }
  return String(value);
}

function renderMetrics(metrics) {
  const section = getSection("metrics-section");
  if (!section) return;
  setStatus(section, "", "");

  const container = section.querySelector(".metrics-grid") || section;
  const summaryList = container.querySelector("[data-role='metrics-summary']");
  const reportTable = container.querySelector("[data-role='classification-report'] table") || container.querySelector("#classification-report");

  if (summaryList) {
    summaryList.innerHTML = "";
    const entries = [
      ["Train accuracy", metrics.train_accuracy],
      ["Test accuracy", metrics.test_accuracy],
    ];
    for (const [label, value] of entries) {
      const li = document.createElement("li");
      li.innerHTML = `<strong>${label}:</strong> ${formatMetricValue(value)}`;
      summaryList.appendChild(li);
    }
  }

  if (reportTable && metrics.classification_report) {
    const tbody = reportTable.tBodies?.[0] || reportTable.querySelector("tbody") || reportTable;
    while (tbody.firstChild) {
      tbody.removeChild(tbody.firstChild);
    }
    const rows = Object.entries(metrics.classification_report)
      .filter(([label]) => !["accuracy", "macro avg", "weighted avg"].includes(label));
    for (const [label, stats] of rows) {
      const tr = document.createElement("tr");
      const cells = [label, stats.precision, stats.recall, stats.f1_score, stats.support];
      for (const cell of cells) {
        const td = document.createElement("td");
        td.textContent = formatMetricValue(cell ?? "—");
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
  }
}

function derivePredictionColumns(records) {
  const columnSet = new Set();
  for (const record of records) {
    if (record && typeof record === "object" && !Array.isArray(record)) {
      Object.keys(record).forEach((key) => columnSet.add(key));
    }
  }
  return Array.from(columnSet);
}

function getPredictionTableElements() {
  const section = getSection("predictions-section") || getSection("predictions");
  const table = byId("predictions-table");
  const tbody = table?.tBodies?.[0] || table?.querySelector("tbody") || table;
  const controls = byId("predictions-pagination") || section?.querySelector(".pagination-controls");
  const footer = section?.querySelector(".predictions-footer") || controls;
  return { section, table, tbody, controls, footer };
}

function updatePaginationControls(totalPages) {
  const { controls } = getPredictionTableElements();
  if (!controls) return;
  controls.innerHTML = "";

  const prevBtn = document.createElement("button");
  prevBtn.type = "button";
  prevBtn.textContent = "Previous";
  prevBtn.disabled = state.currentPage === 0;
  prevBtn.addEventListener("click", () => {
    if (state.currentPage > 0) {
      state.currentPage -= 1;
      renderPredictionTablePage();
    }
  });

  const nextBtn = document.createElement("button");
  nextBtn.type = "button";
  nextBtn.textContent = "Next";
  nextBtn.disabled = state.currentPage >= totalPages - 1;
  nextBtn.addEventListener("click", () => {
    if (state.currentPage < totalPages - 1) {
      state.currentPage += 1;
      renderPredictionTablePage();
    }
  });

  const pageInfo = document.createElement("span");
  pageInfo.textContent = `Page ${totalPages === 0 ? 0 : state.currentPage + 1} of ${totalPages}`;
  pageInfo.style.margin = "0 0.75rem";

  controls.appendChild(prevBtn);
  controls.appendChild(pageInfo);
  controls.appendChild(nextBtn);
}

function renderPredictionTablePage() {
  const { tbody, footer } = getPredictionTableElements();
  if (!tbody) return;

  while (tbody.firstChild) {
    tbody.removeChild(tbody.firstChild);
  }

  const start = state.currentPage * state.rowsPerPage;
  const end = start + state.rowsPerPage;
  const pageRecords = state.predictions.slice(start, end);
  const columns = derivePredictionColumns(state.predictions);

  if (Array.isArray(state.predictions) && state.predictions.length && !columns.length && Array.isArray(state.predictions[0])) {
    // Handle array-of-arrays by mapping to column indices.
    for (let idx = 0; idx < state.predictions[0].length; idx += 1) {
      columns.push(`col_${idx}`);
    }
  }

  if (!columns.length && pageRecords.length) {
    const headerRow = tbody.parentElement?.tHead?.rows?.[0];
    if (headerRow) {
      for (const cell of Array.from(headerRow.cells)) {
        columns.push(cell.dataset.key || cell.textContent || `col_${columns.length}`);
      }
    }
  }

  if (pageRecords.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = Math.max(columns.length, 1);
    td.textContent = "No predictions available for this page.";
    tr.appendChild(td);
    tbody.appendChild(tr);
  } else {
    for (const record of pageRecords) {
      const tr = document.createElement("tr");
      if (record && typeof record === "object" && !Array.isArray(record)) {
        for (const column of columns) {
          const td = document.createElement("td");
          td.textContent = column in record ? formatMetricValue(record[column]) : "—";
          tr.appendChild(td);
        }
      } else if (Array.isArray(record)) {
        for (const value of record) {
          const td = document.createElement("td");
          td.textContent = formatMetricValue(value);
          tr.appendChild(td);
        }
      } else {
        const td = document.createElement("td");
        td.colSpan = columns.length || 1;
        td.textContent = formatMetricValue(record);
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
  }

  if (footer) {
    const note = footer.querySelector(".truncation-note");
    if (note) {
      note.remove();
    }

    if (state.truncated) {
      const truncationNote = document.createElement("p");
      truncationNote.className = "truncation-note";
      truncationNote.style.marginTop = "0.75rem";
      truncationNote.textContent = `Showing the first ${MAX_PREDICTION_ROWS.toLocaleString()} predictions. Download the artifact for the complete list.`;
      footer.appendChild(truncationNote);
    }
  }

  const totalPages = Math.ceil(state.predictions.length / state.rowsPerPage);
  updatePaginationControls(totalPages);
}

async function loadPredictions() {
  const { section } = getPredictionTableElements();
  if (!section) return;
  setStatus(section, "loading", "Loading predictions…");
  try {
    const payload = await fetchJsonArtifact(PREDICTIONS_FILE);
    const records = Array.isArray(payload?.predictions) ? payload.predictions : Array.isArray(payload) ? payload : payload?.rows;
    if (!records || !Array.isArray(records) || records.length === 0) {
      setStatus(section, "empty", "No predictions artifact is available for this run.");
      return;
    }

    state.predictions = records.slice(0, MAX_PREDICTION_ROWS);
    state.truncated = records.length > MAX_PREDICTION_ROWS;
    state.currentPage = 0;
    setStatus(section, "", "");
    renderPredictionTablePage();
  } catch (error) {
    console.error(error);
    setStatus(section, "error", "Predictions artifact could not be loaded.");
  }
}

function ensureCanvas(section, existingCanvas) {
  if (existingCanvas instanceof HTMLCanvasElement) {
    return existingCanvas;
  }
  const canvas = document.createElement("canvas");
  if (section) {
    section.appendChild(canvas);
  }
  return canvas;
}

function destroyExistingChart() {
  if (state.chart && typeof state.chart.destroy === "function") {
    state.chart.destroy();
  }
  state.chart = null;
}

function loadChartJs() {
  if (window.Chart) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = CHART_JS_URL;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("Failed to load Chart.js"));
    document.head.appendChild(script);
  });
}

async function renderLossChart() {
  const section = getSection("loss-chart-section") || getSection("loss-chart") || byId("loss-chart-section");
  if (!section) return;
  setStatus(section, "loading", "Loading training dynamics…");

  let dynamics;
  try {
    dynamics = await fetchJsonArtifact(TRAINING_DYNAMICS_FILE);
  } catch (error) {
    console.warn(error);
    setStatus(section, "error", "Training dynamics artifact is missing.");
    destroyExistingChart();
    return;
  }

  const epochs = Array.isArray(dynamics?.epochs) ? dynamics.epochs : null;
  const losses = Array.isArray(dynamics?.losses) ? dynamics.losses : null;
  const learningRates = Array.isArray(dynamics?.learning_rates) ? dynamics.learning_rates : null;

  if (!epochs || !losses || epochs.length === 0 || losses.length === 0) {
    setStatus(section, "empty", "Loss curve data is unavailable for this run.");
    destroyExistingChart();
    return;
  }

  try {
    await loadChartJs();
  } catch (error) {
    console.error(error);
    setStatus(section, "error", "Unable to load charting library.");
    return;
  }

  const canvas = ensureCanvas(section, byId("loss-chart"));
  const ctx = canvas.getContext("2d");
  destroyExistingChart();

  const datasets = [
    {
      label: "Loss",
      data: losses.map((value, index) => ({ x: epochs[index], y: value })),
      borderColor: "#f97316",
      backgroundColor: "rgba(249, 115, 22, 0.15)",
      tension: 0.2,
      yAxisID: "loss",
    },
  ];

  if (learningRates && learningRates.length === epochs.length) {
    datasets.push({
      label: "Learning rate",
      data: learningRates.map((value, index) => ({ x: epochs[index], y: value })),
      borderColor: "#2563eb",
      backgroundColor: "rgba(37, 99, 235, 0.15)",
      borderDash: [6, 4],
      tension: 0.2,
      yAxisID: "lr",
    });
  }

  state.chart = new Chart(ctx, {
    type: "line",
    data: {
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: "index",
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Epoch" },
          ticks: { precision: 0 },
        },
        loss: {
          type: "linear",
          position: "left",
          title: { display: true, text: "Loss" },
        },
        lr: {
          type: "linear",
          position: "right",
          title: { display: true, text: "Learning rate" },
          grid: { drawOnChartArea: false },
        },
      },
      plugins: {
        legend: {
          position: "top",
        },
        tooltip: {
          callbacks: {
            label(context) {
              const value = context.parsed.y;
              return `${context.dataset.label}: ${typeof value === "number" ? value.toPrecision(4) : value}`;
            },
          },
        },
      },
    },
  });

  setStatus(section, "", "");
}

async function renderMetricsSection() {
  const section = getSection("metrics-section");
  if (!section) return;
  setStatus(section, "loading", "Loading metrics…");
  try {
    const metrics = await fetchJsonArtifact(METRICS_FILE);
    renderMetrics(metrics);
    setStatus(section, "", "");
  } catch (error) {
    console.warn(error);
    setStatus(section, "error", "Metrics artifact is unavailable.");
  }
}

async function init() {
  await Promise.all([renderMetricsSection(), loadPredictions(), renderLossChart()]);
}

document.addEventListener("DOMContentLoaded", init);
