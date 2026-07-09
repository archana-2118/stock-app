/**
 * StockSense AI — Frontend
 *
 * Common causes of "Failed to fetch" and how they are fixed here:
 * 1. Wrong URL          → uses relative paths (/predict, /compare) — same origin, no CORS needed
 * 2. Missing CORS       → flask-cors added on the server (also handles external JS callers)
 * 3. Missing headers    → Content-Type: application/json set on every request
 * 4. Missing body       → JSON.stringify(payload) used for every POST
 * 5. Server not running → /health check helps surface this clearly
 * 6. Invalid ticker     → server validates and returns {error: "..."} with 4xx status
 */

"use strict";

// ─── Chart instances (kept for destroy/re-draw) ────────────────────────────
let chartHistory  = null;
let chartForecast = null;
let chartValidate = null;

// ─── Colour palette ────────────────────────────────────────────────────────
const CLR = {
  lstm:     "#6c63ff",
  lstmFill: "rgba(108,99,255,.15)",
  prophet:  "#00d9a6",
  propFill: "rgba(0,217,166,.15)",
  hist:     "#7a8ca8",
  histFill: "rgba(122,140,168,.12)",
  actual:   "#f59e0b",
  grid:     "rgba(255,255,255,.06)",
  tick:     "#7a8ca8",
};

// ─── Minimal Chart.js defaults ─────────────────────────────────────────────
Chart.defaults.color = CLR.tick;
Chart.defaults.borderColor = CLR.grid;
Chart.defaults.font.family = "'Inter','Segoe UI',system-ui,sans-serif";
Chart.defaults.font.size = 11;

// ─── Helpers ───────────────────────────────────────────────────────────────
function getTicker() {
  return document.getElementById("ticker").value.trim().toUpperCase();
}

function getDays() {
  return parseInt(document.getElementById("days").value, 10);
}

function setLoading(on, msg = "Training model and fetching data…") {
  document.getElementById("loading").classList.toggle("hidden", !on);
  document.getElementById("loading-msg").textContent = msg;
  document.getElementById("results").classList.add("hidden");
  document.getElementById("error-banner").classList.add("hidden");
  ["btn-lstm", "btn-prophet", "btn-compare"].forEach(id =>
    (document.getElementById(id).disabled = on)
  );
}

function showError(msg) {
  setLoading(false);
  const banner = document.getElementById("error-banner");
  document.getElementById("error-text").textContent = msg;
  banner.classList.remove("hidden");
}

/**
 * Core fetch wrapper.
 * - Always sends Content-Type: application/json
 * - Always sends a JSON-stringified body
 * - Returns parsed JSON or throws an Error with a human-readable message
 */
async function apiPost(endpoint, payload) {
  let res;
  try {
    res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (networkErr) {
    // fetch() itself rejected — server is unreachable or CORS preflight failed
    throw new Error(
      "Cannot reach the prediction server. " +
      "Common causes: (1) the Flask server is not running — check the workflow logs; " +
      "(2) a network/proxy error blocked the request. " +
      `Technical detail: ${networkErr.message}`
    );
  }

  // Parse JSON response regardless of status code
  let data;
  try {
    data = await res.json();
  } catch (_) {
    throw new Error(
      `Server returned a non-JSON response (HTTP ${res.status}). ` +
      "The Flask server may have crashed — check the terminal logs."
    );
  }

  if (!res.ok) {
    // Server sent a structured {error: "..."} body
    throw new Error(data.error || `HTTP ${res.status}: ${res.statusText}`);
  }
  return data;
}

// ─── Chart factory ─────────────────────────────────────────────────────────
function makeChart(canvasId, datasets, labels, title) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "top",
          labels: { boxWidth: 12, padding: 14, color: CLR.tick },
        },
        title: { display: false },
        tooltip: {
          callbacks: {
            label: ctx =>
              ` ${ctx.dataset.label}: $${ctx.parsed.y.toFixed(2)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { maxTicksLimit: 8, color: CLR.tick, maxRotation: 0 },
          grid: { color: CLR.grid },
        },
        y: {
          ticks: {
            color: CLR.tick,
            callback: v => "$" + v.toFixed(0),
          },
          grid: { color: CLR.grid },
        },
      },
    },
  });
}

function destroyCharts() {
  [chartHistory, chartForecast, chartValidate].forEach(c => c && c.destroy());
  chartHistory = chartForecast = chartValidate = null;
}

// ─── Render helpers ─────────────────────────────────────────────────────────
function renderHistoryChart(dates, prices) {
  chartHistory = makeChart(
    "chart-history",
    [{
      label: "Close",
      data: prices,
      borderColor: CLR.hist,
      backgroundColor: CLR.histFill,
      fill: true,
      tension: 0.35,
      pointRadius: 0,
      borderWidth: 1.8,
    }],
    dates,
    "Historical"
  );
}

function renderForecastChart(days, lstmPreds, propPreds) {
  const labels = Array.from({ length: days }, (_, i) => `Day ${i + 1}`);
  const datasets = [];
  if (lstmPreds)
    datasets.push({
      label: "LSTM",
      data: lstmPreds,
      borderColor: CLR.lstm,
      backgroundColor: CLR.lstmFill,
      fill: true,
      tension: 0.4,
      pointRadius: 3,
      borderWidth: 2,
    });
  if (propPreds)
    datasets.push({
      label: "Prophet",
      data: propPreds,
      borderColor: CLR.prophet,
      backgroundColor: CLR.propFill,
      fill: true,
      tension: 0.4,
      pointRadius: 3,
      borderWidth: 2,
    });

  document.getElementById("forecast-title").textContent =
    datasets.length === 2
      ? "Forecast Comparison (LSTM vs Prophet)"
      : `${datasets[0]?.label ?? ""} Forecast`;

  chartForecast = makeChart("chart-forecast", datasets, labels, "Forecast");
}

function renderValidationChart(actual, lstmPred, propPred) {
  const labels = actual.map((_, i) => `T-${actual.length - i}`);
  const datasets = [
    {
      label: "Actual",
      data: actual,
      borderColor: CLR.actual,
      backgroundColor: "transparent",
      tension: 0.3,
      pointRadius: 0,
      borderWidth: 2,
      borderDash: [4, 4],
    },
  ];
  if (lstmPred)
    datasets.push({
      label: "LSTM Predicted",
      data: lstmPred,
      borderColor: CLR.lstm,
      backgroundColor: "transparent",
      tension: 0.3,
      pointRadius: 0,
      borderWidth: 1.8,
    });
  if (propPred)
    datasets.push({
      label: "Prophet Predicted",
      data: propPred,
      borderColor: CLR.prophet,
      backgroundColor: "transparent",
      tension: 0.3,
      pointRadius: 0,
      borderWidth: 1.8,
    });

  chartValidate = makeChart("chart-validation", datasets, labels, "Validation");
}

function renderRmseCards(lstm_rmse, prophet_rmse) {
  const row = document.getElementById("rmse-row");
  row.innerHTML = "";
  if (lstm_rmse != null) {
    row.innerHTML += `
      <div class="rmse-card rmse-lstm">
        <div class="model-name">LSTM</div>
        <div class="rmse-value">$${lstm_rmse.toFixed(2)}</div>
        <div class="rmse-label">RMSE on test set</div>
      </div>`;
  }
  if (prophet_rmse != null) {
    row.innerHTML += `
      <div class="rmse-card rmse-prophet">
        <div class="model-name">Prophet</div>
        <div class="rmse-value">$${prophet_rmse.toFixed(2)}</div>
        <div class="rmse-label">RMSE on test set</div>
      </div>`;
  }
  if (lstm_rmse != null && prophet_rmse != null) {
    const better = lstm_rmse <= prophet_rmse ? "LSTM" : "Prophet";
    row.innerHTML += `
      <div class="rmse-card" style="border-color:rgba(245,158,11,.3)">
        <div class="model-name" style="color:#f59e0b">Winner</div>
        <div class="rmse-value" style="color:#f59e0b;font-size:1.1rem">${better}</div>
        <div class="rmse-label">Lower RMSE wins</div>
      </div>`;
  }
}

function renderPredTable(days, lstmPreds, propPreds) {
  const wrap = document.getElementById("pred-table-wrap");
  const hasBoth = lstmPreds && propPreds;

  let rows = "";
  const n = Math.max(lstmPreds?.length ?? 0, propPreds?.length ?? 0);
  for (let i = 0; i < n; i++) {
    const lstm = lstmPreds ? `$${lstmPreds[i].toFixed(2)}` : "—";
    const prop = propPreds ? `$${propPreds[i].toFixed(2)}` : "—";
    rows += `<tr><td>Day ${i + 1}</td>`;
    if (lstmPreds) rows += `<td>${lstm}</td>`;
    if (propPreds) rows += `<td>${prop}</td>`;
    if (hasBoth) {
      const diff = (((propPreds[i] - lstmPreds[i]) / lstmPreds[i]) * 100).toFixed(2);
      const sign = diff >= 0 ? "+" : "";
      rows += `<td style="color:${diff >= 0 ? "#22c55e" : "#ef4444"}">${sign}${diff}%</td>`;
    }
    rows += `</tr>`;
  }

  wrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Trading Day</th>
          ${lstmPreds ? "<th>LSTM Pred.</th>" : ""}
          ${propPreds ? "<th>Prophet Pred.</th>" : ""}
          ${hasBoth ? "<th>Difference</th>" : ""}
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function showResults(ticker, days, data) {
  // Ticker bar
  document.getElementById("result-ticker").textContent = ticker;
  const lastPrice = data.historical_prices?.at(-1);
  document.getElementById("result-last-price").textContent =
    lastPrice != null ? `$${lastPrice.toFixed(2)}` : "";
  document.getElementById("result-days").textContent =
    `Next ${days} day${days > 1 ? "s" : ""} forecast`;

  destroyCharts();

  renderHistoryChart(data.historical_dates, data.historical_prices);
  renderForecastChart(days, data.lstm_predictions ?? null, data.prophet_predictions ?? null);
  renderRmseCards(data.lstm_rmse ?? null, data.prophet_rmse ?? null);

  // Validation chart: use whichever model ran
  const actual = data.test_actual;
  const lstmTestPred = data.test_predicted && data.lstm_predictions ? data.test_predicted : null;
  const propTestPred = data.test_predicted && data.prophet_predictions ? data.test_predicted : null;
  if (actual) {
    document.getElementById("validation-section").classList.remove("hidden");
    renderValidationChart(actual, lstmTestPred, propTestPred);
  } else {
    document.getElementById("validation-section").classList.add("hidden");
  }

  renderPredTable(days, data.lstm_predictions ?? null, data.prophet_predictions ?? null);

  setLoading(false);
  document.getElementById("error-banner").classList.add("hidden");
  document.getElementById("results").classList.remove("hidden");
}

// ─── Public action handlers (called from HTML onclick) ─────────────────────
async function predictSingle(model) {
  const ticker = getTicker();
  if (!ticker) { showError("Please enter a stock ticker symbol (e.g. AAPL)."); return; }
  const days = getDays();
  const label = model === "lstm" ? "LSTM" : "Prophet";
  setLoading(true, `Fetching data for ${ticker} and training ${label} model…`);
  try {
    const data = await apiPost("/predict", { ticker, model, days });
    showResults(ticker, days, data);
  } catch (err) {
    showError(err.message);
  }
}

async function predictCompare() {
  const ticker = getTicker();
  if (!ticker) { showError("Please enter a stock ticker symbol (e.g. AAPL)."); return; }
  const days = getDays();
  setLoading(true, `Running both LSTM & Prophet for ${ticker}… (this takes ~1 min)`);
  try {
    const data = await apiPost("/compare", { ticker, days });
    if (data.lstm_error && data.prophet_error) {
      throw new Error(`Both models failed.\nLSTM: ${data.lstm_error}\nProphet: ${data.prophet_error}`);
    }
    showResults(ticker, days, data);
  } catch (err) {
    showError(err.message);
  }
}

// ─── Keyboard shortcut: Enter inside ticker input ─────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("ticker").addEventListener("keydown", e => {
    if (e.key === "Enter") predictSingle("lstm");
  });
});
