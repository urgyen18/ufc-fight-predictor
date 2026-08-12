from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>UFC Fight Predictor</title>
  <style>
    body {
      margin: 0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding: 40px 20px 80px;
      box-sizing: border-box;
    }

    h1 {
      margin: 0;
      font-size: 2rem;
      letter-spacing: 1px;
    }

    .subtitle {
      margin: 8px 0 40px;
      font-size: 0.95rem;
      color: #777;
      text-align: center;
    }

    .matchup {
      display: flex;
      align-items: flex-start;
      justify-content: center;
      gap: 60px;
      flex-wrap: wrap;
    }

    .fighter-col {
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 220px;
    }

    .fighter-col img {
      width: 200px;
      height: 200px;
      object-fit: contain;
    }

    .fighter-title {
      margin-top: 8px;
      font-weight: 700;
      font-size: 1.1rem;
      letter-spacing: 0.5px;
    }

    .fighter-col.red .fighter-title { color: #d32f2f; }
    .fighter-col.blue .fighter-title { color: #1565c0; }

    .fighter-col input {
      margin-top: 14px;
      width: 100%;
      box-sizing: border-box;
      padding: 10px 12px;
      font-size: 1rem;
      border-radius: 8px;
      border: 2px solid #ccc;
      text-align: center;
      outline: none;
    }

    .fighter-col.red input:focus { border-color: #d32f2f; }
    .fighter-col.blue input:focus { border-color: #1565c0; }

    #predict-btn {
      margin-top: 40px;
      padding: 16px 56px;
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: 1px;
      color: #fff;
      background: #111;
      border: none;
      border-radius: 10px;
      cursor: pointer;
      min-width: 220px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
    }

    #predict-btn:disabled {
      cursor: default;
      opacity: 0.85;
    }

    #predict-btn:not(:disabled):hover {
      background: #333;
    }

    .spinner {
      width: 20px;
      height: 20px;
      border: 3px solid rgba(255, 255, 255, 0.35);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
    }

    @keyframes spin {
      to { transform: rotate(360deg); }
    }

    #result {
      margin-top: 40px;
      width: 100%;
      max-width: 420px;
      display: none;
    }

    #result.visible { display: block; }

    .prob-line {
      text-align: center;
      font-size: 1rem;
      color: #444;
    }

    .prob-value {
      text-align: center;
      font-size: 3rem;
      font-weight: 800;
      margin: 4px 0 24px;
    }

    .tape {
      border-top: 1px solid #e0e0e0;
    }

    .tape-row {
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      gap: 12px;
      padding: 12px 4px;
      border-bottom: 1px solid #e0e0e0;
    }

    .tape-value {
      font-size: 1.05rem;
      font-weight: 600;
      color: #bbb;
    }

    .tape-value.red { text-align: right; }
    .tape-value.blue { text-align: left; }

    .tape-value.leader.red { color: #d32f2f; }
    .tape-value.leader.blue { color: #1565c0; }

    .tape-label {
      text-align: center;
      font-size: 0.85rem;
      color: #666;
      white-space: nowrap;
    }

    .error {
      margin-top: 40px;
      color: #d32f2f;
      text-align: center;
    }
  </style>
</head>
<body>
  <h1>UFC Fight Predictor</h1>
  <p class="subtitle">Fight outcome predictions powered by Elo ratings and a machine learning model trained on real UFC data.</p>

  <div class="matchup">
    <div class="fighter-col red">
      <img src="/static/redFighter.jpg" alt="Red fighter" />
      <div class="fighter-title">Red Fighter</div>
      <input id="red" list="fighters" placeholder="Start typing..." autocomplete="off" />
    </div>

    <div class="fighter-col blue">
      <img src="/static/blueFighter.png" alt="Blue fighter" />
      <div class="fighter-title">Blue Fighter</div>
      <input id="blue" list="fighters" placeholder="Start typing..." autocomplete="off" />
    </div>
  </div>

  <datalist id="fighters"></datalist>

  <button id="predict-btn" onclick="predict()">
    <span id="predict-label">PREDICT</span>
  </button>

  <div id="result"></div>
  <div id="error" class="error"></div>

<script>
async function updateSuggestions(query) {
  const q = (query || "").trim();
  const res = await fetch(`/fighters?q=${encodeURIComponent(q)}&limit=20`);
  const data = await res.json();

  const dl = document.getElementById("fighters");
  dl.innerHTML = "";
  (data.fighters || []).forEach(name => {
    const opt = document.createElement("option");
    opt.value = name;
    dl.appendChild(opt);
  });
}

["red", "blue"].forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener("input", (e) => updateSuggestions(e.target.value));
  el.addEventListener("focus", (e) => updateSuggestions(e.target.value));
});

const STAT_ROWS = [
  { key: "elo", label: "Elo Rating", decimals: 0, unit: "" },
  { key: "reach_cm", label: "Reach", decimals: 1, unit: " cm" },
  { key: "height_cm", label: "Height", decimals: 1, unit: " cm" },
  { key: "weight_lbs", label: "Weight", decimals: 0, unit: " lbs" },
];

function formatStatValue(value, decimals, unit) {
  if (typeof value !== "number") return "—";
  return value.toFixed(decimals) + unit;
}

function buildTapeRow(row, stats) {
  const pair = stats && stats[row.key];
  const red = pair ? pair.red : null;
  const blue = pair ? pair.blue : null;

  let redLeads = false;
  let blueLeads = false;
  if (typeof red === "number" && typeof blue === "number" && red !== blue) {
    redLeads = red > blue;
    blueLeads = blue > red;
  }

  return `
    <div class="tape-row">
      <div class="tape-value red ${redLeads ? "leader" : ""}">${formatStatValue(red, row.decimals, row.unit)}</div>
      <div class="tape-label">${row.label}</div>
      <div class="tape-value blue ${blueLeads ? "leader" : ""}">${formatStatValue(blue, row.decimals, row.unit)}</div>
    </div>
  `;
}

function setLoading(isLoading) {
  const btn = document.getElementById("predict-btn");
  const label = document.getElementById("predict-label");
  btn.disabled = isLoading;
  label.innerHTML = isLoading ? '<span class="spinner"></span>' : "PREDICT";
}

async function predict() {
  const red = document.getElementById("red").value.trim();
  const blue = document.getElementById("blue").value.trim();

  const resultEl = document.getElementById("result");
  const errorEl = document.getElementById("error");
  resultEl.classList.remove("visible");
  resultEl.innerHTML = "";
  errorEl.textContent = "";

  setLoading(true);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ red, blue })
    });

    const data = await res.json();

    if (!res.ok) {
      errorEl.textContent = data.detail || "Something went wrong.";
      return;
    }

    const pct = (data.prob_red_wins * 100).toFixed(2);
    const tapeRows = STAT_ROWS.map(row => buildTapeRow(row, data.stats)).join("");

    resultEl.innerHTML = `
      <div class="prob-line">Red fighter win probability</div>
      <div class="prob-value">${pct}%</div>
      <div class="tape">${tapeRows}</div>
    `;
    resultEl.classList.add("visible");
  } catch (err) {
    errorEl.textContent = "Network error. Please try again.";
  } finally {
    setLoading(false);
  }
}
</script>
</body>
</html>
"""
