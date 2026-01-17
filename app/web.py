from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>UFC Fight Predictor</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; max-width: 760px; margin: 40px auto; padding: 0 16px; }
    h1 { margin-bottom: 6px; }
    .muted { color: #666; margin-top: 0; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-top: 16px; }
    label { display:block; font-weight: 600; margin-top: 10px; }
    input { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 10px; font-size: 16px; }
    button { margin-top: 14px; padding: 10px 14px; border: 0; border-radius: 10px; font-size: 16px; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    pre { background: #111; color: #eee; padding: 12px; border-radius: 12px; overflow: auto; }
    .row { display:flex; gap: 12px; }
    .row > div { flex: 1; }
  </style>
</head>
<body>
  <h1>UFC Fight Predictor</h1>
  <p class="muted">Enter two fighters. Backend: XGBoost + Elo-derived features.</p>

  <div class="card">
    <div class="row">
      <div>
        <label>Red corner</label>
        <input id="red" placeholder="Jon Jones" />
      </div>
      <div>
        <label>Blue corner</label>
        <input id="blue" placeholder="Islam Makhachev" />
      </div>
    </div>
    <button id="btn">Predict</button>
  </div>

  <div class="card">
    <h3>Result</h3>
    <pre id="out">Click Predict…</pre>
  </div>

  <script>
    const btn = document.getElementById("btn");
    const out = document.getElementById("out");

    btn.addEventListener("click", async () => {
      const red = document.getElementById("red").value.trim();
      const blue = document.getElementById("blue").value.trim();

      if (!red || !blue) {
        out.textContent = "Please enter both fighter names.";
        return;
      }

      btn.disabled = true;
      out.textContent = "Running prediction…";

      try {
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ red, blue })
        });

        const data = await res.json();
        if (!res.ok) {
          out.textContent = "Error: " + (data.detail || JSON.stringify(data));
        } else {
          out.textContent = JSON.stringify(data, null, 2);
        }
      } catch (e) {
        out.textContent = "Network error: " + e;
      } finally {
        btn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""

@router.get("/", response_class=HTMLResponse)
def home():
    return HTML
