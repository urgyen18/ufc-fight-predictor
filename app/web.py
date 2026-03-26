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
</head>
<body>
  <h2>UFC Fight Predictor</h2>

  <label>Red fighter</label>
  <input id="red" list="fighters" placeholder="Start typing..." autocomplete="off" />

  <label>Blue fighter</label>
  <input id="blue" list="fighters" placeholder="Start typing..." autocomplete="off" />

  <datalist id="fighters"></datalist>

  <button onclick="predict()">Predict</button>

  <pre id="out">—</pre>

<script>
console.log("UI script loaded");

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

["red","blue"].forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener("input", (e) => {
    console.log("typing:", id, e.target.value);
    updateSuggestions(e.target.value);
  });
  el.addEventListener("focus", (e) => updateSuggestions(e.target.value));
});

async function predict() {
  const red = document.getElementById("red").value.trim();
  const blue = document.getElementById("blue").value.trim();

  const res = await fetch("/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ red, blue })
  });

  const data = await res.json();
  document.getElementById("out").textContent = JSON.stringify(data, null, 2);
}
</script>
</body>
</html>
"""
