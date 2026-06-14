const runId = window.SEARCH_RUN_ID;

const ROLE = {
  boolean: { cls: "r-boolean", tag: "Boolean" },
  numeric: { cls: "r-numeric", tag: "Numeric" },
  time: { cls: "r-time", tag: "Time" },
  context: { cls: "r-context", tag: "Context" },
};

function pct(x) {
  return `${Math.round((x ?? 0) * 100)}%`;
}

function fmtNum(x) {
  if (x == null) return "—";
  return Number(x).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function headlineValue(e) {
  if (e.role === "boolean") return pct(e.boolean_probability);
  if (e.role === "time") return e.median_date ?? "—";
  if (e.role === "numeric") return e.expected_value != null ? `${fmtNum(e.expected_value)}${e.unit ? " " + e.unit : ""}` : "—";
  if (e.role === "context") return e.top_category ?? "—";
  return "—";
}

function metaRow(e) {
  if (e.role === "numeric") {
    return `<span>p10 <b>${fmtNum(e.p10)}</b></span><span>p50 <b>${fmtNum(e.p50)}</b></span>` +
      `<span>p90 <b>${fmtNum(e.p90)}</b></span><span>fit <b>${e.fit_method ?? "—"}</b></span>`;
  }
  if (e.role === "context") {
    return `<span>confidence <b>${pct(e.top_category_probability)}</b></span>`;
  }
  if (e.role === "boolean") {
    return `<span>yes-probability <b>${pct(e.boolean_probability)}</b></span>`;
  }
  if (e.role === "time") {
    return `<span>median resolution date</span>`;
  }
  return "";
}

async function renderDetail(container, objectId) {
  container.innerHTML = `<p class="muted">Loading distribution…</p>`;
  const res = await fetch(`/aggregate/${runId}/${encodeURIComponent(objectId)}`);
  if (!res.ok) { container.innerHTML = `<p class="warning">Failed to load distribution.</p>`; return; }
  const data = await res.json();
  const head = Object.keys(data.rows[0] ?? {});
  // Row values originate from Polymarket text (AI-structured, user-reviewed) and
  // are interpolated unescaped. Acceptable for this local single-user MVP; escape
  // cell values here if this ever becomes multi-user.
  const rows = data.rows.map((row) => `<tr>${head.map((k) => `<td class="num">${row[k]}</td>`).join("")}</tr>`).join("");
  container.innerHTML =
    `<table><thead><tr>${head.map((k) => `<th>${k}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>`;
}

function estimateCard(e, index) {
  const role = ROLE[e.role] ?? { cls: "", tag: e.role };
  const objectId = e.object_id ?? e.object_name;
  const card = document.createElement("div");
  card.className = `estimate-card ${role.cls}`;
  card.style.animationDelay = `${index * 70}ms`;
  card.innerHTML =
    `<div class="ec-top">
       <div>
         <span class="role-tag">${role.tag}</span>
         <div class="ec-name">${e.object_name}</div>
       </div>
       <div class="ec-value num">${headlineValue(e)}</div>
     </div>
     <div class="ec-meta">${metaRow(e)}</div>` +
    (e.anomalies && e.anomalies.length ? `<div class="anomaly">⚠ ${e.anomalies.join(" · ")}</div>` : "") +
    `<button class="btn btn-sm btn-ghost expand">Show distribution</button>
     <div class="detail"></div>`;

  const detail = card.querySelector(".detail");
  const button = card.querySelector(".expand");
  button.addEventListener("click", () => {
    const open = detail.classList.toggle("open");
    button.textContent = open ? "Hide distribution" : "Show distribution";
    if (open && !detail.dataset.loaded) {
      detail.dataset.loaded = "1";
      renderDetail(detail, objectId);
    }
  });
  return card;
}

async function init() {
  const narrative = document.getElementById("narrative");
  const res = await fetch(`/summarize/${runId}`, { method: "POST" });
  if (!res.ok) {
    narrative.classList.remove("loading");
    narrative.textContent = "Failed to load summary.";
    return;
  }
  const summary = await res.json();
  narrative.classList.remove("loading");
  narrative.textContent = summary.narrative;

  const container = document.getElementById("estimates");
  summary.point_estimates.forEach((e, i) => container.appendChild(estimateCard(e, i)));
}

init();
