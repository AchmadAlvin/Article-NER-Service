// Sample presets for quick testing
const SAMPLES = [
  {
    name: "Banking Deal",
    title: "UOB Malaysia, RHB Banking and Maybank execute landmark MYOR-based transaction",
    body: "PETALING JAYA: UOB Malaysia, RHB Banking Group and Malayan Banking (Maybank) recently executed the first Malaysia Overnight Rate-based three-month transaction in Kuala Lumpur on 15 September 2026."
  },
  {
    name: "Politics & Policy",
    title: "Equal work, equal pay debated in Malaysian parliament",
    body: "Lim Guan Eng, the former finance minister and now adviser to the Democratic Action Party, delivered a keynote address in Penang outlining labor economic reforms."
  },
  {
    name: "Sports & Event",
    title: "MHC reschedules Deepavali Day matches after public feedback",
    body: "The Malaysian Hockey Confederation confirmed that all MHL matches in Bukit Jalil will be moved to next November following consultations with community leaders."
  },
  {
    name: "Zero Entities",
    title: "A simple guide to healthy habits",
    body: "Drinking clean water and sleeping early helps improve focus and physical energy."
  },
  {
    name: "Chinese News (中文)",
    title: "顺应公众反馈，MHC重新安排屠妖节比赛",
    body: "马来西亚曲棍球联合会确认，在与社区领袖协商后，武吉加里尔的所有MHL比赛将移至明年11月进行。"
  }
];

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("ner-form");
  const titleInput = document.getElementById("article-title");
  const bodyInput = document.getElementById("article-body");
  const modelSelect = document.getElementById("model-select");
  const submitBtn = document.getElementById("submit-btn");
  const spinner = document.getElementById("btn-spinner");
  const btnText = document.getElementById("btn-text");
  const alertBox = document.getElementById("alert-box");
  const samplesContainer = document.getElementById("samples-container");

  const titleCounter = document.getElementById("title-counter");
  const bodyCounter = document.getElementById("body-counter");

  const urlParams = new URLSearchParams(window.location.search);
  const sampleIdx = urlParams.get("sample");
  const activeTab = urlParams.get("tab") || (window.location.hash ? window.location.hash.substring(1) : null);

  // Results UI elements
  const resultsCard = document.getElementById("results-card");
  const emptyState = document.getElementById("empty-state");
  const resultsContent = document.getElementById("results-content");
  const totalCountEl = document.getElementById("total-count");
  const modelUsedEl = document.getElementById("model-used");
  const entityChipsContainer = document.getElementById("entity-chips");
  const highlightViewer = document.getElementById("highlight-viewer");
  const tableBody = document.getElementById("table-body");
  const jsonViewer = document.getElementById("json-viewer");

  // Render sample preset buttons
  SAMPLES.forEach(sample => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sample-btn";
    btn.textContent = sample.name;
    btn.onclick = () => {
      titleInput.value = sample.title;
      bodyInput.value = sample.body;
      updateCharCounters();
      clearAlert();
    };
    samplesContainer.appendChild(btn);
  });

  // Character counter listeners
  function updateCharCounters() {
    titleCounter.textContent = `${titleInput.value.length} / 1000`;
    bodyCounter.textContent = `${bodyInput.value.length} / 50000`;
  }
  titleInput.addEventListener("input", updateCharCounters);
  bodyInput.addEventListener("input", updateCharCounters);
  updateCharCounters();

  // Tab switching
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      document.getElementById(targetId).classList.add("active");
    });
  });

  function showAlert(message) {
    alertBox.textContent = message;
    alertBox.style.display = "block";
  }

  function clearAlert() {
    alertBox.style.display = "none";
    alertBox.textContent = "";
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    if (isLoading) {
      spinner.style.display = "inline-block";
      btnText.textContent = "Analyzing Entities...";
    } else {
      spinner.style.display = "none";
      btnText.textContent = "Extract Named Entities";
    }
  }

  // Handle form submission
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearAlert();

    const title = titleInput.value.trim();
    const body = bodyInput.value.trim();
    const model = modelSelect.value;

    // Panel validation
    if (!title) {
      showAlert("⚠️ Title cannot be empty.");
      titleInput.focus();
      return;
    }
    if (!body) {
      showAlert("⚠️ Body cannot be empty.");
      bodyInput.focus();
      return;
    }

    setLoading(true);

    try {
      const response = await fetch("/extract-entities", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ title, body, model })
      });

      const data = await response.json();

      if (!response.ok) {
        // Backend validation or error response (HTTP 400)
        const errMsg = data.detail || (data.errors ? data.errors.join(", ") : "Failed to analyze text");
        showAlert(`❌ Request Error (${response.status}): ${errMsg}`);
        setLoading(false);
        return;
      }

      const cleanedText = data.cleaned_text || `${title}\n\n${body}`;
      renderResults(data, cleanedText);
    } catch (err) {
      showAlert(`❌ Network Error: Could not connect to the backend service. (${err.message})`);
    } finally {
      setLoading(false);
    }

  });

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function renderResults(data, combinedText) {
    emptyState.style.display = "none";
    resultsContent.style.display = "block";

    const entities = data.entities || [];
    const meta = data.meta || {};

    totalCountEl.textContent = `${entities.length} entities found`;
    modelUsedEl.textContent = `Model: ${meta.model || 'unknown'}`;

    const langEl = document.getElementById("lang-detected");
    if (langEl) {
      langEl.style.display = "inline-block";
      langEl.textContent = `Language: ${(meta.language || 'en').toUpperCase()}`;
    }

    // Count by category
    const counts = {};
    entities.forEach(ent => {
      counts[ent.label] = (counts[ent.label] || 0) + 1;
    });

    // Render category summary chips
    entityChipsContainer.innerHTML = "";
    if (Object.keys(counts).length === 0) {
      entityChipsContainer.innerHTML = '<span class="stat-pill">No named entities detected</span>';
    } else {
      for (const [lbl, count] of Object.entries(counts)) {
        const chip = document.createElement("div");
        const chipClass = `chip-${lbl}` in document.body.style ? `chip-${lbl}` : (['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'EVENT', 'MONEY'].includes(lbl) ? `chip-${lbl}` : 'chip-OTHER');
        chip.className = `entity-chip ${chipClass}`;
        chip.innerHTML = `<span>${lbl}</span> <strong>${count}</strong>`;
        entityChipsContainer.appendChild(chip);
      }
    }

    // Render highlighted text span
    renderHighlights(combinedText, entities);

    // Render table
    tableBody.innerHTML = "";
    if (entities.length === 0) {
      tableBody.innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-muted);">No entities found in this text.</td></tr>';
    } else {
      entities.forEach((ent, idx) => {
        const row = document.createElement("tr");

        row.innerHTML = `
          <td><strong>${escapeHtml(ent.text)}</strong></td>
          <td><span class="entity-chip chip-${ent.label}">${ent.label}</span></td>
          <td><span class="code-span">${ent.start} &rarr; ${ent.end}</span></td>
        `;
        tableBody.appendChild(row);
      });
    }

    // Render raw JSON
    jsonViewer.textContent = JSON.stringify(data, null, 2);

    if (activeTab) {
      const tabBtn = document.querySelector(`.tab-btn[data-tab="${activeTab}"]`);
      if (tabBtn) tabBtn.click();
    }
  }

  function renderHighlights(text, entities) {
    if (!entities || entities.length === 0) {
      highlightViewer.textContent = text;
      return;
    }

    // Sort entities by start offset
    const sorted = [...entities].sort((a, b) => a.start - b.start);
    let html = "";
    let lastIdx = 0;

    sorted.forEach(ent => {
      // Overlap safeguard
      if (ent.start < lastIdx) return;

      // Append unhighlighted text prior to entity
      html += escapeHtml(text.substring(lastIdx, ent.start));

      // Append marked entity
      const entText = escapeHtml(text.substring(ent.start, ent.end));
      const confPercent = ent.confidence ? Math.round(ent.confidence * 100) : 90;
      const tooltip = `Category: ${ent.label} | Confidence: ${confPercent}%${ent.normalized ? ` | Canonical: ${ent.normalized}` : ''} | Span: [${ent.start}:${ent.end}]`;
      html += `<span class="mark-entity" data-label="${ent.label}" title="${escapeHtml(tooltip)}">${entText}<span class="badge-tag">${ent.label}</span></span>`;

      lastIdx = ent.end;
    });

    // Append remaining tail
    if (lastIdx < text.length) {
      html += escapeHtml(text.substring(lastIdx));
    }

    highlightViewer.innerHTML = html;
  }

  if (sampleIdx !== null) {
    const idx = parseInt(sampleIdx, 10);
    if (!isNaN(idx) && SAMPLES[idx]) {
      titleInput.value = SAMPLES[idx].title;
      bodyInput.value = SAMPLES[idx].body;
      updateCharCounters();
      if (urlParams.get("auto_submit") !== "false") {
        form.dispatchEvent(new Event("submit"));
      }
    }
  }
});
