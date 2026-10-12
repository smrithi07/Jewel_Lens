let lastFile = null;

const STATUS = {
  match: { label: "Close match", color: "#8fc9a8" },
  uncertain: { label: "Similar styles", color: "#f2c078" },
  no_match: { label: "No match", color: "#e28a9a" },
};

const REASON_TEXT = {
  out_of_scope: "This looks like something we don't carry (for example a ring, watch or anklet). We only stock earrings, necklaces and bracelets.",
  low_similarity: "Nothing in our catalogue looks close enough to this piece.",
  no_results: "No catalogue items found for this piece.",
};

function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function filterParams() {
  const params = new URLSearchParams({ sort_by: document.getElementById("sortBy").value });
  const minPrice = document.getElementById("minPrice").value;
  const maxPrice = document.getElementById("maxPrice").value;
  if (minPrice) params.append("min_price", minPrice);
  if (maxPrice) params.append("max_price", maxPrice);
  return params;
}

async function upload() {
  const input = document.getElementById("fileInput");
  if (!input.files.length) {
    document.getElementById("status").innerText = "Please choose a photo first.";
    return;
  }
  lastFile = input.files[0];
  showQueryPreview(lastFile);
  await runPredict();
}

async function reFilter() {
  if (lastFile) await runPredict();
}

async function runPredict() {
  const statusEl = document.getElementById("status");
  const resultsEl = document.getElementById("results");
  statusEl.innerText = "Searching...";
  resultsEl.innerHTML = "";

  const form = new FormData();
  form.append("image", lastFile);

  try {
    const res = await fetch(`/predict?${filterParams().toString()}`, { method: "POST", body: form });
    const data = await res.json();

    if (data.error) {
      statusEl.innerText = "Error: " + data.error;
      return;
    }
    if (!data.items.length) {
      statusEl.innerText = data.message || "No jewellery detected.";
      document.getElementById("filterBar").style.display = "none";
      return;
    }

    document.getElementById("filterBar").style.display = "flex";
    statusEl.innerText = `Found ${data.items.length} item(s) in your photo.`;
    renderResults(data.items);
  } catch (e) {
    statusEl.innerText = "Something went wrong: " + e;
  }
}

function showQueryPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("queryPreviewImg").src = e.target.result;
    document.getElementById("queryPreviewContainer").style.display = "block";
  };
  reader.readAsDataURL(file);
}

function pct(x) {
  return x == null ? "" : `${Math.round(Number(x) * 100)}%`;
}

function capitalise(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : "";
}

function cardHTML(r) {
  return `
    <div class="candidate-card">
      <img src="${esc(r.image_url)}" class="result-img" alt="" loading="lazy" onerror="this.style.display='none'">
      <p class="title">${esc(r.title || r.catalogue_id)}</p>
      <p>Shop: <strong>${esc(r.shop)}</strong></p>
      <p>Price: ₹${esc(r.price ?? "N/A")}</p>
      <p>Similarity: ${esc(r.similarity)}</p>
      <div class="card-actions">
        ${r.listing_url ? `<a href="${esc(r.listing_url)}" target="_blank" rel="noopener">View listing</a>` : ""}
        <button class="more-btn" data-id="${esc(r.catalogue_id)}" data-title="${esc(r.title || r.catalogue_id)}">More like this</button>
      </div>
    </div>`;
}

function tiersHTML(closest, similar) {
  let html = "";
  if (closest && closest.length) {
    html += `<div class="tier"><h4>Closest matches</h4><div class="candidates">${closest.map(cardHTML).join("")}</div></div>`;
  }
  if (similar && similar.length) {
    html += `<div class="tier"><h4>${closest && closest.length ? "Similar items" : "Similar styles"}</h4>
             <div class="candidates">${similar.map(cardHTML).join("")}</div></div>`;
  }
  return html || "<p class='muted'>No items in this price range.</p>";
}

function renderResults(items) {
  const container = document.getElementById("results");
  container.innerHTML = "";

  items.forEach((item) => {
    const block = document.createElement("div");
    block.className = "item-block";
    const st = STATUS[item.status] || STATUS.uncertain;
    const category = capitalise(item.detected_category);

    block.innerHTML = `
      <div class="item-head">
        ${item.crop_image ? `<img class="crop-thumb" src="${esc(item.crop_image)}" alt="Detected ${esc(category)}">` : ""}
        <div class="item-info">
          <h3>Item ${item.item_index + 1}
            <span class="chip" style="background:${st.color}">${st.label}</span>
          </h3>
          <p class="category">${esc(category)} <span class="muted">· ${pct(item.category_confidence)} confident</span></p>
          ${item.status === "no_match" ? `<p class="reason">${esc(REASON_TEXT[item.status_reason] || REASON_TEXT.low_similarity)}</p>` : ""}
        </div>
      </div>
      <div class="item-results"></div>
      <div class="similar-panel" hidden></div>
    `;

    const resultsEl = block.querySelector(".item-results");
    if (item.status === "no_match") {
      resultsEl.innerHTML = `<button class="secondary show-anyway">Show the nearest catalogue items anyway</button>`;
      resultsEl.querySelector(".show-anyway").onclick = () => {
        resultsEl.innerHTML = tiersHTML(item.closest_matches, item.similar_items);
        bindMoreButtons(block);
      };
    } else {
      resultsEl.innerHTML = tiersHTML(item.closest_matches, item.similar_items);
    }
    container.appendChild(block);
    bindMoreButtons(block);
  });
}

function bindMoreButtons(root, block = root) {
  // buttons inside a "more like this" panel open the next panel in the same item block
  root.querySelectorAll(".more-btn").forEach((btn) => {
    btn.onclick = () => showSimilar(block, btn.dataset.id, btn.dataset.title);
  });
}

async function showSimilar(block, catalogueId, title) {
  const panel = block.querySelector(".similar-panel");
  panel.hidden = false;
  panel.innerHTML = `<p class="muted">Finding more like “${esc(title)}”…</p>`;
  try {
    const res = await fetch(`/similar/${encodeURIComponent(catalogueId)}?${filterParams().toString()}`);
    const data = await res.json();
    if (data.error) {
      panel.innerHTML = `<p class="muted">${esc(data.error)}</p>`;
      return;
    }
    panel.innerHTML = `
      <div class="panel-head">
        <h4>More like “${esc(data.query.title || catalogueId)}”</h4>
        <button class="secondary close-btn" aria-label="Close">Close</button>
      </div>
      ${tiersHTML(data.closest_matches, data.similar_items)}`;
    panel.querySelector(".close-btn").onclick = () => { panel.hidden = true; panel.innerHTML = ""; };
    bindMoreButtons(panel, block);
    panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (e) {
    panel.innerHTML = `<p class="muted">Something went wrong: ${esc(e)}</p>`;
  }
}
