let lastFile = null;

async function upload() {
  const input = document.getElementById("fileInput");
  if (!input.files.length) {
    document.getElementById("status").innerText = "Please choose a photo first.";
    return;
  }
  lastFile = input.files[0];
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

  const sortBy = document.getElementById("sortBy").value;
  const minPrice = document.getElementById("minPrice").value;
  const maxPrice = document.getElementById("maxPrice").value;

  const params = new URLSearchParams({ sort_by: sortBy });
  if (minPrice) params.append("min_price", minPrice);
  if (maxPrice) params.append("max_price", maxPrice);

  const form = new FormData();
  form.append("image", lastFile);

  try {
    const res = await fetch(`/predict?${params.toString()}`, { method: "POST", body: form });
    const data = await res.json();

    if (data.error) {
      statusEl.innerText = "Error: " + data.error;
      return;
    }
    if (!data.items.length) {
      statusEl.innerText = data.message || "No jewelry detected.";
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

function renderResults(items) {
  const container = document.getElementById("results");
  container.innerHTML = "";

  items.forEach((item) => {
    const block = document.createElement("div");
    block.className = "item-block";

    const statusColor = { match: "#8fc9a8", uncertain: "#f2c078", no_match: "#e28a9a" }[item.status];
    block.innerHTML = `
      <h3>Item ${item.item_index + 1} — detected as ${item.detected_category}
        <span style="background:${statusColor}">${item.status.toUpperCase()}</span>
      </h3>
      <p>Similarity: ${item.similarity} | Margin: ${item.margin}</p>
      <div class="candidates"></div>
    `;

    const candidatesEl = block.querySelector(".candidates");
    if (item.status === "no_match") {
      candidatesEl.innerHTML = "<p>No confident match — this item may not be in our catalogue.</p>";
    } else {
      item.results.forEach((r) => {
        const card = document.createElement("div");
        card.className = "candidate-card";
        card.innerHTML = `
          <img src="${r.image_url}" class="result-img" onerror="this.style.display='none'">
          <p class="title">${r.title || r.catalogue_id}</p>
          <p>Shop: <strong>${r.shop}</strong></p>
          <p>Price: ₹${r.price ?? "N/A"}</p>
          <p>Similarity: ${r.similarity}</p>
          ${r.listing_url ? `<a href="${r.listing_url}" target="_blank">View listing</a>` : ""}
        `;
        candidatesEl.appendChild(card);
      });
    }

    container.appendChild(block);
  });
}