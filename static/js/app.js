let currentResults = [];
let selectedIndex = -1;

const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const resultsBody = document.getElementById("resultsBody");
const fileHeader = document.getElementById("fileHeader");
const fileContent = document.getElementById("fileContent");
const tipsToggle = document.getElementById("tipsToggle");
const tipsContent = document.getElementById("tipsContent");
const advancedToggle = document.getElementById("advancedToggle");
const advancedContent = document.getElementById("advancedContent");

// Toggle tips section
tipsToggle.addEventListener("click", () => {
  const isExpanded = tipsContent.classList.toggle("expanded");
  tipsToggle.classList.toggle("expanded", isExpanded);
});

// Toggle advanced section
advancedToggle.addEventListener("click", () => {
  const isExpanded = advancedContent.classList.toggle("expanded");
  advancedToggle.classList.toggle("expanded", isExpanded);
});

searchInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") {
    performSearch();
  }
});

searchBtn.addEventListener("click", performSearch);

async function performSearch() {
  const query = searchInput.value.trim();
  if (!query) return;

  const resultsPanel = document.querySelector(".results-panel");
  resultsPanel.scrollTop = 0;

  searchBtn.disabled = true;
  searchBtn.innerHTML = '<span class="spinner"></span>Searching...';

  try {
    const response = await fetch("/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });

    const data = await response.json();
    currentResults = data.results || [];
    displayResults();
  } catch (error) {
    console.error("Search error:", error);
    resultsBody.innerHTML =
      '<tr><td colspan="4" class="empty-state">Search error occurred</td></tr>';
  } finally {
    searchBtn.disabled = false;
    searchBtn.textContent = "Search";
  }
}

function displayResults() {
  if (currentResults.length === 0) {
    resultsBody.innerHTML =
      '<tr><td colspan="4" class="empty-state">No results found</td></tr>';
    fileContent.textContent = "";
    fileHeader.textContent = "No file selected";
    return;
  }

  resultsBody.innerHTML = currentResults
    .map(
      (result, index) => `
          <tr class="result-row" onclick="selectResult(${index})">
              <td>${result.rank}</td>
              <td class="similarity">${result.score}</td>
              <td class="filename">${result.episode}</td>
              <td class="chunk-preview">${escapeHtml(result.preview)}</td>
          </tr>
      `
    )
    .join("");

  if (currentResults.length > 0) {
    selectResult(0);
  }
}

async function selectResult(index) {
  selectedIndex = index;
  const result = currentResults[index];

  document.querySelectorAll(".result-row").forEach((row, i) => {
    row.classList.toggle("selected", i === index);
  });

  fileHeader.textContent = result.episode;

  // Split scene by double newlines
  const sceneText = result.scene_text;
  const formattedLines = sceneText.split("\n\n");
  
  // Extract the parts: before highlight, highlight, after highlight
  const beforeLines = formattedLines.slice(0, result.window_start);
  const highlightLines = formattedLines.slice(result.window_start, result.window_end);
  const afterLines = formattedLines.slice(result.window_end);
  
  // Join back with double newlines
  const beforeText = beforeLines.join("\n\n");
  const highlightText = highlightLines.join("\n\n");
  const afterText = afterLines.join("\n\n");
  
  // Build the full display with highlight
  fileContent.innerHTML = 
    escapeHtml(beforeText) +
    (beforeText ? "\n\n" : "") +
    '<span class="highlight" id="highlighted-chunk">' +
    escapeHtml(highlightText) +
    '</span>' +
    (afterText ? "\n\n" : "") +
    escapeHtml(afterText);

  // Scroll to the highlighted section
  setTimeout(() => {
    const highlightElement = document.getElementById("highlighted-chunk");
    if (highlightElement) {
      highlightElement.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }, 100);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}