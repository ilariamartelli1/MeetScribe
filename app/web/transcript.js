"use strict";

const status = document.querySelector("#status");
const error = document.querySelector("#error");
const download = document.querySelector("#download-transcript");
const jobId = new URLSearchParams(window.location.search).get("job_id");

function showError(text) {
  error.textContent = text;
  error.hidden = false;
}

if (!jobId || !/^[0-9a-f]{32}$/i.test(jobId)) {
  download.disabled = true;
  showError("No valid temporary job was selected.");
}

download.addEventListener("click", async () => {
  if (!jobId) return;
  download.disabled = true;
  error.hidden = true;
  status.textContent = "Preparing transcript download...";
  try {
    const response = await fetch(
      `/api/v1/jobs/${encodeURIComponent(jobId)}/transcript.txt`,
      { cache: "no-store" },
    );
    if (!response.ok) {
      let detail = `Download failed (${response.status}).`;
      try {
        detail = (await response.json()).detail || detail;
      } catch {
        // Keep the generic HTTP failure if the server response is not JSON.
      }
      throw new Error(detail);
    }
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = "meetscribe-transcript.txt";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status.textContent = "Transcript download started.";
  } catch (problem) {
    status.textContent = "Transcript download failed.";
    showError(problem.message);
  } finally {
    download.disabled = false;
  }
});
