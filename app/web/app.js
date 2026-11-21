"use strict";

const form = document.querySelector("#upload-form");
const audio = document.querySelector("#audio");
const language = document.querySelector("#language");
const submit = document.querySelector("#submit");
const cancel = document.querySelector("#cancel");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
const transcript = document.querySelector("#transcript");
const segments = document.querySelector("#segments");
const download = document.querySelector("#download");
let activeJob = null;
let cancellationPending = false;
let pollingError = false;
let timer = null;
let displayedSegments = [];

function message(text) {
  status.textContent = text;
}

function showError(text) {
  error.textContent = text;
  error.hidden = false;
}

function clearError() {
  error.textContent = "";
  error.hidden = true;
}

function setBusy(busy) {
  submit.disabled = busy;
  audio.disabled = busy;
  language.disabled = busy;
  cancel.disabled = !activeJob || cancellationPending;
}

function timecode(milliseconds) {
  const total = Math.floor(milliseconds);
  const hours = Math.floor(total / 3600000);
  const minutes = Math.floor(total / 60000) % 60;
  const seconds = Math.floor(total / 1000) % 60;
  const fraction = total % 1000;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(fraction).padStart(3, "0")}`;
}

function renderTranscript(result) {
  segments.replaceChildren();
  displayedSegments = result.segments.map((segment) => ({
    start_ms: segment.start_ms,
    end_ms: segment.end_ms,
    text: segment.text,
  }));
  displayedSegments.forEach((segment, index) => {
    const heading = document.createElement("h3");
    const text = document.createElement("p");
    const stamp = `${timecode(segment.start_ms)} - ${timecode(segment.end_ms)}`;
    heading.textContent = `Segment ${index + 1} (${stamp})`;
    text.id = `segment-${index}`;
    text.className = "segment-text";
    text.textContent = segment.text;
    segments.append(heading, text);
  });
  transcript.hidden = false;
}

async function request(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  const body = await response.json();
  return { response, body };
}

function finish() {
  activeJob = null;
  cancellationPending = false;
  clearTimeout(timer);
  setBusy(false);
}

function applySnapshot(snapshot, jobId) {
  if (activeJob !== jobId) return;
  if (snapshot.state === "succeeded") {
    renderTranscript(snapshot.result);
    finish();
    clearError();
    message("Transcription complete. Review the draft below.");
  } else if (snapshot.state === "failed") {
    finish();
    message("Transcription failed.");
    showError(snapshot.error?.message || "The transcription job failed.");
  } else if (snapshot.state === "cancelled") {
    finish();
    clearError();
    message("Job cancelled.");
  } else if (snapshot.state === "cancelling" || cancellationPending) {
    message("Cancelling job...");
  } else if (snapshot.state === "running") {
    message("Transcribing and analyzing speakers...");
  } else {
    finish();
    showError("The server returned an unknown job state.");
  }
}

async function poll(jobId) {
  if (activeJob !== jobId) return;
  try {
    const { response, body } = await request(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
    if (!response.ok) {
      if (response.status === 404) {
        finish();
        showError(body.detail || "The job is no longer available.");
        return;
      }
      throw new Error(body.detail || `Status request failed (${response.status}).`);
    }
    if (pollingError) {
      pollingError = false;
      clearError();
    }
    applySnapshot(body, jobId);
  } catch (problem) {
    if (activeJob === jobId) {
      pollingError = true;
      showError(`Could not check the job: ${problem.message}. Retrying...`);
    }
  }
  if (activeJob === jobId) timer = setTimeout(() => poll(jobId), 1000);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (activeJob || submit.disabled || !audio.files.length) return;
  const selectedFile = audio.files[0];
  const selectedLanguage = language.value;
  setBusy(true);
  pollingError = false;
  clearError();
  message("Uploading and validating recording...");
  let replacement = null;
  try {
    while (true) {
      const data = new FormData();
      data.append("file", selectedFile);
      data.append("language", selectedLanguage);
      if (replacement) data.append("replace_oldest_job_id", replacement);
      const { response, body } = await request("/api/v1/jobs", { method: "POST", body: data });
      if (response.status === 409 && body.oldest_job_id) {
        if (!window.confirm("The result limit is full. Replace the oldest result?")) {
          message("Upload cancelled; the oldest result was kept.");
          setBusy(false);
          return;
        }
        replacement = body.oldest_job_id;
        continue;
      }
      if (!response.ok) throw new Error(body.detail || `Upload failed (${response.status}).`);
      activeJob = body.job_id;
      setBusy(true);
      message("Transcribing and analyzing speakers...");
      poll(activeJob);
      return;
    }
  } catch (problem) {
    setBusy(false);
    message("Upload failed.");
    showError(problem.message);
  }
});

cancel.addEventListener("click", async () => {
  const jobId = activeJob;
  if (!jobId || cancellationPending) return;
  cancellationPending = true;
  setBusy(true);
  clearError();
  message("Cancelling job...");
  try {
    const { response, body } = await request(`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
    if (activeJob !== jobId) return;
    if (response.status === 409) {
      cancellationPending = false;
      setBusy(true);
      return;
    }
    if (!response.ok) throw new Error(body.detail || `Cancellation failed (${response.status}).`);
    applySnapshot(body, jobId);
  } catch (problem) {
    if (activeJob === jobId) {
      cancellationPending = false;
      setBusy(true);
      pollingError = false;
      showError(`Could not cancel the job: ${problem.message}`);
    }
  }
});

download.addEventListener("click", () => {
  const lines = ["DRAFT / UNREVIEWED", ""];
  displayedSegments.forEach((segment) => {
    const stamp = `${timecode(segment.start_ms)} - ${timecode(segment.end_ms)}`;
    lines.push(`[${stamp}] ${segment.text}`, "");
  });
  const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "meetscribe-draft.txt";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
