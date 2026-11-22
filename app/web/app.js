"use strict";

const form = document.querySelector("#upload-form");
const audio = document.querySelector("#audio");
const language = document.querySelector("#language");
const submit = document.querySelector("#submit");
const cancel = document.querySelector("#cancel");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
const minutesSection = document.querySelector("#minutes");
const reviewStatus = document.querySelector("#review-status");
const summaryText = document.querySelector("#summary-text");
const topics = document.querySelector("#topics");
const decisions = document.querySelector("#decisions");
const actionItems = document.querySelector("#action-items");
const saveMinutes = document.querySelector("#save-minutes");
const markReviewed = document.querySelector("#mark-reviewed");
const downloadMinutesTxt = document.querySelector("#download-minutes-txt");
const downloadMinutesJson = document.querySelector("#download-minutes-json");
const regenerateMinutes = document.querySelector("#regenerate-minutes");
const retryMinutes = document.querySelector("#retry-minutes");
const transcriptTools = document.querySelector("#transcript-tools");
const downloadTranscript = document.querySelector("#download-transcript");
const transcriptError = document.querySelector("#transcript-error");

let activeJob = null;
let activeStage = null;
let latestJobId = null;
let minutesRevision = null;
let reviewState = null;
let editorMinutes = null;
let dirty = false;
let staleMinutes = false;
let cancellationPending = false;
let pollingError = false;
let timer = null;
let nextFieldId = 0;

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
  cancel.textContent =
    activeStage === "minutes" ? "Cancel summary generation" : "Cancel transcription";
  if (busy) {
    saveMinutes.disabled = true;
    markReviewed.disabled = true;
    downloadMinutesTxt.disabled = true;
    downloadMinutesJson.disabled = true;
    regenerateMinutes.disabled = true;
  } else {
    updateEditorButtons();
  }
}

function finish() {
  activeJob = null;
  activeStage = null;
  cancellationPending = false;
  clearTimeout(timer);
  setBusy(false);
}

async function request(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  let body = {};
  try {
    body = await response.json();
  } catch {
    if (response.ok) throw new Error("The server returned an invalid response.");
  }
  return { response, body };
}

function makeField(labelText, type, value, onInput) {
  const wrapper = document.createElement("div");
  wrapper.className = "editor-field";
  const label = document.createElement("label");
  const input =
    type === "textarea"
      ? document.createElement("textarea")
      : document.createElement("input");
  const id = `minutes-field-${nextFieldId++}`;
  label.htmlFor = id;
  label.textContent = labelText;
  input.id = id;
  if (type !== "textarea") input.type = type;
  input.value = value ?? "";
  input.disabled = staleMinutes;
  input.addEventListener("input", onInput);
  wrapper.append(label, input);
  return wrapper;
}

function setDirty() {
  dirty = true;
  reviewStatus.textContent = "Unsaved edits — save before review or export.";
  updateEditorButtons();
}

function updateEditorButtons() {
  const ready = editorMinutes !== null && minutesRevision !== null;
  saveMinutes.disabled = !ready || !dirty || Boolean(activeJob);
  markReviewed.disabled =
    !ready ||
    dirty ||
    reviewState === "REVIEWED" ||
    staleMinutes ||
    Boolean(activeJob);
  downloadMinutesTxt.disabled = !ready || dirty || Boolean(activeJob);
  downloadMinutesJson.disabled = !ready || dirty || Boolean(activeJob);
  regenerateMinutes.disabled = !ready || Boolean(activeJob);
}

function renderSimpleCollection(container, collectionName, itemName) {
  container.replaceChildren();
  editorMinutes[collectionName].forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "editor-row";
    row.append(
      makeField(itemName, "text", item.text, (event) => {
        editorMinutes[collectionName][index].text = event.target.value;
        setDirty();
      }),
    );
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = `Remove ${itemName.toLowerCase()}`;
    remove.disabled = staleMinutes;
    remove.addEventListener("click", () => {
      editorMinutes[collectionName].splice(index, 1);
      renderEditor();
      setDirty();
    });
    row.append(remove);
    container.append(row);
  });
}

function renderActions() {
  actionItems.replaceChildren();
  editorMinutes.action_items.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "editor-row action-row";
    row.append(
      makeField("Action", "textarea", item.description, (event) => {
        editorMinutes.action_items[index].description = event.target.value;
        setDirty();
      }),
      makeField("Owner", "text", item.owner_name, (event) => {
        editorMinutes.action_items[index].owner_name =
          event.target.value.trim() || null;
        setDirty();
      }),
      makeField("Deadline wording", "text", item.deadline_text, (event) => {
        editorMinutes.action_items[index].deadline_text =
          event.target.value.trim() || null;
        setDirty();
      }),
      makeField("Absolute date", "date", item.deadline_date, (event) => {
        editorMinutes.action_items[index].deadline_date =
          event.target.value || null;
        setDirty();
      }),
    );
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Remove action item";
    remove.disabled = staleMinutes;
    remove.addEventListener("click", () => {
      editorMinutes.action_items.splice(index, 1);
      renderEditor();
      setDirty();
    });
    row.append(remove);
    actionItems.append(row);
  });
}

function renderEditor() {
  if (editorMinutes === null) return;
  summaryText.value = editorMinutes.summary;
  renderSimpleCollection(topics, "topics", "Topic");
  renderSimpleCollection(decisions, "decisions", "Decision");
  renderActions();
}

function showMinutes(snapshot) {
  const minutes = snapshot.minutes;
  if (
    !minutes ||
    typeof minutes.summary !== "string" ||
    !Array.isArray(minutes.topics) ||
    !Array.isArray(minutes.decisions) ||
    !Array.isArray(minutes.action_items)
  ) {
    finish();
    message("Summary generation failed.");
    showError("The server returned invalid draft minutes.");
    retryMinutes.hidden = false;
    return;
  }
  latestJobId = snapshot.job_id;
  minutesRevision = snapshot.minutes_revision;
  reviewState = snapshot.review_status;
  staleMinutes = Boolean(snapshot.stale);
  editorMinutes = JSON.parse(JSON.stringify(minutes));
  dirty = false;
  summaryText.disabled = staleMinutes;
  minutesSection.hidden = false;
  retryMinutes.hidden = true;
  renderEditor();
  reviewStatus.textContent = staleMinutes
    ? `STALE — based on transcript revision ${snapshot.minutes_source_transcript_revision}; current revision is ${snapshot.transcript_revision}. Regenerate before marking reviewed.`
    : snapshot.review_status || "DRAFT / UNREVIEWED";
  for (const selector of ["#add-topic", "#add-decision", "#add-action"]) {
    document.querySelector(selector).disabled = staleMinutes;
  }
  updateEditorButtons();
  finish();
  if (snapshot.regeneration_error) {
    showError(
      `Regeneration failed; the previous draft was retained. ${snapshot.regeneration_error.message}`,
    );
    message("Summary regeneration failed.");
  } else {
    clearError();
    message(
      snapshot.regeneration_cancelled
        ? "Regeneration cancelled; the previous draft was retained."
        : staleMinutes
          ? "The source transcript changed. Regenerate the summary before review."
          : "Draft summary ready. Review and edit it before sharing.",
    );
  }
}

function applyMinutesSnapshot(snapshot, jobId) {
  if (activeJob !== jobId) return;
  if (snapshot.regenerating) {
    message(
      snapshot.regeneration_cancelling || cancellationPending
        ? "Cancelling summary regeneration..."
        : "Regenerating draft minutes...",
    );
  } else if (snapshot.state === "succeeded") {
    showMinutes(snapshot);
  } else if (snapshot.state === "failed") {
    finish();
    message("Summary generation failed.");
    showError(snapshot.error?.message || "The minutes job failed.");
    retryMinutes.hidden = false;
  } else if (snapshot.state === "cancelled") {
    finish();
    message("Summary generation cancelled.");
    retryMinutes.hidden = false;
  } else if (snapshot.state === "cancelling" || cancellationPending) {
    message("Cancelling summary generation...");
  } else if (snapshot.state === "running") {
    message("Generating draft minutes...");
  } else {
    finish();
    showError("The server returned an unknown minutes job state.");
  }
}

async function applyTranscriptionSnapshot(snapshot, jobId) {
  if (activeJob !== jobId) return;
  if (snapshot.state === "succeeded") {
    latestJobId = jobId;
    transcriptTools.hidden = false;
    activeStage = "minutes";
    await beginMinutes(jobId);
  } else if (snapshot.state === "failed") {
    finish();
    message("Transcription failed.");
    showError(snapshot.error?.message || "The transcription job failed.");
  } else if (snapshot.state === "cancelled") {
    finish();
    message("Transcription cancelled.");
  } else if (snapshot.state === "cancelling" || cancellationPending) {
    message("Cancelling transcription...");
  } else if (snapshot.state === "running") {
    message("Transcribing and analyzing speakers...");
  } else {
    finish();
    showError("The server returned an unknown transcription state.");
  }
}

async function beginMinutes(jobId) {
  if (activeJob !== jobId) return;
  activeStage = "minutes";
  setBusy(true);
  message("Generating draft minutes...");
  try {
    const { response, body } = await request(
      `/api/v1/jobs/${encodeURIComponent(jobId)}/minutes`,
      { method: "POST" },
    );
    if (!response.ok) {
      throw new Error(body.detail || `Minutes request failed (${response.status}).`);
    }
    applyMinutesSnapshot(body, jobId);
  } catch (problem) {
    if (activeJob === jobId) {
      finish();
      message("Summary generation could not start.");
      showError(problem.message);
      retryMinutes.hidden = false;
    }
  }
}

async function poll(jobId) {
  if (activeJob !== jobId) return;
  try {
    const endpoint =
      activeStage === "minutes"
        ? `/api/v1/jobs/${encodeURIComponent(jobId)}/minutes`
        : `/api/v1/jobs/${encodeURIComponent(jobId)}`;
    const { response, body } = await request(endpoint);
    if (!response.ok) {
      if (response.status === 404) {
        finish();
        showError(body.detail || "The job is no longer available.");
        retryMinutes.hidden = true;
        return;
      }
      throw new Error(body.detail || `Status request failed (${response.status}).`);
    }
    if (pollingError) {
      pollingError = false;
      clearError();
    }
    if (activeStage === "minutes") {
      applyMinutesSnapshot(body, jobId);
    } else {
      await applyTranscriptionSnapshot(body, jobId);
    }
  } catch (problem) {
    if (activeJob === jobId) {
      pollingError = true;
      showError(`Could not check the job: ${problem.message}. Retrying...`);
    }
  }
  if (activeJob === jobId) timer = setTimeout(() => poll(jobId), 1000);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function downloadFile(url, filename, reportError = showError) {
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      let detail = `Download failed (${response.status}).`;
      try {
        detail = (await response.json()).detail || detail;
      } catch {
        // Keep the generic HTTP failure if the server response is not JSON.
      }
      throw new Error(detail);
    }
    downloadBlob(await response.blob(), filename);
  } catch (problem) {
    reportError(problem.message);
  }
}

function startMinutes(jobId) {
  latestJobId = jobId;
  activeJob = jobId;
  activeStage = "minutes";
  cancellationPending = false;
  retryMinutes.hidden = true;
  setBusy(true);
  clearError();
  message("Generating draft minutes...");
  beginMinutes(jobId).then(() => {
    if (activeJob === jobId) timer = setTimeout(() => poll(jobId), 1000);
  });
}

summaryText.addEventListener("input", () => {
  if (editorMinutes !== null) {
    editorMinutes.summary = summaryText.value;
    setDirty();
  }
});

document.querySelector("#add-topic").addEventListener("click", () => {
  if (editorMinutes === null) return;
  editorMinutes.topics.push({ text: "", source_segment_ids: [] });
  renderEditor();
  setDirty();
});

document.querySelector("#add-decision").addEventListener("click", () => {
  if (editorMinutes === null) return;
  editorMinutes.decisions.push({ text: "", source_segment_ids: [] });
  renderEditor();
  setDirty();
});

document.querySelector("#add-action").addEventListener("click", () => {
  if (editorMinutes === null) return;
  editorMinutes.action_items.push({
    description: "",
    owner_name: null,
    owner_speaker_id: null,
    deadline_text: null,
    deadline_date: null,
    source_segment_ids: [],
  });
  renderEditor();
  setDirty();
});

saveMinutes.addEventListener("click", async () => {
  if (!latestJobId || !editorMinutes || minutesRevision === null || !dirty) return;
  saveMinutes.disabled = true;
  try {
    const { response, body } = await request(
      `/api/v1/jobs/${encodeURIComponent(latestJobId)}/minutes`,
      {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          expected_revision: minutesRevision,
          minutes: editorMinutes,
        }),
      },
    );
    if (!response.ok) {
      throw new Error(body.detail || `Saving edits failed (${response.status}).`);
    }
    showMinutes(body);
    message("Summary edits saved.");
  } catch (problem) {
    saveMinutes.disabled = false;
    showError(problem.message);
  }
});

markReviewed.addEventListener("click", async () => {
  if (!latestJobId || minutesRevision === null || dirty) return;
  markReviewed.disabled = true;
  try {
    const { response, body } = await request(
      `/api/v1/jobs/${encodeURIComponent(latestJobId)}/minutes/review`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ expected_revision: minutesRevision }),
      },
    );
    if (!response.ok) {
      throw new Error(body.detail || `Review update failed (${response.status}).`);
    }
    showMinutes(body);
    message("Current summary revision marked as reviewed.");
  } catch (problem) {
    markReviewed.disabled = false;
    showError(problem.message);
  }
});

downloadMinutesTxt.addEventListener("click", () => {
  if (latestJobId && !dirty) {
    downloadFile(
      `/api/v1/jobs/${encodeURIComponent(latestJobId)}/minutes.txt`,
      "meetscribe-minutes.txt",
    );
  }
});

downloadMinutesJson.addEventListener("click", () => {
  if (latestJobId && !dirty) {
    downloadFile(
      `/api/v1/jobs/${encodeURIComponent(latestJobId)}/minutes.json`,
      "meetscribe-minutes.json",
    );
  }
});

downloadTranscript.addEventListener("click", () => {
  if (latestJobId) {
    transcriptError.hidden = true;
    transcriptError.textContent = "";
    downloadFile(
      `/api/v1/jobs/${encodeURIComponent(latestJobId)}/transcript.txt`,
      "meetscribe-transcript.txt",
      (text) => {
        transcriptError.textContent = text;
        transcriptError.hidden = false;
      },
    );
  }
});

retryMinutes.addEventListener("click", () => {
  if (latestJobId) startMinutes(latestJobId);
});

regenerateMinutes.addEventListener("click", async () => {
  if (!latestJobId || minutesRevision === null || activeJob) return;
  if (
    !window.confirm(
      "Regeneration will replace the saved summary and any unsaved edits if it succeeds. Continue?",
    )
  ) {
    return;
  }

  const jobId = latestJobId;
  activeJob = jobId;
  activeStage = "minutes";
  cancellationPending = false;
  setBusy(true);
  clearError();
  message("Regenerating draft minutes...");
  try {
    const { response, body } = await request(
      `/api/v1/jobs/${encodeURIComponent(jobId)}/minutes/regenerate`,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          expected_revision: minutesRevision,
          confirm_overwrite: true,
        }),
      },
    );
    if (!response.ok) {
      throw new Error(
        body.detail || `Regeneration request failed (${response.status}).`,
      );
    }
    applyMinutesSnapshot(body, jobId);
    if (activeJob === jobId) timer = setTimeout(() => poll(jobId), 1000);
  } catch (problem) {
    if (activeJob === jobId) {
      finish();
      message("Summary regeneration could not start.");
      showError(problem.message);
    }
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (activeJob || submit.disabled || !audio.files.length) return;
  const selectedFile = audio.files[0];
  const selectedLanguage = language.value;
  setBusy(true);
  pollingError = false;
  retryMinutes.hidden = true;
  clearError();
  message("Uploading and validating recording...");
  let replacement = null;
  try {
    while (true) {
      const data = new FormData();
      data.append("file", selectedFile);
      data.append("language", selectedLanguage);
      if (replacement) data.append("replace_oldest_job_id", replacement);
      const { response, body } = await request("/api/v1/jobs", {
        method: "POST",
        body: data,
      });
      if (response.status === 409 && body.oldest_job_id) {
        if (!window.confirm("The result limit is full. Replace the oldest result?")) {
          message("Upload cancelled; the oldest result was kept.");
          setBusy(false);
          return;
        }
        replacement = body.oldest_job_id;
        continue;
      }
      if (!response.ok) {
        throw new Error(body.detail || `Upload failed (${response.status}).`);
      }
      activeJob = body.job_id;
      activeStage = "transcription";
      latestJobId = body.job_id;
      minutesSection.hidden = true;
      editorMinutes = null;
      minutesRevision = null;
      reviewState = null;
      dirty = false;
      staleMinutes = false;
      transcriptTools.hidden = true;
      transcriptError.hidden = true;
      transcriptError.textContent = "";
      updateEditorButtons();
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
  message(
    activeStage === "minutes"
      ? "Cancelling summary generation..."
      : "Cancelling transcription...",
  );
  const endpoint =
    activeStage === "minutes"
      ? `/api/v1/jobs/${encodeURIComponent(jobId)}/minutes/cancel`
      : `/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`;
  try {
    const { response, body } = await request(endpoint, { method: "POST" });
    if (activeJob !== jobId) return;
    if (response.status === 409) {
      cancellationPending = false;
      setBusy(true);
      return;
    }
    if (!response.ok) {
      throw new Error(body.detail || `Cancellation failed (${response.status}).`);
    }
    if (activeStage === "minutes") {
      applyMinutesSnapshot(body, jobId);
    } else {
      await applyTranscriptionSnapshot(body, jobId);
    }
  } catch (problem) {
    if (activeJob === jobId) {
      cancellationPending = false;
      setBusy(true);
      pollingError = false;
      showError(`Could not cancel the job: ${problem.message}`);
    }
  }
});
