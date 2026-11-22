"""Verify summary-only browser flows with fake inference.

Set MEETSCRIBE_BROWSER_TESTS=1 and install a Chromium browser to run these tests.
"""

import json
import os
import shutil
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import uvicorn
from playwright.sync_api import Browser, Page, Route, sync_playwright

import app.main as main
from app.core.asr_contract import Language, ModelProvenance, TranscriptSegment
from app.core.diarization_contract import (
    Diarization,
    DiarizationProvenance,
    SpeakerTurn,
)
from app.core.minutes_contract import MinutesDraft
from app.core.minutes_extractor import MinutesExtractionResult
from app.core.speaker_alignment import AttributedTranscript, SpeakerAttribution

pytestmark = pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_BROWSER_TESTS") != "1",
    reason="Set MEETSCRIBE_BROWSER_TESTS=1 to run Chromium browser tests.",
)

SAMPLE = Path(__file__).resolve().parents[1] / "data/smoke/synthetic-meeting.wav"
UNTRUSTED = '<img src=x onerror="window.injected=true"> no'
MINUTES_INJECTION = '<img src=x onerror="window.injected=true"> minutes'


class FakeBrowserJob:
    def __init__(self, transcriber_factory: object, diarizer_factory: object) -> None:
        pass

    def __call__(self, audio_path: Path, language: Language) -> AttributedTranscript:
        if language == "en":
            time.sleep(30)
        provenance = ModelProvenance("fake", "1", "model", "revision", "digest", "cpu")
        turns = (SpeakerTurn(100, 900, "speaker-1"),)
        return AttributedTranscript(
            language=language,
            duration_ms=1000,
            segments=(TranscriptSegment(100, 900, UNTRUSTED),),
            provenance=provenance,
            diarization=Diarization(
                duration_ms=1000,
                turns=turns,
                exclusive_turns=turns,
                provenance=DiarizationProvenance(
                    "fake", "1", "model", "revision", "cpu"
                ),
            ),
            attributions=(SpeakerAttribution(100, 900, "Voice 1"),),
        )


class FakeBrowserMinutesExtractor:
    def __init__(self, client: object) -> None:
        pass

    async def extract(
        self, transcript: AttributedTranscript
    ) -> MinutesExtractionResult:
        return MinutesExtractionResult(
            minutes=MinutesDraft(
                summary=MINUTES_INJECTION,
                topics=(),
                decisions=(),
                action_items=(),
            ),
            included_segment_ids=tuple(
                f"segment-{index}" for index in range(1, len(transcript.segments) + 1)
            ),
            context_size=4096,
            prompt_token_upper_bound=256,
            prompt_tokens=128,
            generated_tokens=64,
        )


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setattr(main, "SpeakerAttributedJob", FakeBrowserJob)
    monkeypatch.setattr(
        main, "create_ollama_minutes_extractor", FakeBrowserMinutesExtractor
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(main.app, log_level="warning"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started:
                if not thread.is_alive() or time.monotonic() >= deadline:
                    raise RuntimeError("Browser test server failed to start.")
                time.sleep(0.01)
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            if thread.is_alive():
                raise RuntimeError("Browser test server failed to stop.")


@pytest.fixture
def browser() -> Iterator[Browser]:
    executable = os.environ.get("MEETSCRIBE_CHROME_PATH") or shutil.which(
        "google-chrome"
    )
    with sync_playwright() as playwright:
        options: dict[str, Any] = {"headless": True}
        if executable:
            options["executable_path"] = executable
        instance = playwright.chromium.launch(**options)
        try:
            yield instance
        finally:
            instance.close()


@pytest.fixture
def page(browser: Browser, server: str) -> Iterator[Page]:
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()
    page.goto(server)
    try:
        yield page
    finally:
        context.close()


def select_sample(page: Page) -> None:
    page.locator("#audio").set_input_files(str(SAMPLE))


def _minutes_snapshot(job_id: str, state: str = "succeeded") -> dict[str, object]:
    return {
        "job_id": job_id,
        "state": state,
        "minutes": {
            "summary": "A concise draft summary.",
            "topics": [],
            "decisions": [],
            "action_items": [],
        }
        if state == "succeeded"
        else None,
        "error": None,
        "transcript_revision": 1,
        "minutes_revision": 1 if state == "succeeded" else None,
        "minutes_source_transcript_revision": 1 if state == "succeeded" else None,
        "review_status": "DRAFT / UNREVIEWED" if state == "succeeded" else None,
        "stale": False,
        "regenerating": False,
        "regeneration_cancelling": False,
        "regeneration_error": None,
        "regeneration_cancelled": False,
        "extraction_metrics": None,
    }


def test_real_api_upload_summary_edit_review_and_direct_transcript_download(
    page: Page, tmp_path: Path
) -> None:
    assert (
        "synthetic or explicitly authorized"
        in page.locator(".notice").first.inner_text()
    )
    page.locator("#language").select_option("it")
    select_sample(page)
    page.get_by_role("button", name="Create draft summary").click()
    page.get_by_text(
        "Draft summary ready. Review and edit it before sharing."
    ).wait_for()

    assert page.locator("#minutes").is_visible()
    assert page.locator("#summary-text").input_value() == MINUTES_INJECTION
    assert UNTRUSTED not in page.locator("body").inner_text()
    assert page.locator("#minutes img").count() == 0
    assert page.evaluate("window.injected") is None
    assert page.locator("#segments, #transcript, .segment-text").count() == 0
    assert page.locator("#minutes [contenteditable]").count() == 0
    assert page.locator("#transcript-tools").is_visible()
    assert page.locator("#review-status").inner_text() == "DRAFT / UNREVIEWED"

    page.locator("#summary-text").fill("La città rivedrà il piano.")
    page.get_by_role("button", name="Add action item").click()
    action_editor = page.locator("#action-items")
    action_editor.get_by_label("Action").fill("Ask Marta to review the plan")
    action_editor.get_by_label("Owner").fill("Marta")
    action_editor.get_by_label("Deadline wording").fill("March 5, 2027")
    action_editor.get_by_label("Absolute date").fill("2027-03-05")
    page.get_by_role("button", name="Save edits").click()
    page.get_by_text("Summary edits saved.").wait_for()
    assert page.locator("#review-status").inner_text() == "DRAFT / UNREVIEWED"

    with page.expect_download() as downloaded:
        page.get_by_role("button", name="Download TXT").click()
    saved = tmp_path / "minutes.txt"
    downloaded.value.save_as(saved)
    text = saved.read_text(encoding="utf-8")
    assert text.startswith("DRAFT / UNREVIEWED\n")
    assert "La città rivedrà il piano." in text
    assert "Marta" in text
    assert UNTRUSTED not in text

    with page.expect_download() as downloaded:
        page.get_by_role("button", name="Download JSON").click()
    saved = tmp_path / "minutes.json"
    downloaded.value.save_as(saved)
    export = json.loads(saved.read_text(encoding="utf-8"))
    assert export["schema_version"] == 1
    assert export["minutes_revision"] == 2
    assert export["minutes"]["summary"] == "La città rivedrà il piano."
    assert export["segments"][0]["text"] == UNTRUSTED

    page.get_by_role("button", name="Mark as reviewed").click()
    page.get_by_text("Current summary revision marked as reviewed.").wait_for()
    assert page.locator("#review-status").inner_text() == "REVIEWED"

    page.locator("#summary-text").fill("Unsaved summary changes.")
    original_url = page.url
    page.route(
        "**/transcript.txt",
        lambda route: route.fulfill(status=404, json={"detail": "Transcript expired."}),
    )
    page.get_by_role("button", name="Download transcript TXT").click()
    page.get_by_text("Transcript expired.").wait_for()
    assert page.url == original_url
    assert page.locator("#summary-text").input_value() == "Unsaved summary changes."
    assert page.get_by_role("button", name="Save edits").is_enabled()
    page.unroute("**/transcript.txt")

    assert UNTRUSTED not in page.locator("body").inner_text()
    with page.expect_download() as downloaded:
        page.get_by_role("button", name="Download transcript TXT").click()
    saved = tmp_path / "transcript.txt"
    downloaded.value.save_as(saved)
    transcript = saved.read_text(encoding="utf-8")
    assert "Speaker labels are anonymous" in transcript
    assert f"Voice 1: {UNTRUSTED}" in transcript
    assert page.url == original_url
    assert page.locator("#summary-text").input_value() == "Unsaved summary changes."
    assert page.get_by_role("button", name="Save edits").is_enabled()
    assert page.locator("#transcript-error").is_hidden()


def test_regeneration_requires_confirmation_before_replacing_saved_edits(
    page: Page,
) -> None:
    page.locator("#language").select_option("it")
    select_sample(page)
    page.get_by_role("button", name="Create draft summary").click()
    page.get_by_text(
        "Draft summary ready. Review and edit it before sharing."
    ).wait_for()
    page.locator("#summary-text").fill("Human-edited summary.")
    page.get_by_role("button", name="Save edits").click()
    page.get_by_text("Summary edits saved.").wait_for()

    page.once("dialog", lambda dialog: dialog.dismiss())
    page.get_by_role("button", name="Regenerate summary").click()
    assert page.locator("#summary-text").input_value() == "Human-edited summary."

    page.once("dialog", lambda dialog: dialog.accept())
    page.get_by_role("button", name="Regenerate summary").click()
    page.wait_for_function(
        "expected => document.querySelector('#summary-text').value === expected",
        arg=MINUTES_INJECTION,
    )
    assert page.locator("#review-status").inner_text() == "DRAFT / UNREVIEWED"


def test_stale_minutes_are_read_only_until_regenerated(page: Page) -> None:
    stale = _minutes_snapshot("stale")
    stale.update(
        {
            "minutes": {
                "summary": "Stale summary.",
                "topics": [],
                "decisions": [],
                "action_items": [],
            },
            "transcript_revision": 2,
            "minutes_source_transcript_revision": 1,
            "stale": True,
        }
    )
    fresh = _minutes_snapshot("stale")
    fresh["minutes"] = {
        "summary": "Updated summary.",
        "topics": [],
        "decisions": [],
        "action_items": [],
    }
    fresh["minutes_revision"] = 2
    fresh["transcript_revision"] = 2
    fresh["minutes_source_transcript_revision"] = 2
    regenerated = {"started": False}

    def handle(route: Route) -> None:
        path = urlsplit(route.request.url).path
        method = route.request.method
        if path.endswith("/api/v1/jobs") and method == "POST":
            route.fulfill(status=202, json={"job_id": "stale"})
        elif path == "/api/v1/jobs/stale":
            route.fulfill(json={"job_id": "stale", "state": "succeeded", "error": None})
        elif path.endswith("/minutes/regenerate") and method == "POST":
            regenerated["started"] = True
            response = {**stale, "regenerating": True}
            route.fulfill(status=202, json=response)
        elif path.endswith("/minutes") and method == "POST":
            route.fulfill(status=202, json=stale)
        elif path.endswith("/minutes") and method == "GET":
            route.fulfill(json=fresh if regenerated["started"] else stale)
        else:
            route.fulfill(status=404, json={"detail": "Unexpected test request."})

    page.route("**/api/v1/jobs**", handle)
    page.locator("#language").select_option("it")
    select_sample(page)
    page.get_by_role("button", name="Create draft summary").click()
    page.get_by_text(
        "The source transcript changed. Regenerate the summary before review."
    ).wait_for()

    assert page.locator("#summary-text").input_value() == "Stale summary."
    assert page.locator("#summary-text").is_disabled()
    assert page.get_by_role("button", name="Mark as reviewed").is_disabled()
    assert not page.get_by_role("button", name="Download JSON").is_disabled()

    page.once("dialog", lambda dialog: dialog.accept())
    page.get_by_role("button", name="Regenerate summary").click()
    page.wait_for_function(
        "expected => document.querySelector('#summary-text').value === expected",
        arg="Updated summary.",
    )
    assert page.locator("#review-status").inner_text() == "DRAFT / UNREVIEWED"


@pytest.mark.parametrize("confirm", [False, True])
def test_capacity_confirmation_precedes_replacement(page: Page, confirm: bool) -> None:
    uploads: list[str] = []

    def handle(route: Route) -> None:
        path = route.request.url.split("?", 1)[0]
        if route.request.method == "POST" and path.endswith("/minutes"):
            route.fulfill(status=202, json=_minutes_snapshot("new"))
        elif route.request.method == "POST":
            uploads.append(route.request.post_data or "")
            if len(uploads) == 1:
                route.fulfill(status=409, json={"oldest_job_id": "oldest"})
            else:
                route.fulfill(status=202, json={"job_id": "new"})
        else:
            route.fulfill(json={"job_id": "new", "state": "succeeded", "error": None})

    page.route("**/api/v1/jobs**", handle)
    page.on("dialog", lambda dialog: dialog.accept() if confirm else dialog.dismiss())
    select_sample(page)
    page.get_by_role("button", name="Create draft summary").click()
    if confirm:
        page.get_by_text(
            "Draft summary ready. Review and edit it before sharing."
        ).wait_for()
        assert len(uploads) == 2
        assert 'name="replace_oldest_job_id"' in uploads[1]
        assert "oldest" in uploads[1]
    else:
        page.get_by_text("Upload cancelled; the oldest result was kept.").wait_for()
        assert len(uploads) == 1


def test_cancel_running_transcription(page: Page) -> None:
    cancelled = False

    def handle(route: Route) -> None:
        nonlocal cancelled
        path = route.request.url.split("?", 1)[0]
        if path.endswith("/cancel"):
            cancelled = True
            route.fulfill(json={"job_id": "job", "state": "cancelled", "error": None})
        elif route.request.method == "POST":
            route.fulfill(status=202, json={"job_id": "job"})
        else:
            route.fulfill(
                json={
                    "job_id": "job",
                    "state": "cancelled" if cancelled else "running",
                    "error": None,
                }
            )

    page.route("**/api/v1/jobs**", handle)
    select_sample(page)
    page.get_by_role("button", name="Create draft summary").click()
    page.get_by_text("Transcribing and analyzing speakers...").wait_for()
    page.get_by_role("button", name="Cancel transcription").click()
    page.get_by_text("Transcription cancelled.").wait_for()
    assert cancelled
    assert page.locator("#minutes").is_hidden()


def test_cancel_running_minutes_generation(page: Page) -> None:
    def handle(route: Route) -> None:
        path = route.request.url.split("?", 1)[0]
        if path.endswith("/minutes/cancel"):
            route.fulfill(json=_minutes_snapshot("job", state="cancelled"))
        elif path.endswith("/minutes") and route.request.method == "POST":
            route.fulfill(status=202, json=_minutes_snapshot("job", state="running"))
        elif path.endswith("/minutes"):
            route.fulfill(json=_minutes_snapshot("job", state="running"))
        elif route.request.method == "POST":
            route.fulfill(status=202, json={"job_id": "job"})
        else:
            route.fulfill(json={"job_id": "job", "state": "succeeded", "error": None})

    page.route("**/api/v1/jobs**", handle)
    select_sample(page)
    page.get_by_role("button", name="Create draft summary").click()
    page.get_by_role("button", name="Cancel summary generation").wait_for()
    page.get_by_role("button", name="Cancel summary generation").click()
    page.get_by_text("Summary generation cancelled.").wait_for()
    assert page.get_by_role("button", name="Retry summary generation").is_visible()


def test_real_api_cancels_running_fake_transcription(page: Page) -> None:
    select_sample(page)
    page.get_by_role("button", name="Create draft summary").click()
    page.get_by_text("Transcribing and analyzing speakers...").wait_for()
    page.get_by_role("button", name="Cancel transcription").click()
    page.get_by_text("Transcription cancelled.").wait_for()
    assert page.get_by_role("button", name="Create draft summary").is_enabled()
    assert page.locator("#minutes").is_hidden()


def test_cancel_error_remains_visible_and_cancel_can_be_retried(page: Page) -> None:
    attempts = 0

    def handle(route: Route) -> None:
        nonlocal attempts
        path = route.request.url.split("?", 1)[0]
        if path.endswith("/cancel"):
            attempts += 1
            if attempts == 1:
                route.fulfill(status=503, json={"detail": "Cannot stop worker."})
            else:
                route.fulfill(
                    json={"job_id": "job", "state": "cancelled", "error": None}
                )
        elif route.request.method == "POST":
            route.fulfill(status=202, json={"job_id": "job"})
        else:
            route.fulfill(json={"job_id": "job", "state": "running", "error": None})

    page.route("**/api/v1/jobs**", handle)
    select_sample(page)
    page.get_by_role("button", name="Create draft summary").click()
    page.get_by_text("Transcribing and analyzing speakers...").wait_for()
    page.get_by_role("button", name="Cancel transcription").click()
    page.get_by_role("alert").wait_for()
    page.wait_for_timeout(1200)
    assert "Cannot stop worker." in page.get_by_role("alert").inner_text()
    page.get_by_role("button", name="Cancel transcription").click()
    page.get_by_text("Transcription cancelled.").wait_for()
    assert attempts == 2


def test_job_failure_is_visible_as_text(page: Page) -> None:
    def handle(route: Route) -> None:
        if route.request.method == "POST":
            route.fulfill(status=202, json={"job_id": "job"})
        else:
            route.fulfill(
                json={
                    "job_id": "job",
                    "state": "failed",
                    "error": {"kind": "RuntimeError", "message": UNTRUSTED},
                }
            )

    page.route("**/api/v1/jobs**", handle)
    select_sample(page)
    page.get_by_role("button", name="Create draft summary").click()
    page.get_by_role("alert").wait_for()
    assert page.get_by_role("alert").inner_text() == UNTRUSTED
    assert page.locator("#error img").count() == 0
