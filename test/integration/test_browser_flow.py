"""Verify the browser flow with fake inference and controlled API responses.

Set MEETSCRIBE_BROWSER_TESTS=1 and install a Chromium browser to run these tests.
"""

import os
import shutil
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from playwright.sync_api import Browser, Page, Route, sync_playwright

import app.main as main
from app.asr_contract import Language, ModelProvenance, Transcript, TranscriptSegment

pytestmark = pytest.mark.skipif(
    os.environ.get("MEETSCRIBE_BROWSER_TESTS") != "1",
    reason="Set MEETSCRIBE_BROWSER_TESTS=1 to run Chromium browser tests.",
)

SAMPLE = Path(__file__).resolve().parents[1] / "data/synthetic-meeting.wav"
UNTRUSTED = '<img src=x onerror="window.injected=true"> no'


class FakeBrowserJob:
    def __init__(self, model_dir: Path, device: str) -> None:
        pass

    def __call__(self, audio_path: Path, language: Language) -> Transcript:
        if language == "en":
            time.sleep(30)
        return Transcript(
            language,
            1000,
            (TranscriptSegment(100, 900, UNTRUSTED),),
            ModelProvenance(
                "fake", "1", "model", "revision", "digest", "cpu", "int8", True, 5
            ),
        )


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setattr(main, "FasterWhisperJob", FakeBrowserJob)
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


def test_real_api_upload_edit_and_download(page: Page, tmp_path: Path) -> None:
    assert "synthetic or explicitly authorized" in page.locator(".notice").inner_text()
    assert "do not anonymize" in page.locator(".notice").inner_text()
    page.locator("#language").select_option("it")
    select_sample(page)
    page.get_by_role("button", name="Transcribe").click()
    page.get_by_text("Transcription complete. Review the draft below.").wait_for()
    text = page.locator("#segment-0")
    assert text.input_value() == UNTRUSTED
    assert page.locator("#segments img").count() == 0
    assert page.evaluate("window.injected") is None
    text.fill("Corrected Italian text")
    with page.expect_download() as downloaded:
        page.get_by_role("button", name="Download draft TXT").click()
    saved = tmp_path / "draft.txt"
    downloaded.value.save_as(saved)
    assert saved.read_text() == (
        "DRAFT / UNREVIEWED\n\n[00:00:00.100 - 00:00:00.900] Corrected Italian text\n"
    )
    assert page.evaluate("localStorage.length") == 0
    assert page.evaluate("sessionStorage.length") == 0
    assert page.evaluate("indexedDB.databases().then(items => items.length)") == 0
    page.reload()
    assert page.locator("#transcript").is_hidden()


@pytest.mark.parametrize("confirm", [False, True])
def test_capacity_confirmation_precedes_replacement(page: Page, confirm: bool) -> None:
    uploads: list[str] = []

    def handle(route: Route) -> None:
        if route.request.method == "POST":
            uploads.append(route.request.post_data or "")
            if len(uploads) == 1:
                route.fulfill(status=409, json={"oldest_job_id": "oldest"})
            else:
                route.fulfill(status=202, json={"job_id": "new"})
        else:
            route.fulfill(
                json={
                    "state": "succeeded",
                    "result": {
                        "segments": [{"start_ms": 0, "end_ms": 100, "text": "No"}]
                    },
                }
            )

    page.route("**/api/v1/jobs**", handle)
    page.on("dialog", lambda dialog: dialog.accept() if confirm else dialog.dismiss())
    select_sample(page)
    page.get_by_role("button", name="Transcribe").click()
    if confirm:
        page.get_by_text("Transcription complete. Review the draft below.").wait_for()
        assert len(uploads) == 2
        assert 'name="replace_oldest_job_id"' in uploads[1]
        assert "oldest" in uploads[1]
    else:
        page.get_by_text("Upload cancelled; the oldest result was kept.").wait_for()
        assert len(uploads) == 1


def test_cancel_running_job(page: Page) -> None:
    cancelled = False

    def handle(route: Route) -> None:
        nonlocal cancelled
        if route.request.url.endswith("/cancel"):
            cancelled = True
            route.fulfill(json={"state": "cancelled"})
        elif route.request.method == "POST":
            route.fulfill(status=202, json={"job_id": "job"})
        else:
            route.fulfill(json={"state": "cancelled" if cancelled else "running"})

    page.route("**/api/v1/jobs**", handle)
    select_sample(page)
    page.get_by_role("button", name="Transcribe").click()
    page.get_by_text("Transcribing...").wait_for()
    page.get_by_role("button", name="Cancel job").click()
    page.get_by_text("Job cancelled.").wait_for()
    assert cancelled
    assert page.locator("#transcript").is_hidden()


def test_real_api_cancels_running_fake_job(page: Page) -> None:
    select_sample(page)
    page.get_by_role("button", name="Transcribe").click()
    page.get_by_text("Transcribing...").wait_for()
    page.get_by_role("button", name="Cancel job").click()
    page.get_by_text("Job cancelled.").wait_for()
    assert page.get_by_role("button", name="Transcribe").is_enabled()
    assert page.locator("#transcript").is_hidden()


def test_cancel_error_remains_visible_and_cancel_can_be_retried(page: Page) -> None:
    attempts = 0

    def handle(route: Route) -> None:
        nonlocal attempts
        if route.request.url.endswith("/cancel"):
            attempts += 1
            if attempts == 1:
                route.fulfill(status=503, json={"detail": "Cannot stop worker."})
            else:
                route.fulfill(json={"state": "cancelled"})
        elif route.request.method == "POST":
            route.fulfill(status=202, json={"job_id": "job"})
        else:
            route.fulfill(json={"state": "running"})

    page.route("**/api/v1/jobs**", handle)
    select_sample(page)
    page.get_by_role("button", name="Transcribe").click()
    page.get_by_text("Transcribing...").wait_for()
    page.get_by_role("button", name="Cancel job").click()
    page.get_by_role("alert").wait_for()
    page.wait_for_timeout(1200)
    assert "Cannot stop worker." in page.get_by_role("alert").inner_text()
    page.get_by_role("button", name="Cancel job").click()
    page.get_by_text("Job cancelled.").wait_for()
    assert attempts == 2


def test_job_failure_is_visible_as_text(page: Page) -> None:
    def handle(route: Route) -> None:
        if route.request.method == "POST":
            route.fulfill(status=202, json={"job_id": "job"})
        else:
            route.fulfill(json={"state": "failed", "error": {"message": UNTRUSTED}})

    page.route("**/api/v1/jobs**", handle)
    select_sample(page)
    page.get_by_role("button", name="Transcribe").click()
    page.get_by_role("alert").wait_for()
    assert page.get_by_role("alert").inner_text() == UNTRUSTED
    assert page.locator("#error img").count() == 0
