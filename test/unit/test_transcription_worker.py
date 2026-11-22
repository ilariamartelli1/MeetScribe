"""Verify provider-independent jobs and model release."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.core.asr_contract import (
    Language,
    ModelProvenance,
    Transcript,
    TranscriptSegment,
)
from app.core.diarization_contract import (
    Diarization,
    DiarizationProvenance,
    SpeakerTurn,
)
from app.core.transcription_worker import SpeakerAttributedJob, TranscriptionJob
from app.core.transcription_jobs import TranscriptionCoordinator
from test.unit.test_transcription_jobs import wait_for


def _transcript() -> Transcript:
    return Transcript(
        "en",
        1000,
        (TranscriptSegment(0, 100, "No"),),
        ModelProvenance("fake", "1", "model", "revision"),
    )


@pytest.mark.parametrize("fails", [False, True])
def test_model_is_unloaded_after_transcription(fails: bool) -> None:
    transcriber = Mock()
    transcriber.transcribe.return_value = _transcript()
    if fails:
        transcriber.transcribe.side_effect = RuntimeError("model failed")
    factory = Mock(return_value=transcriber)
    job = TranscriptionJob(factory)
    if fails:
        with pytest.raises(RuntimeError, match="model failed"):
            job(Path("/sample.wav"), "en")
    else:
        assert job(Path("/sample.wav"), "en") == _transcript()
    factory.assert_called_once_with()
    transcriber.close.assert_called_once_with()


@pytest.mark.parametrize("fails", [False, True])
def test_speaker_job_unloads_asr_before_diarization_and_releases_diarizer(
    fails: bool,
) -> None:
    transcriber = Mock()
    transcriber.transcribe.return_value = _transcript()
    diarizer = Mock()
    diarizer.diarize.return_value = Diarization(
        1000,
        (SpeakerTurn(0, 100, "speaker"),),
        (),
        DiarizationProvenance("fake", "1", "model", "revision", "cpu"),
    )

    def create_diarizer() -> Mock:
        transcriber.close.assert_called_once_with()
        return diarizer

    if fails:
        diarizer.diarize.side_effect = RuntimeError("diarization failed")
    job = SpeakerAttributedJob(lambda: transcriber, create_diarizer)
    if fails:
        with pytest.raises(RuntimeError, match="diarization failed"):
            job(Path("/sample.wav"), "en")
    else:
        result = job(Path("/sample.wav"), "en")
        assert result.segments == _transcript().segments
        assert result.attributions[0].speaker == "Voice 1"
    diarizer.close.assert_called_once_with()


def test_speaker_job_does_not_load_diarizer_when_asr_fails() -> None:
    transcriber = Mock()
    transcriber.transcribe.side_effect = RuntimeError("asr failed")
    diarizer_factory = Mock()
    with pytest.raises(RuntimeError, match="asr failed"):
        SpeakerAttributedJob(lambda: transcriber, diarizer_factory)(
            Path("/sample.wav"), "en"
        )
    transcriber.close.assert_called_once_with()
    diarizer_factory.assert_not_called()


@dataclass(frozen=True)
class FakeTranscriberFactory:
    marker: Path

    def __call__(self) -> "FakeTranscriber":
        return FakeTranscriber(self.marker)


@dataclass
class FakeTranscriber:
    marker: Path

    def transcribe(self, audio_path: Path, language: Language) -> Transcript:
        return _transcript()

    def close(self) -> None:
        self.marker.write_text("asr released", encoding="utf-8")


@dataclass(frozen=True)
class FakeDiarizerFactory:
    marker: Path

    def __call__(self) -> "FakeDiarizer":
        if self.marker.read_text(encoding="utf-8") != "asr released":
            raise RuntimeError("ASR was not released before diarization")
        return FakeDiarizer(self.marker)


@dataclass
class FakeDiarizer:
    marker: Path

    def diarize(self, audio_path: Path) -> Diarization:
        return Diarization(
            1000,
            (SpeakerTurn(0, 100, "speaker"),),
            (),
            DiarizationProvenance("fake", "1", "model", "revision", "cpu"),
        )

    def close(self) -> None:
        self.marker.write_text("both released", encoding="utf-8")


def test_injected_factories_run_in_spawned_worker_and_release_models(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "released.txt"
    coordinator = TranscriptionCoordinator(
        SpeakerAttributedJob(
            FakeTranscriberFactory(marker), FakeDiarizerFactory(marker)
        )
    )
    try:
        job_id = coordinator.submit(tmp_path / "audio.wav")
        wait_for(lambda: coordinator.get(job_id).state != "running")
        snapshot = coordinator.get(job_id)
        assert snapshot.state == "succeeded", snapshot.error
        assert marker.read_text(encoding="utf-8") == "both released"
        assert snapshot.result is not None
        assert snapshot.result.segments == _transcript().segments
    finally:
        coordinator.shutdown()
