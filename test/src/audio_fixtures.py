"""Assemble synthetic test audio from versioned source recordings."""

import wave
from pathlib import Path

import av

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "test/data/smoke"
ENGLISH_CHECKPOINT = ROOT / "test/data/checkpoint/english/meeting-en.wav"
SAMPLE_RATE_HZ = 16_000
BYTES_PER_SAMPLE = 2
MILLISECONDS_PER_SECOND = 1000
SILENCE_MS = 1000
ESPEAK_INTERVALS_MS = ((1850, 6630), (13500, 19100))
VOICE_INTERVALS_MS = ((0, 3531), (4531, 9311), (10311, 12583), (13583, 19183))


def _mono_pcm(path: Path) -> bytes:
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE_HZ)
    pcm = bytearray()
    with av.open(str(path), mode="r") as container:
        for frame in container.decode(audio=0):
            for output in resampler.resample(frame):
                pcm.extend(output.to_ndarray().tobytes())
        for output in resampler.resample(None):
            pcm.extend(output.to_ndarray().tobytes())
    return bytes(pcm)


def _slice_pcm(pcm: bytes, start_ms: int, end_ms: int) -> bytes:
    start = start_ms * SAMPLE_RATE_HZ // MILLISECONDS_PER_SECOND * BYTES_PER_SAMPLE
    end = end_ms * SAMPLE_RATE_HZ // MILLISECONDS_PER_SECOND * BYTES_PER_SAMPLE
    if start < 0 or start >= end or end > len(pcm):
        raise ValueError("Synthetic voice source interval is invalid.")
    return pcm[start:end]


def _write_wav(
    path: Path, clips: tuple[bytes, ...], silence_ms: int
) -> list[tuple[int, int]]:
    silence = b"\x00" * (
        silence_ms * SAMPLE_RATE_HZ // MILLISECONDS_PER_SECOND * BYTES_PER_SAMPLE
    )
    frames_written = 0
    intervals = []
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(BYTES_PER_SAMPLE)
        audio.setframerate(SAMPLE_RATE_HZ)
        for clip in clips:
            if not clip:
                raise ValueError("Synthetic voice source audio is empty.")
            start_ms = frames_written * MILLISECONDS_PER_SECOND // SAMPLE_RATE_HZ
            audio.writeframes(clip)
            frames_written += len(clip) // BYTES_PER_SAMPLE
            intervals.append(
                (start_ms, frames_written * MILLISECONDS_PER_SECOND // SAMPLE_RATE_HZ)
            )
            audio.writeframes(silence)
            frames_written += len(silence) // BYTES_PER_SAMPLE
    return intervals


def write_two_voices(path: Path) -> list[tuple[int, int]]:
    espeak = _mono_pcm(ENGLISH_CHECKPOINT)
    clips = (
        _mono_pcm(SMOKE / "synthetic-meeting.wav"),
        _slice_pcm(espeak, *ESPEAK_INTERVALS_MS[0]),
        _mono_pcm(SMOKE / "alternate-flite.wav"),
        _slice_pcm(espeak, *ESPEAK_INTERVALS_MS[1]),
    )
    return _write_wav(path, clips, SILENCE_MS)


def write_repeated_sentence(path: Path) -> None:
    flite = _mono_pcm(SMOKE / "synthetic-meeting.wav")
    _write_wav(path, (flite, flite), 0)
