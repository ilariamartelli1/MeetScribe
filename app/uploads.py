import logging
import multiprocessing
import os
import re
import struct
import tempfile
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import BinaryIO

import av
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header

from app.asr_contract import Language

MAX_AUDIO_BYTES = 100_000_000
MAX_REQUEST_BYTES = MAX_AUDIO_BYTES + 64 * 1024
MAX_DURATION_SECONDS = 300
MAX_SAMPLE_RATE = 48000
MAX_HEADER_BYTES = 8192
MAX_FIELD_BYTES = 1024
DECODE_TIMEOUT_SECONDS = 30
logger = logging.getLogger(__name__)


class UploadError(ValueError):
    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


class ValidationWorkerStuck(RuntimeError):
    def __init__(self, process: BaseProcess) -> None:
        super().__init__("The audio validation worker did not stop.")
        self.process = process


class UploadParser:
    def __init__(self, content_type: str, directory: Path) -> None:
        if len(content_type) > MAX_HEADER_BYTES:
            raise UploadError("Content-Type is too large.", 413)
        kind, options = parse_options_header(content_type)
        boundary = options.get(b"boundary", b"")
        if (
            kind.lower() != b"multipart/form-data"
            or not 1 <= len(boundary) <= 70
            or any(byte < 33 or byte > 126 for byte in boundary)
        ):
            raise UploadError("A valid multipart/form-data boundary is required.", 400)
        descriptor, name = tempfile.mkstemp(prefix="upload-", dir=directory)
        self.path = Path(name)
        self._file: BinaryIO = os.fdopen(descriptor, "wb")
        self._bytes = 0
        self._file_bytes = 0
        self._parts = 0
        self._header_bytes = 0
        self._header_name = bytearray()
        self._header_value = bytearray()
        self._headers: dict[bytes, bytes] = {}
        self._field = b""
        self._field_data = bytearray()
        self._fields: dict[bytes, bytes] = {}
        self._file_format = ""
        self._complete = False
        try:
            self._parser = MultipartParser(
                boundary,
                {
                    "on_part_begin": self._part_begin,
                    "on_header_field": self._header_field,
                    "on_header_value": self._header_value_data,
                    "on_header_end": self._header_end,
                    "on_headers_finished": self._headers_finished,
                    "on_part_data": self._part_data,
                    "on_part_end": self._part_end,
                    "on_end": self._end,
                },
            )
        except BaseException:
            self.discard()
            raise

    def _part_begin(self) -> None:
        self._parts += 1
        if self._parts > 3:
            raise UploadError("Too many multipart parts.", 400)
        self._headers = {}
        self._header_bytes = 0
        self._field = b""
        self._field_data = bytearray()

    def _header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_bytes += end - start
        if self._header_bytes > MAX_HEADER_BYTES:
            raise UploadError("Multipart headers are too large.", 413)
        self._header_name.extend(data[start:end])

    def _header_value_data(self, data: bytes, start: int, end: int) -> None:
        self._header_bytes += end - start
        if self._header_bytes > MAX_HEADER_BYTES:
            raise UploadError("Multipart headers are too large.", 413)
        self._header_value.extend(data[start:end])

    def _header_end(self) -> None:
        name = bytes(self._header_name).lower()
        if name in self._headers:
            raise UploadError("Duplicate multipart header.", 400)
        self._headers[name] = bytes(self._header_value)
        self._header_name.clear()
        self._header_value.clear()

    def _headers_finished(self) -> None:
        disposition, options = parse_options_header(
            self._headers.get(b"content-disposition")
        )
        field = options.get(b"name", b"")
        if (
            disposition.lower() != b"form-data"
            or field not in (b"file", b"language", b"replace_oldest_job_id")
            or field in self._fields
            or (field == b"file" and self._file_format)
            or b"content-transfer-encoding" in self._headers
        ):
            raise UploadError("Invalid or duplicate multipart part.", 400)
        filename = options.get(b"filename")
        if field == b"file":
            if filename is None or not filename:
                raise UploadError("An audio filename is required.", 400)
            suffix = filename.lower().rsplit(b".", 1)[-1]
            if suffix not in (b"wav", b"mp3"):
                raise UploadError("Only WAV and MP3 files are supported.")
            self._file_format = suffix.decode("ascii")
        elif filename is not None:
            raise UploadError("Unexpected multipart file.", 400)
        self._field = field

    def _part_data(self, data: bytes, start: int, end: int) -> None:
        part = data[start:end]
        if self._field == b"file":
            self._file_bytes += len(part)
            if self._file_bytes > MAX_AUDIO_BYTES:
                raise UploadError("Audio exceeds 100 MB.", 413)
            self._file.write(part)
        else:
            if len(self._field_data) + len(part) > MAX_FIELD_BYTES:
                raise UploadError("Multipart field is too large.", 413)
            self._field_data.extend(part)

    def _part_end(self) -> None:
        if self._field != b"file":
            self._fields[self._field] = bytes(self._field_data)

    def _end(self) -> None:
        self._complete = True

    def write(self, data: bytes) -> None:
        self._bytes += len(data)
        if self._bytes > MAX_REQUEST_BYTES:
            raise UploadError("Request is too large.", 413)
        try:
            self._parser.write(data)
        except MultipartParseError as exc:
            raise UploadError("Malformed multipart request.", 400) from exc

    def finish(self) -> tuple[Path, Language, str | None, str]:
        self._parser.finalize()
        self._file.close()
        if not self._complete:
            raise UploadError("Incomplete multipart request.", 400)
        if not self._file_format or not self._file_bytes:
            raise UploadError("A nonempty audio file is required.")
        language_bytes = self._fields.get(b"language", b"en")
        if language_bytes == b"en":
            language: Language = "en"
        elif language_bytes == b"it":
            language = "it"
        else:
            raise UploadError("Language must be 'en' or 'it'.")
        confirmation = self._fields.get(b"replace_oldest_job_id")
        if confirmation is not None and not re.fullmatch(b"[0-9a-f]{32}", confirmation):
            raise UploadError("Invalid result replacement confirmation.")
        return (
            self.path,
            language,
            confirmation.decode("ascii") if confirmation is not None else None,
            self._file_format,
        )

    def discard(self) -> None:
        self._file.close()
        self.path.unlink(missing_ok=True)


def _validate_in_child(path: Path, expected_format: str, sender: Connection) -> None:
    try:
        try:
            with av.open(str(path), mode="r") as container:
                if expected_format not in container.format.name.split(","):
                    raise UploadError("Audio content does not match its filename.")
                if expected_format == "wav":
                    with path.open("rb") as audio_file:
                        header = audio_file.read(12)
                    if (
                        len(header) != 12
                        or header[:4] != b"RIFF"
                        or header[8:] != b"WAVE"
                        or struct.unpack("<I", header[4:8])[0] + 8 > path.stat().st_size
                    ):
                        raise UploadError("Audio is corrupt or unsupported.")
                if len(container.streams) != 1 or len(container.streams.audio) != 1:
                    raise UploadError("Exactly one audio stream is supported.")
                stream = container.streams.audio[0]
                codec = stream.codec_context.name
                if (expected_format == "wav" and not codec.startswith("pcm_")) or (
                    expected_format == "mp3" and codec not in ("mp3", "mp3float")
                ):
                    raise UploadError("Unsupported audio codec.")
                stream.codec_context.thread_count = 1
                samples = 0
                sample_rate = 0
                for frame in container.decode(stream):
                    if len(frame.layout.channels) != 1:
                        raise UploadError("Only mono audio is supported.")
                    if not 8000 <= frame.sample_rate <= MAX_SAMPLE_RATE:
                        raise UploadError("Audio sample rate must be 8-48 kHz.")
                    if sample_rate and sample_rate != frame.sample_rate:
                        raise UploadError("Changing audio sample rate is unsupported.")
                    sample_rate = frame.sample_rate
                    samples += frame.samples
                    if samples > MAX_DURATION_SECONDS * sample_rate:
                        raise UploadError("Audio exceeds five minutes.")
                if not samples:
                    raise UploadError("Audio contains no decodable samples.")
            sender.send(None)
        except UploadError as exc:
            sender.send((exc.status_code, str(exc)))
        except av.FFmpegError as exc:
            if isinstance(exc, OSError):
                logger.exception("Could not read temporary audio.")
                sender.send((503, "Could not read temporary audio."))
            else:
                sender.send((422, "Audio is corrupt or unsupported."))
        except OSError:
            logger.exception("Could not read temporary audio.")
            sender.send((503, "Could not read temporary audio."))
        except ValueError:
            sender.send((422, "Audio is corrupt or unsupported."))
    finally:
        sender.close()


def validate_audio(path: Path, expected_format: str) -> None:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_validate_in_child, args=(path, expected_format, sender)
    )
    try:
        process.start()
        sender.close()
        process.join(DECODE_TIMEOUT_SECONDS)
        if process.is_alive():
            try:
                stop_validation_worker(process)
            except (OSError, RuntimeError) as exc:
                raise ValidationWorkerStuck(process) from exc
            raise UploadError("Audio decoding timed out.")
        if process.exitcode != 0 or not receiver.poll():
            logger.error(
                "Audio validation worker exited with code %s.", process.exitcode
            )
            raise RuntimeError("Audio validation worker failed.")
        try:
            error = receiver.recv()
        except EOFError as exc:
            raise RuntimeError("Audio validation worker sent no result.") from exc
        if error is not None:
            if (
                not isinstance(error, tuple)
                or len(error) != 2
                or not isinstance(error[0], int)
                or not isinstance(error[1], str)
            ):
                raise RuntimeError(
                    "Audio validation worker returned an invalid result."
                )
            raise UploadError(error[1], error[0])
    finally:
        sender.close()
        receiver.close()


def stop_validation_worker(process: BaseProcess) -> None:
    if process.is_alive():
        try:
            process.terminate()
        except ProcessLookupError:
            if process.is_alive():
                raise
        process.join(5)
    if process.is_alive():
        try:
            process.kill()
        except ProcessLookupError:
            if process.is_alive():
                raise
        process.join(5)
    if process.is_alive():
        raise RuntimeError("The audio validation worker did not stop.")
