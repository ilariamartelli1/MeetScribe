"""Helpers for evaluating synthetic ASR checkpoint transcripts."""

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jiwer

SCRIPT_LINE = re.compile(r"^\[\d+[–-]\d+ ms\] (.+)$")
NEGATIONS = {
    "en": frozenset({"no", "not", "never"}),
    "it": frozenset({"no", "non", "mai"}),
}


@dataclass(frozen=True)
class Evaluation:
    reference_words: int
    hits: int
    substitutions: int
    deletions: int
    insertions: int
    wer: float
    alignment: str
    error_counts: str
    negation_differences: tuple[str, ...]


def read_reference(script_path: Path, json_path: Path) -> str:
    lines = script_path.read_text(encoding="utf-8").splitlines()
    sentences = [
        match.group(1) for line in lines if (match := SCRIPT_LINE.fullmatch(line))
    ]
    if not sentences:
        raise ValueError(f"No timed reference sentences found in {script_path}")
    expected = json.loads(json_path.read_text(encoding="utf-8"))
    if sentences != [segment["sentence"] for segment in expected["segments"]]:
        raise ValueError(f"Script and structured reference differ: {script_path}")
    return " ".join(sentences)


def scoring_text(text: str, normalize: Callable[[str], str]) -> str:
    normalized = unicodedata.normalize("NFKC", normalize(text.casefold())).casefold()
    # Preserve signs, currency symbols and numeric separators when removing punctuation.
    characters = []
    for index, character in enumerate(normalized):
        if not unicodedata.category(character).startswith("P"):
            characters.append(character)
        elif character in "+-" or (
            character in ".,"
            and index > 0
            and index + 1 < len(normalized)
            and normalized[index - 1].isdigit()
            and normalized[index + 1].isdigit()
        ):
            characters.append(character)
        else:
            characters.append(" ")
    return " ".join("".join(characters).split())


def evaluate(
    reference: str, hypothesis: str, language: str, normalize: Callable[[str], str]
) -> Evaluation:
    if language not in NEGATIONS:
        raise ValueError(f"Unsupported language: {language}")
    ref = scoring_text(reference, normalize)
    hyp = scoring_text(hypothesis, normalize)
    if not ref or not hyp:
        raise ValueError(
            "Reference and transcript must not be empty after normalization."
        )
    result = jiwer.process_words(ref, hyp)
    critical = NEGATIONS[language]
    differences = []
    for chunk in result.alignments[0]:
        if chunk.type == "equal":
            continue
        ref_negations = Counter(
            word
            for word in result.references[0][chunk.ref_start_idx : chunk.ref_end_idx]
            if word in critical
        )
        hyp_negations = Counter(
            word
            for word in result.hypotheses[0][chunk.hyp_start_idx : chunk.hyp_end_idx]
            if word in critical
        )
        if ref_negations != hyp_negations:
            differences.append(
                f"{chunk.type}: reference={dict(ref_negations)}, hypothesis={dict(hyp_negations)}"
            )
    return Evaluation(
        reference_words=len(result.references[0]),
        hits=result.hits,
        substitutions=result.substitutions,
        deletions=result.deletions,
        insertions=result.insertions,
        wer=result.wer,
        alignment=jiwer.visualize_alignment(result),
        error_counts=jiwer.visualize_error_counts(result),
        negation_differences=tuple(differences),
    )


def quality_failures(result: Evaluation, max_wer: float) -> tuple[str, ...]:
    if not 0 <= max_wer <= 1:
        raise ValueError("WER threshold must be between 0 and 1.")
    failures = []
    if result.wer > max_wer:
        failures.append(f"WER {result.wer:.2%} exceeds {max_wer:.2%}")
    if result.negation_differences:
        failures.extend(result.negation_differences)
    return tuple(failures)
