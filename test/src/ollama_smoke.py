"""Run a short local Ollama generation from the application container."""

import json
import sys
import urllib.request
from typing import Any

BASE_URL = "http://ollama:11434"
MODEL = sys.argv[1]
EXPECTED_DIGEST = sys.argv[2]
EXPECTED_API_DIGEST = EXPECTED_DIGEST.removeprefix("sha256:")
CONTEXT_SIZE = int(sys.argv[3])
MAX_GENERATED_TOKENS = int(sys.argv[4])
TEMPERATURE = float(sys.argv[5])
KEEP_ALIVE = sys.argv[6]
THINK_VALUE = sys.argv[7].lower()
if THINK_VALUE not in {"true", "false"}:
    raise SystemExit("The smoke-test thinking option must be true or false.")
THINK = THINK_VALUE == "true"
RESPONSE_FORMAT = sys.argv[8]


def request_json(path: str, payload: dict[str, object] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        decoded: object = json.load(response)
    if not isinstance(decoded, dict):
        raise ValueError(f"Ollama returned a non-object response for {path}.")
    result: dict[str, Any] = {}
    for key, value in decoded.items():
        if not isinstance(key, str):
            raise ValueError(f"Ollama returned a non-string key for {path}.")
        result[key] = value
    return result


tags = request_json("/api/tags")
matches = [model for model in tags["models"] if model["name"] == MODEL]
if len(matches) != 1:
    raise SystemExit(f"Expected exactly one installed {MODEL} model.")
model = matches[0]
actual_digest = model["digest"].removeprefix("sha256:")
if actual_digest != EXPECTED_API_DIGEST:
    raise SystemExit(
        f"Unexpected {MODEL} digest: {model['digest']} (expected {EXPECTED_API_DIGEST})."
    )

details = request_json("/api/show", {"model": MODEL})["details"]
quantization = details["quantization_level"]
if quantization != "Q4_K_M":
    raise SystemExit(f"Unexpected model quantization: {quantization}.")

generation = request_json(
    "/api/generate",
    {
        "model": MODEL,
        "prompt": "Return a JSON object with the single field ready set to true.",
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "think": THINK,
        "format": RESPONSE_FORMAT,
        "options": {
            "num_ctx": CONTEXT_SIZE,
            "num_predict": MAX_GENERATED_TOKENS,
            "temperature": TEMPERATURE,
        },
    },
)
response_text = generation.get("response", "").strip()
if not generation.get("done") or not response_text:
    raise SystemExit("Ollama did not complete the smoke generation.")
try:
    parsed_response = json.loads(response_text)
except json.JSONDecodeError as exc:
    raise SystemExit("Ollama returned invalid JSON in the smoke generation.") from exc
if parsed_response != {"ready": True}:
    raise SystemExit("Ollama returned an unexpected JSON smoke result.")

print(
    json.dumps(
        {
            "ollama_version": request_json("/api/version").get("version"),
            "model": MODEL,
            "model_digest": actual_digest,
            "model_size_bytes": model["size"],
            "quantization": quantization,
            "context_size": CONTEXT_SIZE,
            "max_generated_tokens": MAX_GENERATED_TOKENS,
            "temperature": TEMPERATURE,
            "keep_alive": KEEP_ALIVE,
            "think": THINK,
            "response_format": RESPONSE_FORMAT,
            "prompt_tokens": generation.get("prompt_eval_count"),
            "generated_tokens": generation.get("eval_count"),
            "done_reason": generation.get("done_reason"),
            "response_characters": len(response_text),
            "thinking_characters": len(generation.get("thinking", "")),
            "model_load_ms": generation.get("load_duration", 0) // 1_000_000,
            "prompt_eval_ms": generation.get("prompt_eval_duration", 0) // 1_000_000,
            "generation_ms": generation.get("eval_duration", 0) // 1_000_000,
            "response_nonempty": bool(response_text),
        }
    )
)
