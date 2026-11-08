FROM ghcr.io/astral-sh/uv:0.11.6@sha256:b1e699368d24c57cda93c338a57a8c5a119009ba809305cc8e86986d4a006754 AS uv
FROM python:3.12.3-slim-bookworm@sha256:afc139a0a640942491ec481ad8dda10f2c5b753f5c969393b12480155fe15a63

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_MANAGED_PYTHON=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group cuda --no-install-project
COPY app ./app

ENV PATH="/app/.venv/bin:${PATH}" \
    LD_LIBRARY_PATH="/app/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:/app/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/app/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"
USER 65532:65532
EXPOSE 8000
CMD ["python", "-m", "app.container_start"]
