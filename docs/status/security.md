# Security overview

## Scope and status

This document describes the current local-development configuration; it is not
an independent security audit or production-readiness assessment. MeetScribe is
a single-user local application, not a multi-user service. Use synthetic or
explicitly authorized, non-sensitive recordings. The current controls do not
make it suitable for deployment on a company VM or an untrusted network without
additional design and review.

## Network and model provisioning

Ollama intentionally gets external network access **during model provisioning**
so it can download the pinned Qwen3 model from the Ollama registry. The
`compose.ollama-provision.yaml` override temporarily attaches the Ollama
container to Compose's normal `default` network. The model files are stored in
the gitignored local `ollama/` directory. The provisioning stack is stopped
after the download; do not use the provisioning override for normal runtime.

In normal runtime:

- The MeetScribe app is published on host loopback only:
  `127.0.0.1:${MEETSCRIBE_PORT:-8000}:8000`. This is intended for a browser on
  the same computer, not for LAN access.
- The app joins both the normal Compose network and the private `inference`
  network. Ollama joins only `inference`, which is marked `internal: true`.
- Ollama has no published host port. The app reaches it over Docker's private
  network at the fixed endpoint `http://ollama:11434`; clients cannot supply a
  remote model-server URL.
- `OLLAMA_NO_CLOUD=1` disables Ollama cloud features. The app verifies the
  installed model digest before generation. Ollama image/model pins are
  recorded in `app/model_manifests/summary_model.toml`.

The app remains attached to Compose's normal network in this standard runtime,
so its container may have outbound connectivity depending on Docker and host
network policy. The local model adapters use local model paths and the fixed
internal Ollama endpoint, but this Compose topology is **not a blanket egress
firewall for the app**. A private Docker network by itself is not proof that
all traffic is blocked. The offline integration-test override additionally
removes the app's normal network and published port to test an isolated run.
For a strict no-egress deployment, enforce egress restrictions at the host or
network layer and verify them in that deployment.

## Data flow and handling

1. The browser uploads audio to the local app through the loopback-published
   port.
2. The app runs ASR and speaker attribution locally using model files mounted
   read-only into the app container.
3. For minutes generation, the app sends transcript text and anonymous speaker
   labels to the local Ollama service over `inference`; it does not send audio
   to Ollama.
4. The app validates the generated draft before making it available for
   editing or export.

Uploads are limited to 100 MB and five minutes, streamed to a temporary app
directory, validated, and removed after processing, cancellation or failure.
Transcripts and minutes are held in memory temporarily; they are not restored
after restart. Persistent meetings, saved audio, accounts and the planned
retention/deletion workflow are not implemented. A downloaded transcript or
export remains under the user's browser/device control. JSON export v1 contains
the cited source transcript segments, so treat it as transcript data even
though the front page does not display transcript text.

## Implemented safeguards

- Job routes require a localhost host name; state-changing job requests require
  a matching same-origin `Origin`. These checks reduce accidental remote and
  cross-site access, but they are **not authentication**.
- The browser uses a restrictive Content Security Policy, local assets,
  `Cache-Control: no-store` responses and `X-Content-Type-Options: nosniff`.
  Model-generated text is rendered as text, not executable HTML.
- Upload parsing bounds request size, multipart fields and duration, validates
  the audio, and uses a time-limited validation worker.
- Container images, Python dependencies and model artifacts are pinned.
  ASR/diarization files are SHA-256 verified; the Ollama client checks the
  expected model digest before generation.
- ASR, diarization and Ollama use a shared inference gate to serialize GPU
  work. The app requests Ollama to unload after generation and verifies
  release before allowing the next GPU job. This primarily protects resource
  availability; it is not a substitute for OS or container isolation.
- Transcript content is treated as untrusted input in the minutes prompt, and
  generated minutes are schema/evidence checked. These checks do not prove
  factual accuracy or eliminate prompt injection or hallucinations.

## Important limitations before broader deployment

- There is no login, user identity, authorization or tenant isolation. Any
  process/user able to reach the local app can use its API. Do not bind the app
  to `0.0.0.0`, publish it through a reverse proxy, or expose it to a LAN as-is.
- The local HTTP configuration does not provide TLS. Loopback binding limits
  network reachability but does not protect against a compromised host or
  other local processes.
- The application has not been configured as a fully egress-blocked service.
  Enforce and test host-level egress policy if required.
- Docker, the host OS, GPU driver, image registry and model publisher remain
  part of the trusted computing base. Digest pinning prevents unexpected
  artifact changes; it does not replace security updates or vulnerability
  review.
- File validation and size/time limits reduce exposure to malformed or
  resource-exhausting uploads but cannot guarantee that native media decoders
  are free of vulnerabilities.
- No application-level encryption-at-rest, persistent-storage protection,
  backup policy, security event monitoring or incident-response process is
  provided. Browser downloads are outside application deletion control.

Before company or multi-user deployment, separately design and review
authentication/authorization, TLS, network egress controls, secret handling,
storage encryption and permissions, retention/deletion, backups, audit
logging, resource quotas and dependency/model update procedures. Do not use
real company recordings until that review and the required approvals are in
place.
