# AGENTS.md — repo orientation

Read this first. It is the condensed mental model for working on this repo;
the README is the user-facing version of the same material.

## What this repo is

Serve configs and image builds that run **Qwen 3.8 27B** on a single GPU with
**Claude Code** working end to end against it (Anthropic `/v1/messages` API,
tool calls, WebSearch, and for one variant long-video ingest). The repo owns
no model code: all behavior deltas live as pinned commits in three forks
(vllm, litellm, transformers under github.com/dkrisman), every one of which is
an open upstream PR or a fix branch attached to an upstream issue. The design
goal is that the forks shrink to nothing as PRs merge and this repo collapses
into compose files pulling stock images.

## Repo map

- `compose.<variant>.yaml` — one file per variant, mutually exclusive (one
  GPU). The header comment of each file is the authoritative record of its
  measured memory ledger, why each flag exists, and what features were traded
  for context ("buy-backs"). Read the header before changing any flag.
- `configs/litellm-*.yaml` — one LiteLLM proxy config per variant: master-key
  auth, `claude-*` wildcard route to the vLLM worker, WebSearch callback.
- `docker/pins.env` — the single source of truth for what is deployed: the
  `bq38-N` pin name plus the fork refs and base images. CI rebuilds and
  publishes images on every change to this file.
- `docker/vllm.Dockerfile`, `docker/litellm.Dockerfile` — pure-Python overlay
  builds on pinned official images; both end in baked smoke tests that fail
  the build if a fork feature is missing.
- `scripts/smoke-test.sh` — model list, chat, claude-* route, websearch.
- `scripts/claude-code.sh` — launches Claude Code against the local proxy.
- `.env.example` — every secret and machine-local path; nothing secret is
  tracked.

## The pin system

`bq38-N` tags on the forks are **immutable**: a pin is cut by tagging all
three forks, updating `docker/pins.env`, and updating the `image:` tags in
the compose files. Never retag or force-move a published pin; cut `bq38-N+1`.
The images `ghcr.io/dkrisman/bq38-{vllm,litellm}:<pin>` are built by
`.github/workflows/build-images.yml`; the compose `build:` blocks reproduce
them locally from nothing but this repo.

## Rules that are easy to violate

- **No monkeypatches.** No mounted files over container internals, no
  sitecustomize shims, no checkpoint edits. A behavior change is a fork
  commit with an upstream PR, or it does not ship.
- **Numbers are measured, not estimated.** Decode rates, KV pool sizes, and
  VRAM ledgers in the README and compose headers come from booting the exact
  config. If you change a memory-relevant flag, re-measure and update the
  header and the README table together. Computed VRAM floors (the `‡` rows)
  are labeled as such.
- **Two hard ceilings on this model** (Gated DeltaNet architecture):
  `--max-num-batched-tokens` above 16384 boots fine and then OOM-kills the
  engine on the first cold ~100K prefill; and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is required or cold big
  prefills OOM on fragmentation. Do not remove either without re-testing a
  cold 100K+ prefill.
- **Small-card tiers are sized by emulation**: `--gpu-memory-utilization`
  capped to the target card's exact budget on the dev card (96 GB RTX PRO
  6000). Keep that caveat attached to any number quoted for a 5090/3090.
- **README upstream table tracks reality.** When an upstream PR merges or
  closes, update its row and, at the next pin, drop the delta from the fork.

## Common tasks

- **Bump a pin**: tag the forks `bq38-N+1` → edit `docker/pins.env` (pin name
  + refs) → CI publishes → update `image:` tags in all compose files and any
  pin references in the README → boot the FP8 variant and run
  `scripts/smoke-test.sh`.
- **Add a tier**: pick the target VRAM budget, cap `--gpu-memory-utilization`
  to it, boot with a generous context, read the engine's memory ledger from
  the boot log, then maximize `--max-model-len` inside the pool. Document the
  ledger and the buy-back trades in the compose header; add the README rows.
- **Verify a change**: `docker compose -f <variant> up -d`, wait for
  `Application startup complete` in `docker logs -f bq38-vllm`, then
  `./scripts/smoke-test.sh`. For anything touching video, also run one
  `file://` video request against the direct vLLM port.

## Context that saves re-derivation

- KV cost is ~40 KiB/token even at fp8 KV (16 full-attention layers × 4 KV
  heads × head_dim 256, plus GDN page padding): 1 GiB of pool ≈ 26K tokens.
- The MTP speculative head costs ~0.79 GiB (~20K tokens of context) and pays
  ~1.5–2.2× decode; the micro tier drops it, the small tier keeps it.
- The video variant runs 500K context, not 1M: vLLM reserves ~26 GiB for the
  worst-case video at profiling time.
- The NVFP4 checkpoint (Unsloth Dynamic) loads 21.97 GiB of weights and
  cannot fit a 24 GB card; the micro tier uses a W4A16 AWQ checkpoint on
  `awq_marlin` (SM80+) instead.
