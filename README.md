# BlackwellQwen38

Run **Qwen 3.8 27B at 1M context** on a single Blackwell workstation GPU, with
**Claude Code** and its **WebSearch** tool working end to end against the local
model — three quantization variants plus a long-video variant, one
`docker compose` command each. Sized-down tiers bring the same stack to
32 GB (RTX 5090, 128K context) and 24 GB (RTX 3090/4090, 64K context) cards.

Everything is built from public sources at pinned refs: three patched forks
([dkrisman/vllm](https://github.com/dkrisman/vllm),
[dkrisman/litellm](https://github.com/dkrisman/litellm),
[dkrisman/transformers](https://github.com/dkrisman/transformers)) carrying
changes that are under review upstream (see
[Upstream PRs and issues](#upstream-prs-and-issues) — reviews and merges
welcome), plus the
[litellm-claude-code-websearch](https://github.com/dkrisman/litellm-claude-code-websearch)
plugin. No monkeypatches, no mounted patch files, no checkpoint edits.
Prebuilt images are published to ghcr by this repo's own CI (see
[How the images are built](#how-the-images-are-built)).

## Hardware

Developed and measured on an **NVIDIA RTX PRO 6000 Blackwell (96 GB)**. The
BF16 variant needs ~91 GB at 1M context; FP8 and NVFP4 fit with more headroom.
NVFP4 additionally **requires** a Blackwell-generation GPU. For smaller cards
use the sized-down tiers: `compose.nvfp4.small.yaml` (32 GB Blackwell) and
`compose.awq.micro.yaml` (24 GB, Ampere and newer) — see
[The small-card tiers](#the-small-card-tiers).

## The variants

| Variant | Min VRAM | Context | Decode | KV pool* | Notes |
|---|---|---|---|---|---|
| `compose.fp8.yaml` | ~70 GB‡ | 1M | **84 tok/s** | ~1.7M tok | **Recommended.** [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8): official quant, near-lossless, 1.48× BF16 decode |
| `compose.bf16.yaml` | 96 GB | 1M | 56.5 tok/s | ~1.05M tok | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B): quality reference, full precision |
| `compose.nvfp4.yaml` | ~65 GB‡ | 1M | **113 tok/s** | ~1.7M tok | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4): fastest (2× BF16); 92–97% accuracy retention per Unsloth, text-only |
| `compose.fp8.video.yaml` | 96 GB | 500K | 84 tok/s | ~1.66M tok | **Long-video understanding**: same FP8 checkpoint, the model card's 224K-video-token recipe as pure serve config (see below) |
| `compose.nvfp4.small.yaml` | 32 GB (RTX 5090) | 128K | 105 tok/s | ~145K tok | Same NVFP4 checkpoint sized for one 32 GB Blackwell card; keeps MTP |
| `compose.awq.micro.yaml` | 24 GB (RTX 3090/4090) | 64K | 73 tok/s† | ~78K tok | [philbert440/Qwen3.8-27B-W4A16-AWQ](https://huggingface.co/philbert440/Qwen3.8-27B-W4A16-AWQ) via `awq_marlin`, runs on Ampere+; text-only, no MTP |

All decode figures measured single-stream on the RTX PRO 6000; the 96 GB
variants use MTP k=3. †The micro figure is the dev card running the same
`awq_marlin` kernels a 3090 would — expect roughly half on a 3090 itself
(half the memory bandwidth). *KV pool as measured at each variant's shipped
budget: the 96 GB card for the four full variants (everything above the
Min-VRAM floor becomes pool), the 32 GB / 24 GB target-card budgets for
small/micro. ‡Computed 1M-context floors, not validated cards: weights +
non-torch + activation + ~39 GiB KV (1,010,000 tokens × 40.5 KiB/token
measured at fp8 KV). FP8: ~27 + ~2.4 + 39 ≈ 70 GB. NVFP4: 22.6 + 2.4 + 39
≈ 65 GB (weights and activation measured, small-tier chunking — the shipped
16384-token chunks need some extra activation headroom). Only the 96 GB
card is validated; an 80 GB H100/H200-class card should hold either, but
that's a different architecture and untested here. The three
quantization variants run 1M context (model card's `max_position_embeddings` lift — native extension,
not YaRN), MTP speculative decoding (k=3), prefix caching, FP8 KV cache, and
the `qwen3_coder` tool parser. The compose files carry inline comments for
every non-obvious flag, including two hard-won ceilings: `--max-num-batched-tokens`
16384 (larger chunk sizes boot fine, then the first cold ~100K prefill
OOM-kills the engine via the Gated DeltaNet prefill kernel's workspace) and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (without it, cold big
prefills OOM on fragmentation at high memory utilization).

## The small-card tiers

Both tiers were sized empirically on the 96 GB card by capping
`--gpu-memory-utilization` to the target card's exact budget (0.95 × its
VRAM), then maximizing context inside it. Context is the scarce resource:
this model's KV costs ~40 KiB/token even at fp8 KV (16 full-attention
layers × 4 KV heads × head_dim 256, plus GDN page padding), so one GiB of
budget buys only ~26K tokens. The compose headers carry the measured memory
ledgers and the buy-back trades.

- **`compose.nvfp4.small.yaml` — 32 GB (RTX 5090)**: the same NVFP4
  checkpoint at 131,072 context with MTP kept on (144,584-token KV pool,
  1.10× concurrency). 105 tok/s measured at this config; the 5090 is the
  same GB202 die as the dev card with near-identical bandwidth, so that
  number should transfer.
- **`compose.awq.micro.yaml` — 24 GB (RTX 3090/4090)**: the NVFP4
  checkpoint cannot fit here — Unsloth Dynamic keeps enough layers at
  FP8/BF16 that its weights alone load at 21.97 GiB — so this tier switches
  to a W4A16 AWQ quant (16.68 GiB) on the `awq_marlin` kernels, which run
  on SM80+ (Ampere and newer, no Blackwell requirement). 65,536 context;
  MTP and vision are off by default, and the compose header shows how to
  trade context to re-enable either. Not yet validated on a real Ampere
  card: if fp8 KV cache fails to boot there, drop `--kv-cache-dtype fp8`
  and halve `--max-model-len`.

Both files expose `GPU_MEMORY_UTILIZATION` in `.env` (default 0.95) for
desktops where the display compositor already holds a GiB.

## The long-video variant

The Qwen3.8 model card's "Long Video Understanding" recipe raises the video
processor's pixel budget from ~12K to ~224K video tokens — by telling you to
**edit `video_preprocessor_config.json` inside the checkpoint**, which isn't
reproducible and silently reverts on re-download. Stock vLLM can't express
the override from serve config either: a flat `--mm-processor-kwargs size`
leaks into the *image* budget (448K-token images, profiling grinds), and the
HF-scoped `videos_kwargs` form is honored at processing time but ignored by
memory profiling — both filed upstream as
[vllm#52834](https://github.com/vllm-project/vllm/issues/52834) and
[vllm#52835](https://github.com/vllm-project/vllm/issues/52835) (a processed
item bigger than the processor cache kills engine boot).

`compose.fp8.video.yaml` runs an image (pin tag `bq38-4`) carrying fixes for
both, plus a one-file overlay from
[dkrisman/transformers](https://github.com/dkrisman/transformers/tree/qwen3vl-video-max-pixels-per-frame)
adding the `max_pixels_per_frame` video kwarg (makes short clips cost tokens
proportional to duration: a 90 s clip ~53K instead of ~184K). The entire
recipe becomes one serve flag:

```
--mm-processor-kwargs '{"videos_kwargs": {"size": {"longest_edge": 469762048,
  "shortest_edge": 4096}, "max_pixels_per_frame": 611669}}'
```

Hour-scale videos ingest via `file://` URLs from the `VIDEO_DIR` mount on the
**direct vLLM port** (the Anthropic API surface has no video content type):

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen-3.8-fp8-27b-video-500k", "max_tokens": 300,
  "messages": [{"role": "user", "content": [
    {"type": "video_url", "video_url": {"url": "file:///videos/clip.mp4"}},
    {"type": "text", "text": "Describe what happens in this clip."}]}]}'
```

Context is 500K rather than 1M on this variant: vLLM reserves peak encoder
memory for one max-size video (~26 GiB), which the 1M KV floor cannot spare.
Audio tracks are discarded (Qwen 3.8 is vision-only). Measured on the same
hardware: a 1080p feature-length film is ~225K prompt tokens, first query
~2 min.

## Quickstart

```bash
git clone https://github.com/dkrisman/BlackwellQwen38.git && cd BlackwellQwen38
cp .env.example .env    # set HF cache dir, Brave API key file, master key
docker compose -f compose.fp8.yaml up -d --build
```

First boot downloads the checkpoint (~28 GB for FP8) and takes a few minutes
of engine init/compile; watch with `docker logs -f bq38-vllm` until
`Application startup complete`. Then:

```bash
./scripts/smoke-test.sh     # model list, chat, claude-* route, websearch
./scripts/claude-code.sh    # launch Claude Code against the local model
```

Switching variants (one GPU — they are mutually exclusive):

```bash
docker compose -f compose.fp8.yaml down
docker compose -f compose.nvfp4.yaml up -d --build
```

All variants share the same two images, published to ghcr by this repo's CI
(compose pulls them; `docker compose -f <variant> build` reproduces them
locally from the pinned fork refs if you can't pull).

## How Claude Code works against this stack

- **Routing**: LiteLLM exposes the Anthropic `/v1/messages` API natively; a
  `claude-*` wildcard route lands every Claude model name (`claude-sonnet-5`,
  `claude-haiku-4-5`, …) on the local vLLM worker. `scripts/claude-code.sh`
  just points `ANTHROPIC_BASE_URL` at the proxy.
- **Tool calls**: Anthropic `tool_use` ↔ OpenAI `tool_calls` translate through
  LiteLLM + vLLM's `qwen3_coder` parser.
- **Reasoning effort**: Claude Code sends `reasoning_effort` values the Qwen
  chat template rejects (`high`, `max`, `minimal`). The vLLM fork's
  `--supported-reasoning-efforts` flag (PR [#52739][vllm-52739]) remaps them
  to the nearest supported level instead of returning a 400.
- **WebSearch**: Claude Code's WebSearch tool expects the backend to run the
  search server-side, which a local model can't. The
  [litellm-claude-code-websearch](https://github.com/dkrisman/litellm-claude-code-websearch)
  plugin (baked into the LiteLLM image at build time) short-circuits those
  requests through the Brave Search API and returns the native
  `server_tool_use` → `web_search_tool_result` block sequence, so "Did 1
  search" and result links render exactly as they do against real Anthropic.
  Streaming those blocks relies on LiteLLM PR [#37318][ll-37318], carried in
  the fork.

## How the images are built

This repo's [build-images workflow](.github/workflows/build-images.yml)
publishes both images to ghcr on every change to
[`docker/pins.env`](docker/pins.env):

- `ghcr.io/dkrisman/bq38-vllm:<pin>` — a pinned official vLLM nightly with the
  fork's Python tree and the transformers Qwen3-VL patch overlaid
  ([`docker/vllm.Dockerfile`](docker/vllm.Dockerfile)). No kernel
  compilation: the fork deltas are frontend Python only. When a future pin's
  upstream base has no published nightly image, the Dockerfile can instead
  swap in that commit's wheel from vLLM's per-commit wheel index
  (`wheels.vllm.ai`).
- `ghcr.io/dkrisman/bq38-litellm:<pin>` — the LiteLLM fork overlaid on the
  nearest release image with the WebSearch plugin baked in
  ([`docker/litellm.Dockerfile`](docker/litellm.Dockerfile)).

Both Dockerfiles end in baked smoke tests (compiled extension present, fork
features importable, transformers kwarg applied). The compose files reference
the ghcr images and carry equivalent `build:` blocks, so
`docker compose -f <variant> build` reproduces the exact images locally from
nothing but this repo. The `bq38-N` tags on the forks are immutable pins; the
current pin set lives in `docker/pins.env`.

## Upstream PRs and issues

The forks exist only to carry these changes until they merge — if any of them
would help you, a review or a 👍 upstream accelerates that. Once merged, the
overlay builds collapse back into stock images.

**vLLM** ([fork](https://github.com/dkrisman/vllm), tag `bq38-3`):

| PR / issue | What it does | Used here |
|---|---|---|
| [#52739][vllm-52739] Map unsupported reasoning_effort to nearest supported level | Claude Code's `reasoning_effort` values stop 400ing on Qwen templates | ✅ every request with thinking |
| [#52759][vllm-52759] Surface TorchCodec video decode failures as client errors | Bad video inputs 400 instead of 500 | ✅ `fp8.video` variant |
| [#52834][vllm-52834] (issue) Modality-scoped `mm-processor-kwargs` | `videos_kwargs` overrides work without inflating the image budget — fix on branch [`feat/mm-kwargs-modality-scoped`](https://github.com/dkrisman/vllm/commits/feat/mm-kwargs-modality-scoped) | ✅ `fp8.video` variant |
| [#52835][vllm-52835] (issue) Oversized item kills engine boot | Items bigger than the processor cache are served uncached instead of raising — fix on branch [`fix/mm-cache-skip-oversized`](https://github.com/dkrisman/vllm/commits/fix/mm-cache-skip-oversized) | ✅ `fp8.video` variant |
| [#52754][vllm-52754] Make Qwen3-VL video cost duration-proportional | Closed as superseded: per review the knob belongs in the HF processor, now transformers PR [#48071][tf-48071] | ✅ `fp8.video` variant |

**transformers** ([fork](https://github.com/dkrisman/transformers), tag `bq38-3`):

| PR | What it does | Used here |
|---|---|---|
| [#48071][tf-48071] Opt-in per-frame pixel cap for the Qwen3-VL video processor | Video token cost scales with clip duration instead of every clip filling the whole budget; upstream now a boolean `cap_pixels_per_frame` applying the qwen-vl-utils formula (the pinned image still carries the numeric `max_pixels_per_frame` form) | ✅ `fp8.video` variant |

**LiteLLM** ([fork](https://github.com/dkrisman/litellm), tag `bq38-3`):

| PR | What it does | Used here |
|---|---|---|
| [#37318][ll-37318] Stream native server-tool blocks to clients | WebSearch results stream as real `server_tool_use`/`web_search_tool_result` blocks | ✅ every WebSearch call |
| [#36671][ll-36671] Count Anthropic native image content blocks (third-party) | Token counting stops crashing on image blocks in `/v1/messages` | ✅ image inputs |
| [#31332][ll-31332] Backfill response.completed output from output_item.done (third-party) | Responses-API bridge stops dropping streamed output | carried (Responses-API providers) |
| [#37287][ll-37287] Opt-in stable session id derived from conversation prefix | Reliable upstream prompt-cache hits for the ChatGPT provider | carried (no ChatGPT route in this stack) |
| [#37276][ll-37276] Keep structured output text.format in Responses API requests | Structured output survives the ChatGPT provider transform | carried (no ChatGPT route in this stack) |
| [#37351][ll-37351] Opt-in demotion of mid-turn system messages | Claude Code's mid-conversation system reminders stop 400ing on Qwen templates | ✅ every multi-turn session |

The fork also carries a fix for `map_system_message_pt` crashing on
content-block system messages (several equivalent PRs are already open
upstream, so it is not filed separately) and one more feature, found while
testing this repo and now under review as PR [#37351][ll-37351]: recent
Claude Code versions send mid-conversation `system`-role reminder messages,
which OpenAI accepts but Qwen's chat template rejects (`"System message must
be at the beginning."`). `LITELLM_DEMOTE_MIDTURN_SYSTEM` controls the
fix: the compose files default to `drop` (remove the reminders entirely,
which also keeps the conversation prefix stable for vLLM's prefix cache);
set `true` to instead rewrite them as user turns — the place Claude Code
historically put them. Upstream preserves mid-turn system rows deliberately
(PR [#34290](https://github.com/BerriAI/litellm/pull/34290)), so this stays
opt-in.

[vllm-52739]: https://github.com/vllm-project/vllm/pull/52739
[tf-48071]: https://github.com/huggingface/transformers/pull/48071
[vllm-i52834]: https://github.com/vllm-project/vllm/issues/52834
[vllm-i52835]: https://github.com/vllm-project/vllm/issues/52835
[vllm-52834]: https://github.com/vllm-project/vllm/issues/52834
[vllm-52835]: https://github.com/vllm-project/vllm/issues/52835
[vllm-52754]: https://github.com/vllm-project/vllm/pull/52754
[vllm-52759]: https://github.com/vllm-project/vllm/pull/52759
[ll-37318]: https://github.com/BerriAI/litellm/pull/37318
[ll-36671]: https://github.com/BerriAI/litellm/pull/36671
[ll-31332]: https://github.com/BerriAI/litellm/pull/31332
[ll-37287]: https://github.com/BerriAI/litellm/pull/37287
[ll-37276]: https://github.com/BerriAI/litellm/pull/37276
[ll-37351]: https://github.com/BerriAI/litellm/pull/37351

## Configuration

Secrets and machine-local paths live only in `.env` (gitignored; template in
`.env.example`):

| Variable | Purpose |
|---|---|
| `HF_CACHE_DIR` | host directory for HuggingFace checkpoint cache |
| `BRAVE_API_KEY_FILE` | file containing your [Brave Search API](https://brave.com/search/api/) key |
| `LITELLM_MASTER_KEY` | proxy auth key (any secret you choose) |
| `VLLM_PORT` / `LITELLM_PORT` | host ports (defaults 8000 / 4000, bound to 127.0.0.1) |
| `WEBSEARCH_DAILY_LIMIT` | Brave search daily cap (default 500; plugin degrades gracefully) |

The LiteLLM configs in `configs/` contain no secrets (the master key is read
from the environment), so they are tracked as-is.

## License

MIT
