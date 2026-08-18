# BlackwellQwen38

Run **Qwen 3.8 27B at 1M context** on a single Blackwell workstation GPU, with
**Claude Code** and its **WebSearch** tool working end to end against the local
model — three quantization variants, one `docker compose` command each.

Everything is built from public sources at pinned refs: two patched forks
([dkrisman/vllm](https://github.com/dkrisman/vllm),
[dkrisman/litellm](https://github.com/dkrisman/litellm)) carrying changes that
are under review upstream (see [Upstream PRs](#upstream-prs) — reviews and
merges welcome), plus the
[litellm-claude-code-websearch](https://github.com/dkrisman/litellm-claude-code-websearch)
plugin. No monkeypatches, no mounted patch files.

## Hardware

Developed and measured on an **NVIDIA RTX PRO 6000 Blackwell (96 GB)**. The
BF16 variant needs ~91 GB at 1M context; FP8 and NVFP4 fit with more headroom.
NVFP4 additionally **requires** a Blackwell-generation GPU. Smaller cards can
run FP8/NVFP4 by lowering `--max-model-len`.

## The three variants

| Variant | Checkpoint | Decode (MTP k=3) | KV pool | Notes |
|---|---|---|---|---|
| `compose.fp8.yaml` | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | **84 tok/s** | ~1.7M tok | **Recommended.** Official quant, near-lossless, 1.48× BF16 decode |
| `compose.bf16.yaml` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 56.5 tok/s | ~1.05M tok | Quality reference, full precision |
| `compose.nvfp4.yaml` | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | **113 tok/s** | ~1.7M tok | Fastest (2× BF16); 92–97% accuracy retention per Unsloth, text-only |

All decode figures measured single-stream on the RTX PRO 6000. All three run
1M context (model card's `max_position_embeddings` lift — native extension,
not YaRN), MTP speculative decoding (k=3), prefix caching, FP8 KV cache, and
the `qwen3_coder` tool parser. The compose files carry inline comments for
every non-obvious flag, including two hard-won ceilings: `--max-num-batched-tokens`
16384 (larger chunk sizes boot fine, then the first cold ~100K prefill
OOM-kills the engine via the Gated DeltaNet prefill kernel's workspace) and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (without it, cold big
prefills OOM on fragmentation at high memory utilization).

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

The two locally built images are shared across variants, so switching after
the first build is fast.

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

Both images are **pure-Python overlays onto pinned official images**, built by
compose directly from the public forks — reproducible from nothing but this
repo:

```yaml
build:
  context: https://github.com/dkrisman/vllm.git#bq38-2
  dockerfile_inline: |
    FROM vllm/vllm-openai:nightly-aa9903490c616dc6871e5acc62cec7bb1e5e9434
    COPY vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
```

The `bq38-2` tags pin fork commits whose upstream base matches the base image
exactly (`aa9903490` for vLLM; the fork delta is frontend Python only, so
compiled kernels and dependencies are untouched). The LiteLLM build does the
same against `ghcr.io/berriai/litellm:v1.98.0-rc.1` and additionally clones
the WebSearch plugin into the image.

## Upstream PRs

The forks exist only to carry these changes until they merge — if any of them
would help you, a review or a 👍 upstream accelerates that. Once merged, the
overlay builds collapse back into stock images.

**vLLM** ([fork](https://github.com/dkrisman/vllm), tag `bq38-2`):

| PR | What it does | Used here |
|---|---|---|
| [#52739][vllm-52739] Map unsupported reasoning_effort to nearest supported level | Claude Code's `reasoning_effort` values stop 400ing on Qwen templates | ✅ every request with thinking |
| [#52754][vllm-52754] Make Qwen3-VL video cost duration-proportional | Short clips stop consuming a full-length video token budget | fork lineage (video work, not wired here) |
| [#52759][vllm-52759] Surface TorchCodec video decode failures as client errors | Bad video inputs 400 instead of 500 | fork lineage (video work, not wired here) |

**LiteLLM** ([fork](https://github.com/dkrisman/litellm), tag `bq38-2`):

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
be at the beginning."`). With `LITELLM_DEMOTE_MIDTURN_SYSTEM=true` (set in the
compose files), the Anthropic bridge rewrites those as user turns — the same
place Claude Code historically put them. Upstream preserves mid-turn system
rows deliberately (PR [#34290](https://github.com/BerriAI/litellm/pull/34290)),
so this stays opt-in.

[vllm-52739]: https://github.com/vllm-project/vllm/pull/52739
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
