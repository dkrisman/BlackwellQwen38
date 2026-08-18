#!/usr/bin/env bash
# Smoke test for a running BlackwellQwen38 variant. Verifies:
#   1. vLLM is healthy and litellm lists the models
#   2. an OpenAI-style chat completion answers through the proxy
#   3. the claude-* wildcard route answers via the Anthropic /v1/messages API
#   4. Claude Code's WebSearch tool short-circuits to Brave and streams the
#      native server_tool_use / web_search_tool_result blocks (litellm PR
#      #37318 + the litellm-claude-code-websearch plugin)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_DIR/.env"
PORT="${LITELLM_PORT:-4000}"
KEY="${LITELLM_MASTER_KEY:?LITELLM_MASTER_KEY missing from .env}"
BASE="http://127.0.0.1:$PORT"
AUTH=(-H "Authorization: Bearer $KEY")

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== 1. model list =="
MODELS=$(curl -sS "${AUTH[@]}" "$BASE/v1/models")
echo "$MODELS" | grep -q '"id"' || fail "no models listed: $MODELS"
MODEL=$(echo "$MODELS" | python3 -c 'import json,sys; ids=[m["id"] for m in json.load(sys.stdin)["data"]]; print(next(i for i in ids if i.startswith("qwen")))')
echo "   model: $MODEL"

echo "== 2. chat completion (OpenAI API) =="
R=$(curl -sS "${AUTH[@]}" -H "Content-Type: application/json" "$BASE/v1/chat/completions" \
  -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: pong\"}], \"max_tokens\": 4000}")
echo "$R" | grep -qi "pong" || fail "chat completion: $R"
echo "   ok"

echo "== 3. claude-* wildcard (Anthropic API) =="
R=$(curl -sS "${AUTH[@]}" -H "Content-Type: application/json" "$BASE/v1/messages" \
  -d '{"model": "claude-sonnet-5", "max_tokens": 4000, "messages": [{"role": "user", "content": "Reply with exactly: pong"}]}')
echo "$R" | grep -qi "pong" || fail "/v1/messages: $R"
echo "   ok"

echo "== 4. WebSearch short-circuit (streaming native blocks) =="
# The exact request shape Claude Code sends for a standalone WebSearch call.
R=$(curl -sS -N "${AUTH[@]}" -H "Content-Type: application/json" "$BASE/v1/messages" \
  -d '{"model": "claude-sonnet-5", "max_tokens": 4000, "stream": true,
       "messages": [{"role": "user", "content": "Perform a web search for the query: current vllm release version"}],
       "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}]}')
echo "$R" | grep -q '"type":[ ]*"server_tool_use"' || fail "no server_tool_use block in stream"
echo "$R" | grep -q 'web_search_tool_result' || fail "no web_search_tool_result block in stream"
echo "   ok — native search blocks streamed"

echo
echo "ALL CHECKS PASSED ($MODEL)"
