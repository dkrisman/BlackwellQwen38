#!/usr/bin/env bash
# Launch Claude Code against the local BlackwellQwen38 proxy.
#
#   ./scripts/claude-code.sh              # interactive session
#   ./scripts/claude-code.sh -p "hello"   # args pass through to claude
#
# Works with any of the three variants — the proxy's claude-* wildcard route
# lands every Claude model name on the local Qwen worker.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$REPO_DIR/.env" ]]; then
  echo "error: $REPO_DIR/.env not found — cp .env.example .env and fill it in" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$REPO_DIR/.env"

export ANTHROPIC_BASE_URL="http://127.0.0.1:${LITELLM_PORT:-4000}"
export ANTHROPIC_AUTH_TOKEN="${LITELLM_MASTER_KEY:?LITELLM_MASTER_KEY missing from .env}"
# Any claude-* names work; these just pin what the CLI asks for by default.
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-sonnet-5}"
export ANTHROPIC_SMALL_FAST_MODEL="${ANTHROPIC_SMALL_FAST_MODEL:-claude-haiku-4-5}"
export DISABLE_TELEMETRY=1
export DISABLE_AUTOUPDATER=1

exec claude "$@"
