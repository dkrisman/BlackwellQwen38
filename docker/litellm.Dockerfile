# BlackwellQwen38 LiteLLM proxy image.
#
# Pure-Python overlay of the dkrisman/litellm fork onto the nearest official
# release image, with the litellm-claude-code-websearch plugin baked in.

ARG BASE_IMAGE=ghcr.io/berriai/litellm:v1.98.0-rc.1

FROM busybox AS fetch
ARG LITELLM_REPO=https://github.com/dkrisman/litellm.git
ARG LITELLM_REF=bq38-7
ARG PLUGIN_REPO=https://github.com/dkrisman/litellm-claude-code-websearch.git
ARG PLUGIN_REF=main
ADD ${LITELLM_REPO}#${LITELLM_REF} /litellm-src
ADD ${PLUGIN_REPO}#${PLUGIN_REF} /plugin

FROM ${BASE_IMAGE}
COPY --from=fetch /litellm-src/litellm/ /app/.venv/lib/python3.13/site-packages/litellm/
COPY --from=fetch /plugin /plugin
ENV PYTHONPATH=/plugin/src:/app

# Smoke test: fork feature present, plugin importable.
RUN /app/.venv/bin/python - <<'EOF'
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
assert hasattr(LiteLLMAnthropicMessagesAdapter, "_demote_midturn_system_messages")
import litellm_claude_code_websearch as m
print("smoke OK:", type(m.handler_instance).__name__)
EOF
