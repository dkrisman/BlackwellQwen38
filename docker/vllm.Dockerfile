# BlackwellQwen38 vLLM serving image.
#
# Pure-Python overlay of the dkrisman/vllm fork onto a pinned official vLLM
# image, plus the dkrisman/transformers Qwen3-VL video-processor patch
# (huggingface/transformers#48071). No kernel compilation: the fork deltas are
# frontend Python only.
#
# Two ways to match the fork's upstream base to the binary bits:
#   - BASE_IMAGE is a nightly built from the fork's upstream merge-base
#     (leave VLLM_WHEEL_SHA empty), or
#   - no such nightly exists: pick any recent nightly as BASE_IMAGE and set
#     VLLM_WHEEL_SHA to the merge-base commit — vLLM publishes a per-commit
#     wheel index at wheels.vllm.ai for every main commit, and that wheel
#     replaces the image's vllm before the overlay.

ARG BASE_IMAGE=vllm/vllm-openai:nightly-aa9903490c616dc6871e5acc62cec7bb1e5e9434

FROM busybox AS fetch
ARG VLLM_REPO=https://github.com/dkrisman/vllm.git
ARG VLLM_REF=bq38-3
ARG TRANSFORMERS_REPO=https://github.com/dkrisman/transformers.git
ARG TRANSFORMERS_REF=bq38-3
ADD ${VLLM_REPO}#${VLLM_REF} /vllm-src
ADD ${TRANSFORMERS_REPO}#${TRANSFORMERS_REF} /tfs-src

FROM ${BASE_IMAGE}
ARG VLLM_WHEEL_SHA=
RUN if [ -n "${VLLM_WHEEL_SHA}" ]; then \
      python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
        --index-url "https://wheels.vllm.ai/${VLLM_WHEEL_SHA}/cu130/" vllm; \
    fi
COPY --from=fetch /vllm-src/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY --from=fetch /tfs-src/src/transformers/models/qwen3_vl/ /usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_vl/

# Smoke test (CPU-only, runs on plain CI runners): the compiled extension
# loads, the fork features are present, and the transformers patch took.
RUN python3 - <<'EOF'
import pathlib

import vllm

print("vllm", vllm.__version__)
assert list(pathlib.Path(vllm.__file__).parent.glob("_C*.abi3.so")), "compiled ext missing"
import vllm._custom_ops
from vllm.multimodal.cache import MultiModalCache
assert hasattr(MultiModalCache, "put_if_fits")
from vllm.model_executor.models.qwen3_vl import _resolve_modality_mm_kwargs
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessorInitKwargs,
)
assert "max_pixels_per_frame" in Qwen3VLVideoProcessorInitKwargs.__annotations__
print("smoke OK")
EOF
