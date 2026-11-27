FROM python:3.15.2-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT=/usr/local
WORKDIR /app
COPY ./pyproject.toml ./uv.lock ./
RUN uv sync --frozen --no-dev --no-cache
# torch 2.14 routes some eager ops (e.g. the Qwen3 RoPE bmm) to Triton
# kernels, whose launcher Triton builds with a C compiler at first use. The
# slim image has none and stays apt-free, so keep those ops on the regular
# cuBLAS path. Read once when torch is imported.
ENV TORCH_DISABLE_NATIVE_JIT=1
