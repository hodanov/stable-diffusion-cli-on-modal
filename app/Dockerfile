FROM python:3.11.11-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT=/usr/local
COPY ./pyproject.toml ./uv.lock ./
RUN apt-get update \
    && apt-get install wget git libgl1-mesa-glx libglib2.0-0 --no-install-recommends -y \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    && uv sync --frozen --no-dev --no-cache
