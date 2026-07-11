FROM python:3.14.6-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT=/usr/local
WORKDIR /app
COPY ./pyproject.toml ./uv.lock ./
RUN uv sync --frozen --no-dev --no-cache
