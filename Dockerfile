# OpenAlfred Agent - Python Backend
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Layer cache: install deps first
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy source and entrypoint
COPY . .

EXPOSE 2024 7788 5883 5884

ENTRYPOINT ["bash", "docker-entrypoint.sh"]
