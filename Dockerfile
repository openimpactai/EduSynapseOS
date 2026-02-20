# ==============================================================================
# EduSynapseOS Backend Dockerfile
# ==============================================================================
# Multi-stage build for optimized production images.
#
# Stages:
#   1. base      - Common dependencies and system packages
#   2. builder   - Python dependencies installation
#   3. development - Development image with hot reload
#   4. runtime   - Production-optimized image
#   5. worker    - Dramatiq background worker
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Base - Common dependencies
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app

WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ------------------------------------------------------------------------------
# Stage 2: Builder - Install Python dependencies
# ------------------------------------------------------------------------------
FROM base AS builder

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --upgrade pip setuptools wheel \
    && pip install .

# ------------------------------------------------------------------------------
# Stage 3: Development - For local development with hot reload
# ------------------------------------------------------------------------------
FROM base AS development

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy all source files first for editable install
COPY . .

# Install with dev dependencies (editable mode)
RUN pip install -e ".[dev]"

RUN useradd --create-home --shell /bin/bash edusynapse \
    && mkdir -p /app/.cache \
    && chown -R edusynapse:edusynapse $APP_HOME

USER edusynapse

EXPOSE 8000

CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ------------------------------------------------------------------------------
# Stage 4: Runtime - Optimized production image
# ------------------------------------------------------------------------------
FROM base AS runtime

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src/ ./src/
COPY config/ ./config/
COPY docs/ ./docs/
COPY pyproject.toml ./
COPY alembic.ini ./

RUN useradd --create-home --shell /bin/bash edusynapse \
    && mkdir -p /app/logs /app/data /app/.cache \
    && chown -R edusynapse:edusynapse $APP_HOME

USER edusynapse

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# ------------------------------------------------------------------------------
# Stage 5: Worker - For Dramatiq background workers
# ------------------------------------------------------------------------------
FROM runtime AS worker

CMD ["dramatiq", "src.infrastructure.background.tasks", "--processes", "2", "--threads", "4"]

# ------------------------------------------------------------------------------
# Labels
# ------------------------------------------------------------------------------
LABEL org.opencontainers.image.title="EduSynapseOS Backend" \
      org.opencontainers.image.description="AI-native educational platform backend" \
      org.opencontainers.image.vendor="Global Digital Labs" \
      org.opencontainers.image.url="https://gdlabs.io" \
      org.opencontainers.image.source="https://github.com/gdlabs-io/EduSynapseOS" \
      org.opencontainers.image.licenses="LGPL-3.0-or-later"
