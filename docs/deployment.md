# Deployment Guide

> Setting up EduSynapseOS for development and production environments.

## Prerequisites

- **Python**: 3.10 or higher
- **Docker**: 20.10+ with Docker Compose v2
- **LLM Provider**: At least one of:
  - [Ollama](https://ollama.ai/) (recommended for development -- free, local)
  - OpenAI API key
  - Anthropic API key
  - Google Gemini API key

## Docker Compose Deployment

### Services Overview

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| `central-db` | PostgreSQL 16 | 34001 | Platform database |
| `redis` | Redis 7 | 34002 | Cache + message broker |
| `qdrant` | Qdrant v1.16 | 34003/34004 | Vector database |
| `api` | EduSynapseOS | 34000 | API server |
| `worker` | EduSynapseOS | -- | Background worker |
| `mcp` | EduSynapseOS | 34005 | MCP server (optional) |

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/gdlabs-io/EduSynapseOS.git
cd EduSynapseOS

# 2. Configure environment
cp .env.example .env
# Edit .env -- at minimum, set strong passwords and JWT secret

# 3. Start all services
docker compose up -d

# 4. Verify health
docker compose ps
curl http://localhost:34000/health
```

### Starting Individual Services

```bash
# Infrastructure only (for local development)
docker compose up -d central-db redis qdrant

# API + Worker
docker compose up -d api worker

# MCP server (optional profile)
docker compose --profile mcp up -d mcp
```

### Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f worker
```

## Local Development Setup

For active development, run the API server locally while using Docker for infrastructure:

```bash
# 1. Start infrastructure
docker compose up -d central-db redis qdrant

# 2. Create Python environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 3. Configure for local development
cp .env.example .env
# Set hosts to localhost, adjust ports to match docker-compose

# 4. Run API with hot reload
uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 34000 --reload

# 5. Run background worker (separate terminal)
dramatiq src.infrastructure.background.tasks --processes 1 --threads 2
```

## LLM Provider Setup

### Ollama (Recommended for Development)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull qwen2.5:7b

# Verify
ollama list
```

In `.env`:
```
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=qwen2.5:7b
```

### Cloud Providers

Set the appropriate API key in `.env`:

```
# OpenAI
OPENAI_API_KEY=sk-...
DEFAULT_LLM_PROVIDER=openai

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
DEFAULT_LLM_PROVIDER=anthropic

# Google Gemini
GOOGLE_API_KEY=AIza...
DEFAULT_LLM_PROVIDER=google
```

You can also configure providers in `config/llm/providers.yaml` for more granular control, including routing strategies and per-agent provider overrides.

## Database Migrations

EduSynapseOS uses Alembic for database migrations with two targets:

- **Central**: Platform-level schema (tenants, system users, licenses)
- **Tenant**: Per-tenant schema (applied when provisioning new tenants)

```bash
# Central database migrations are applied automatically on startup
# Manual migration (if needed):
alembic upgrade head
```

## Production Considerations

### Security Checklist

- [ ] Generate strong, unique passwords for all database services
- [ ] Generate a strong JWT secret: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- [ ] Configure CORS origins to only allow your frontend domains
- [ ] Enable HTTPS via a reverse proxy (nginx, Caddy, etc.)
- [ ] Set `ENVIRONMENT=production` and `DEBUG=false`
- [ ] Restrict Docker socket access (required for tenant container management)
- [ ] Set up log aggregation and monitoring

### Reverse Proxy

Place a reverse proxy (nginx, Caddy, Traefik) in front of the API:

```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:34000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Scaling

- **API Workers**: Adjust `API_WORKERS` in `.env` (default: 2)
- **Background Workers**: Adjust `WORKER_PROCESSES` and `WORKER_THREADS`
- **Database**: Consider connection pooling with PgBouncer for high tenant counts
- **Redis**: Increase `maxmemory` for larger deployments
- **Qdrant**: Scale storage volume based on embedding count

### Monitoring

EduSynapseOS exposes:
- **Health endpoint**: `GET /health`
- **Prometheus metrics**: `GET /metrics`
- **OpenTelemetry traces**: Configure `OTEL_EXPORTER_OTLP_ENDPOINT` for Jaeger/Tempo
- **Structured logs**: JSON-formatted via structlog
