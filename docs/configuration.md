# Configuration Guide

> Complete reference for configuring EduSynapseOS.

## Overview

EduSynapseOS uses a layered configuration system:

1. **Environment variables** (`.env`) -- Infrastructure, secrets, feature flags
2. **YAML configuration** (`config/`) -- Agent behaviors, personas, theories, LLM routing
3. **Database settings** -- Per-tenant runtime configuration

## Environment Variables

Copy `.env.example` to `.env` and customize. All variables with descriptions are in the example file.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `development` | Environment mode (`development`, `production`) |
| `DEBUG` | `true` | Enable debug mode |
| `LOG_LEVEL` | `DEBUG` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `CENTRAL_DB_USER` | `edusynapse` | Central PostgreSQL username |
| `CENTRAL_DB_PASSWORD` | -- | Central PostgreSQL password |
| `CENTRAL_DB_HOST` | `localhost` | Central PostgreSQL host |
| `CENTRAL_DB_PORT` | `34001` | Central PostgreSQL port |
| `TENANT_DB_USER` | `edusynapse` | Default username for tenant databases |
| `TENANT_DB_PASSWORD` | -- | Default password for tenant databases |
| `TENANT_DB_PORT_RANGE_START` | `35000` | Starting port for tenant DB containers |

### Redis

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `34002` | Redis port |
| `REDIS_PASSWORD` | -- | Redis password |

### Vector Database

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_HTTP_PORT` | `34003` | Qdrant HTTP port |
| `QDRANT_GRPC_PORT` | `34004` | Qdrant gRPC port |
| `QDRANT_API_KEY` | -- | Qdrant API key (optional) |

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `JWT_SECRET_KEY` | -- | JWT signing secret (generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`) |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Access token TTL |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `7` | Refresh token TTL |

### LLM Providers

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_LLM_PROVIDER` | `ollama` | Default LLM provider |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_API_KEY` | -- | Ollama API key (if required) |
| `OLLAMA_DEFAULT_MODEL` | `qwen2.5:7b` | Default Ollama model |
| `OPENAI_API_KEY` | -- | OpenAI API key |
| `ANTHROPIC_API_KEY` | -- | Anthropic API key |
| `GOOGLE_API_KEY` | -- | Google Gemini API key |

### CORS

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Allowed CORS origins (comma-separated) |
| `CORS_ALLOW_CREDENTIALS` | `true` | Allow credentials in CORS |

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_ENABLED` | `false` | Enable OpenTelemetry |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP exporter endpoint |
| `OTEL_SERVICE_NAME` | `edusynapseos` | Service name for traces |

## YAML Configuration

### Agent Configuration (`config/agents/`)

Each agent is defined by a YAML file specifying its behavior:

```yaml
# config/agents/companion.yaml
name: companion
description: "Main AI learning companion"

llm:
  provider: local          # References config/llm/providers.yaml
  model: qwen2.5:7b       # Optional, uses provider default if omitted
  temperature: 0.7
  max_tokens: 1024

system_prompt:
  role: |
    You are a friendly learning companion for {student_name}.
    Grade level: {grade_level}
  tool_instructions: |
    Use tools to gather information before responding.

tools:
  enabled: true
  max_iterations: 5
  definitions:
    - name: get_student_context
      enabled: true
      group: information_gathering
      order: 1

capabilities:
  - emotional_support
  - message_analysis
  - companion_decision
```

### Persona Configuration (`config/personas/`)

Personas define the AI's communication style:

```yaml
# config/personas/tutor.yaml
name: tutor
display_name: "Patient Tutor"
description: "A patient, knowledgeable tutor"

tone: warm, encouraging, patient
communication_style: structured, step-by-step
language_level: age-appropriate

boundaries:
  - Always stay on educational topics
  - Never share personal opinions on sensitive subjects
  - Redirect off-topic conversations gently
```

### Educational Theory Configuration (`config/theories/`)

```yaml
# config/theories/bloom.yaml
name: bloom
display_name: "Bloom's Taxonomy"

levels:
  - name: remember
    order: 1
    verbs: [define, list, recall, identify]
    threshold: 0.3
  - name: understand
    order: 2
    verbs: [explain, describe, summarize, interpret]
    threshold: 0.5
  # ... higher levels
```

### LLM Provider Configuration (`config/llm/providers.yaml`)

Defines available LLM providers and routing strategies:

```yaml
default_provider: local

providers:
  local:
    enabled: true
    type: ollama
    api_base: "http://localhost:11434"
    default_model: qwen2.5:7b
    timeout_seconds: 60
    max_retries: 3

routing_strategies:
  hybrid:
    primary: local
    fallbacks: [google, openai]
    description: "Local first, fallback to cloud"

embedding:
  provider: google
  model: text-embedding-004
  dimension: 768
```

### Diagnostic Configuration (`config/diagnostics/`)

```yaml
# config/diagnostics/thresholds.yaml
detectors:
  dyslexia:
    min_signals: 5
    confidence_threshold: 0.7
    alert_threshold: 0.85

  dyscalculia:
    min_signals: 5
    confidence_threshold: 0.7
    alert_threshold: 0.85
```

## H5P Content Configuration

### Locales (`config/h5p-locales/`)

Multi-language support for 24 H5P content types across 5 languages (AR, EN, ES, FR, TR). Each content type has a JSON file per language with translated strings.

### Schemas (`config/h5p-schemas/`)

JSON schemas defining the structure of each H5P content type, organized by category:
- `assessment/` -- Quiz and evaluation content types
- `game/` -- Interactive game content types
- `learning/` -- Educational content types
- `media/` -- Media-rich content types
- `vocabulary/` -- Vocabulary building content types

## Customization Tips

### Adding a New AI Agent

1. Create `config/agents/your_agent.yaml` following the schema above
2. Create a domain service in `src/domains/your_domain/service.py`
3. Create a workflow in `src/core/orchestration/workflows/your_workflow.py`
4. Register API routes in `src/api/v1/your_routes.py`

### Adding a New LLM Provider

1. Add the provider definition in `config/llm/providers.yaml`
2. LiteLLM handles the API translation automatically
3. Reference the provider code in agent configs

### Customizing Educational Parameters

All educational theory parameters are in `config/theories/`. Adjust thresholds, progression rules, and scheduling parameters without code changes.
