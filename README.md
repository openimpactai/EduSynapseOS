<p align="center">
  <h1 align="center">EduSynapseOS</h1>
  <p align="center">
    AI-native educational platform backend for personalized learning at scale
    <br />
    <a href="docs/architecture.md"><strong>Architecture</strong></a> &middot;
    <a href="docs/deployment.md"><strong>Deployment</strong></a> &middot;
    <a href="docs/configuration.md"><strong>Configuration</strong></a> &middot;
    <a href="docs/api.md"><strong>API Reference</strong></a>
  </p>
</p>

---

EduSynapseOS is an open-source, AI-native educational platform backend that delivers personalized learning experiences through dynamic AI tutoring, intelligent practice, and proactive student engagement. Built by [Global Digital Labs](https://gdlabs.io).

## Key Features

### AI-Powered Learning
- **Dynamic AI Agents** -- Companion, Tutor, Practice Helper, and specialized subject tutors powered by LLM (supports Ollama, OpenAI, Anthropic, Google Gemini via LiteLLM)
- **7 Educational Theories** -- Bloom's Taxonomy, Zone of Proximal Development, VARK Learning Styles, Scaffolding, Mastery Learning, Socratic Method, Spaced Repetition (FSRS-6)
- **4-Layer Cognitive Memory** -- Episodic, Semantic, Procedural, and Associative memory with RAG-based retrieval
- **LangGraph Workflows** -- Stateful, checkpointed conversation orchestration with PostgreSQL persistence

### Intelligent Assessment
- **5 Learning Difficulty Detectors** -- Dyslexia, Dyscalculia, Attention Deficit, Auditory Processing, Visual Processing
- **Emotional Intelligence** -- Real-time emotion detection and adaptive response
- **Proactive Engagement** -- Monitors for struggle, inactivity, milestones, and emotional state

### Content & Gamification
- **H5P Content Creation** -- 30+ interactive content type converters with multi-language support (AR, EN, ES, FR, TR)
- **Educational Board Games** -- Chess, Connect4, Gomoku, Othello, Checkers with AI coaching
- **Adaptive Practice Modes** -- Multiple practice strategies with spaced repetition scheduling

### Enterprise Architecture
- **Multi-Tenant SaaS** -- Per-tenant isolated PostgreSQL databases via Docker SDK
- **MCP Integration** -- Model Context Protocol server for Claude Desktop and compatible AI assistants
- **Observability** -- OpenTelemetry, Prometheus metrics, structured logging (structlog)
- **Background Processing** -- Dramatiq task queue with APScheduler for recurring jobs

## Architecture Overview

```
                    +------------------+
                    |   Frontend App   |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   FastAPI + JWT   |
                    |   Rate Limiting   |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v------+  +----v-------+
     | AI Agents  |  | Workflows   |  | Background |
     | (LiteLLM)  |  | (LangGraph) |  | (Dramatiq) |
     +--------+---+  +------+------+  +----+-------+
              |              |              |
     +--------v--------------v--------------v-------+
     |              Domain Services                  |
     |  Companion | Tutor | Practice | Gaming | ...  |
     +------+----------+-----------+--------+-------+
            |          |           |        |
   +--------v--+ +-----v-----+ +--v----+ +-v--------+
   | PostgreSQL | |   Redis   | |Qdrant | |  Docker  |
   |  Central + | |  Cache +  | |Vector | |  SDK     |
   |  Tenants   | |  Broker   | |  DB   | | (Tenant  |
   +------------+ +-----------+ +-------+ |  DBs)    |
                                          +----------+
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- An LLM provider (local Ollama recommended for development)

### 1. Clone and Configure

```bash
git clone https://github.com/gdlabs-io/EduSynapseOS.git
cd EduSynapseOS

# Copy environment template and configure
cp .env.example .env
# Edit .env with your settings (database passwords, JWT secret, LLM API keys)
```

### 2. Start with Docker Compose

```bash
# Start all infrastructure services
docker compose up -d

# Check service health
docker compose ps

# View API logs
docker compose logs -f api
```

### 3. Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e ".[dev]"

# Run the API server
uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 34000 --reload

# Run tests
pytest

# Code quality
ruff check src/
mypy src/
```

### 4. Access the API

- **Swagger UI**: http://localhost:34000/docs
- **ReDoc**: http://localhost:34000/redoc
- **Health Check**: http://localhost:34000/health
- **Metrics**: http://localhost:34000/metrics

## Project Structure

```
EduSynapseOS/
├── src/
│   ├── api/              # FastAPI application, routes, middleware
│   │   ├── v1/           # API v1 endpoints
│   │   ├── middleware/    # Auth, rate limiting, tenant resolution
│   │   └── routes/       # Health, metrics
│   ├── core/             # Core business logic
│   │   ├── agents/       # Dynamic AI agent system
│   │   ├── diagnostics/  # Learning difficulty detectors
│   │   ├── educational/  # Educational theory implementations
│   │   ├── emotional/    # Emotion analysis
│   │   ├── intelligence/ # LLM client, embeddings, MCP
│   │   ├── memory/       # 4-layer cognitive memory + RAG
│   │   ├── orchestration/# LangGraph workflow definitions
│   │   ├── personas/     # AI persona management
│   │   └── proactive/    # Proactive engagement monitors
│   ├── domains/          # Domain services (companion, gaming, learning, etc.)
│   ├── infrastructure/   # Database, cache, events, telemetry
│   ├── models/           # Pydantic schemas / DTOs
│   ├── services/         # External service integrations (H5P)
│   ├── tools/            # Agent tool implementations
│   └── utils/            # Shared utilities
├── config/               # YAML configuration
│   ├── agents/           # AI agent configurations
│   ├── diagnostics/      # Diagnostic thresholds
│   ├── h5p-locales/      # H5P content localizations
│   ├── h5p-schemas/      # H5P content type schemas
│   ├── llm/              # LLM provider configuration
│   ├── personas/         # AI persona definitions
│   └── theories/         # Educational theory parameters
├── scripts/              # Database init & utility scripts
├── tests/                # Unit, integration, and E2E tests
├── docs/                 # Documentation
├── Dockerfile            # Multi-stage Docker build
├── docker-compose.yml    # Service orchestration
└── pyproject.toml        # Python project configuration
```

## Configuration

EduSynapseOS uses a layered configuration approach:

- **Environment variables** (`.env`) -- Infrastructure, secrets, feature flags
- **YAML configs** (`config/`) -- Agent behaviors, personas, educational theories, LLM providers
- **Database settings** -- Per-tenant runtime configuration

See [Configuration Guide](docs/configuration.md) for details.

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System architecture, design decisions, and component overview |
| [Deployment](docs/deployment.md) | Production deployment, Docker setup, and scaling |
| [Configuration](docs/configuration.md) | Environment variables, YAML configs, and customization |
| [API Reference](docs/api.md) | REST API endpoints, authentication, and usage |
| [Contributing](CONTRIBUTING.md) | How to contribute to the project |
| [Security](SECURITY.md) | Security policy and vulnerability reporting |
| [Changelog](CHANGELOG.md) | Version history and release notes |

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **API Framework** | FastAPI + Uvicorn |
| **Database** | PostgreSQL 16 (multi-tenant) |
| **Cache & Broker** | Redis 7 |
| **Vector Database** | Qdrant |
| **LLM Integration** | LiteLLM (Ollama, OpenAI, Anthropic, Google) |
| **Workflow Engine** | LangGraph |
| **Background Tasks** | Dramatiq + APScheduler |
| **Spaced Repetition** | FSRS-6 |
| **MCP Support** | Model Context Protocol |
| **Observability** | OpenTelemetry + Prometheus + structlog |
| **Container Runtime** | Docker + Docker Compose |

## Contributing

We welcome contributions! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct, development workflow, and the process for submitting pull requests.

## Security

For security concerns, please review our [Security Policy](SECURITY.md). Do not open public issues for security vulnerabilities -- follow the responsible disclosure process described in the policy.

## License

Copyright (C) 2025 [Global Digital Labs](https://gdlabs.io)

This project is licensed under the **GNU Lesser General Public License v3.0 or later** (LGPL-3.0-or-later). See the [LICENSE](LICENSE) file for the full license text.

You are free to:
- Use this software in your own applications (including proprietary ones)
- Modify and distribute this software
- Use this as a library in proprietary software

Under the condition that:
- Modifications to EduSynapseOS itself must be released under LGPL-3.0
- You must preserve the original copyright and licence notices

See the [GNU LGPL v3.0](https://www.gnu.org/licenses/lgpl-3.0.html) for full terms.

## LGPL-3.0 Compliance Notes

EduSynapseOS is licensed under the GNU Lesser General Public License v3.0 or later (LGPL-3.0-or-later). The following points summarise how the project is intended to comply with, and be used under, the LGPL:

- **Library vs Application**
  EduSynapseOS is intended to be used as a Library in the sense of LGPL-3.0 §0, with external systems (e.g. LMS platforms or custom services) acting as Applications that make use of its interfaces while remaining under their own licences.

- **Rights for modified versions**
  Users are free to fork, modify, and redistribute EduSynapseOS under the terms of LGPL-3.0-or-later. Deployments (including container images) should allow replacing EduSynapseOS with a modified version by rebuilding or overriding the provided source, in line with LGPL-3.0 §4.

- **Use in proprietary or differently licensed software**
  Applications that use EduSynapseOS as a Library (for example, by calling its APIs or importing it as a dependency) may be licensed under their own terms, provided they comply with the LGPL obligations for EduSynapseOS itself, including preservation of copyright and licence notices.

- **Third-party components**
  Any third-party libraries used by EduSynapseOS, and their licences, are (or will be) documented in a "Third-Party Licences" section or file. Where such components are LGPL-licensed, they are used in a manner consistent with their weak-copyleft terms (e.g. as dynamically-linked or standard interpreter-level dependencies).

For the full legal terms, see the included [LICENSE](LICENSE) file (LGPL-3.0-or-later) and the official text at: https://www.gnu.org/licenses/lgpl-3.0.en.html
