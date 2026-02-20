# Architecture

> EduSynapseOS System Architecture and Design Decisions

## Overview

EduSynapseOS follows a **Clean Architecture** pattern where domain logic is independent of infrastructure concerns. The system is designed as a multi-tenant SaaS backend with AI-first capabilities.

## System Layers

```
┌─────────────────────────────────────────────────────┐
│                    API Layer                         │
│         FastAPI + JWT Auth + Rate Limiting           │
├─────────────────────────────────────────────────────┤
│                  Domain Layer                        │
│   Companion │ Tutor │ Practice │ Gaming │ Content    │
├─────────────────────────────────────────────────────┤
│                   Core Layer                         │
│  Agents │ Memory │ Diagnostics │ Theories │ Emotion  │
├─────────────────────────────────────────────────────┤
│              Infrastructure Layer                    │
│  PostgreSQL │ Redis │ Qdrant │ Docker SDK │ Events   │
└─────────────────────────────────────────────────────┘
```

### API Layer (`src/api/`)

- **FastAPI application factory** (`app.py`) with middleware stack
- **JWT authentication** with role-based access control
- **Tenant resolution** middleware for multi-tenancy
- **Rate limiting** via SlowAPI
- **API Key authentication** for LMS/service integration
- **Versioned routes** (`v1/`) with 25+ endpoint modules

### Core Layer (`src/core/`)

The brain of the system:

#### AI Agent System (`core/agents/`)
- **DynamicAgent**: Configurable AI agents loaded from YAML definitions
- **PromptBuilder**: Template-based system prompt construction with variable interpolation
- **Capabilities**: Pluggable agent capabilities (question generation, feedback, emotional support, etc.)
- **Factory**: Agent instantiation with configuration validation

#### Cognitive Memory (`core/memory/`)
A 4-layer memory system inspired by cognitive science:
- **Episodic Memory**: Stores specific learning events and interactions
- **Semantic Memory**: Stores factual knowledge and concept relationships
- **Procedural Memory**: Tracks skill acquisition and mastery progression
- **Associative Memory**: Links concepts, interests, and learning patterns
- **RAG Retriever + Reranker**: Vector-based memory retrieval with reranking

#### Educational Theories (`core/educational/`)
Seven theory implementations that guide learning:
- **Bloom's Taxonomy**: Cognitive complexity progression
- **Zone of Proximal Development (ZPD)**: Vygotsky's scaffolded learning
- **VARK**: Visual, Auditory, Read/Write, Kinesthetic learning styles
- **Scaffolding**: Progressive complexity reduction
- **Mastery Learning**: Criterion-based progression
- **Socratic Method**: Question-driven discovery
- **Spaced Repetition**: FSRS-6 algorithm for optimal review scheduling

#### Workflow Orchestration (`core/orchestration/`)
LangGraph-based stateful workflows:
- Each conversation type (tutoring, practice, companion, gaming) has a dedicated workflow
- PostgreSQL-backed checkpointing for conversation persistence
- Typed state objects for each workflow

#### Diagnostics (`core/diagnostics/`)
Learning difficulty detection:
- **Dyslexia Detector**: Letter reversal, reading speed, phonological patterns
- **Dyscalculia Detector**: Number sense, calculation patterns, mathematical reasoning
- **Attention Detector**: Focus duration, task switching, completion patterns
- **Auditory Processing Detector**: Listening comprehension, auditory discrimination
- **Visual Processing Detector**: Spatial reasoning, visual pattern recognition

### Domain Layer (`src/domains/`)

Business logic organized by bounded context:
- **Companion**: Main AI companion service with proactive engagement
- **Gaming**: Board game engines (Chess, Connect4, Gomoku, Othello, Checkers) with AI coaching
- **Content Creation**: H5P interactive content generation via LLM
- **Learning**: Learning session management and progress tracking
- **Practice**: Adaptive practice with multiple modes
- **Teacher**: Teacher dashboard and analytics
- **Parent**: Parent portal and reporting
- **Curriculum**: External curriculum sync and management

### Infrastructure Layer (`src/infrastructure/`)

- **Database**: SQLAlchemy async with Alembic migrations, tenant database manager
- **Cache**: Redis client with tenant-prefixed keys
- **Vectors**: Qdrant client for embedding storage and similarity search
- **Background**: Dramatiq actors with APScheduler for periodic tasks
- **Events**: Internal event bus for decoupled communication
- **Notifications**: Multi-channel notification system (email, push, in-app)
- **Telemetry**: OpenTelemetry instrumentation setup

## Multi-Tenant Architecture

```
                   ┌──────────────────┐
                   │  Central Database │
                   │  (Platform Data)  │
                   └────────┬─────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
    ┌─────────v──┐ ┌───────v────┐ ┌──────v───────┐
    │ Tenant A   │ │ Tenant B   │ │ Tenant C     │
    │ PostgreSQL │ │ PostgreSQL │ │ PostgreSQL   │
    │ Container  │ │ Container  │ │ Container    │
    └────────────┘ └────────────┘ └──────────────┘
```

- **Central Database**: Stores tenants, system users, licenses, feature flags
- **Tenant Databases**: One PostgreSQL container per tenant, created dynamically via Docker SDK
- **Redis Isolation**: Key prefix pattern `tenant:{code}:*`
- **Qdrant Isolation**: Collection naming `tenant_{code}_{collection}`

## Configuration Architecture

```
config/
├── agents/        # Agent behavior (system prompts, tools, capabilities)
├── diagnostics/   # Diagnostic thresholds and recommendations
├── h5p-locales/   # Content type localizations (5 languages)
├── h5p-schemas/   # H5P content type JSON schemas
├── llm/           # LLM provider routing and configuration
├── personas/      # AI persona definitions (tone, style, boundaries)
└── theories/      # Educational theory parameters
```

Agents are fully configurable via YAML -- no code changes needed to adjust agent behavior, add new personas, or tune educational parameters.

## Data Flow

### Typical Conversation Flow

```
1. Student sends message
   → API receives request with JWT + tenant context
   → Tenant middleware resolves database connection

2. Domain service processes request
   → Loads agent config + persona from YAML
   → Retrieves relevant memories (RAG)
   → Builds system prompt with context

3. LangGraph workflow executes
   → Agent processes message with LLM
   → Tools are called (curriculum lookup, navigation, etc.)
   → State is checkpointed to PostgreSQL

4. Post-processing
   → Emotion analysis on response
   → Diagnostic signals collected
   → Memory layers updated
   → Analytics events emitted
   → Proactive monitors evaluate engagement

5. Response returned to student
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM abstraction | LiteLLM | Provider-agnostic, easy switching, cost routing |
| Workflow engine | LangGraph | Stateful conversations, checkpointing, tool use |
| Multi-tenancy | Docker containers | Full data isolation, independent scaling |
| Memory system | Custom 4-layer | Mirrors cognitive science for natural learning |
| Background tasks | Dramatiq | Redis-backed, reliable, simple API |
| Spaced repetition | FSRS-6 | State-of-the-art algorithm, proven effectiveness |
| Config format | YAML | Human-readable, version-controllable, no deploys needed |
