# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-06-01

### Added

- **AI Agent System**: Dynamic AI agents (Companion, Tutor, Practice Helper, Learning Tutor, Game Coach) with YAML-driven configuration
- **Educational Theories**: Bloom's Taxonomy, ZPD, VARK, Scaffolding, Mastery Learning, Socratic Method, Spaced Repetition (FSRS-6)
- **Cognitive Memory**: 4-layer memory system (Episodic, Semantic, Procedural, Associative) with RAG retrieval
- **Diagnostic Detectors**: Dyslexia, Dyscalculia, Attention Deficit, Auditory Processing, Visual Processing
- **Emotional Intelligence**: Real-time emotion detection and adaptive engagement
- **Proactive Engagement**: Monitors for struggle, inactivity, milestones, and emotional state
- **LangGraph Workflows**: Stateful, checkpointed conversation orchestration
- **Multi-Tenant Architecture**: Per-tenant PostgreSQL databases via Docker SDK with full data isolation
- **Board Games**: Chess, Connect4, Gomoku, Othello, Checkers with AI coaching integration
- **H5P Content Creation**: 30+ interactive content type converters with multi-language support (AR, EN, ES, FR, TR)
- **Persona System**: 8 configurable AI personas (Tutor, Mentor, Coach, Friend, Companion, Socratic, Teacher Assistant, System Explainer)
- **MCP Server**: Model Context Protocol integration for Claude Desktop
- **Background Processing**: Dramatiq task queue with APScheduler for curriculum sync, diagnostics, analytics
- **Observability**: OpenTelemetry tracing, Prometheus metrics, structured logging
- **API**: RESTful API v1 with 25+ route modules, Swagger/ReDoc documentation, JWT authentication
- **Rate Limiting**: Configurable per-minute rate limiting with burst protection
