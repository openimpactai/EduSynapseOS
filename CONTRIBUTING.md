# Contributing to EduSynapseOS

Thank you for your interest in contributing to EduSynapseOS! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Be kind, constructive, and professional in all interactions.

## Getting Started

### Development Environment

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/EduSynapseOS.git
   cd EduSynapseOS
   ```

2. **Set up the environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Copy and configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your local settings
   ```

4. **Start infrastructure services**:
   ```bash
   docker compose up -d central-db redis qdrant
   ```

5. **Run the API server**:
   ```bash
   uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 34000 --reload
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

### Code Quality

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/
```

## How to Contribute

### Reporting Bugs

- Search existing [issues](https://github.com/gdlabs-io/EduSynapseOS/issues) first
- Use the bug report template
- Include: steps to reproduce, expected vs actual behavior, environment details

### Suggesting Features

- Open a [discussion](https://github.com/gdlabs-io/EduSynapseOS/discussions) or issue
- Describe the use case and proposed solution
- Consider the project's scope and architecture

### Submitting Pull Requests

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Write tests** for new functionality

4. **Ensure all checks pass**:
   ```bash
   pytest
   ruff check src/ tests/
   mypy src/
   ```

5. **Commit with a descriptive message**:
   ```bash
   git commit -m "feat: add support for new content type"
   ```

6. **Push and open a Pull Request** against `main`

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` -- New feature
- `fix:` -- Bug fix
- `docs:` -- Documentation changes
- `refactor:` -- Code refactoring (no functional change)
- `test:` -- Adding or updating tests
- `chore:` -- Maintenance tasks
- `perf:` -- Performance improvements

## Coding Standards

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) with a line length of 100
- Use type hints for all function signatures
- Use `ruff` for linting and formatting
- Use `mypy` for type checking

### Project Conventions

- **API routes**: Define in `src/api/v1/`, use dependency injection from `src/api/dependencies.py`
- **Domain services**: Place in `src/domains/{domain}/service.py`
- **Agent configs**: YAML files in `config/agents/`
- **Database models**: SQLAlchemy models in `src/infrastructure/database/models/`
- **Pydantic schemas**: Define in `src/models/`
- **Tests**: Mirror the `src/` structure under `tests/unit/` and `tests/integration/`

### License Headers

All new source files must include the SPDX license header:

```python
# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
```

## Architecture Decisions

Major architectural changes should be discussed in an issue before implementation. Key design principles:

- **Clean Architecture** -- Domain logic is independent of infrastructure
- **Multi-tenancy** -- All features must respect tenant isolation
- **Configuration-driven** -- Agent behavior is defined via YAML, not hardcoded
- **Provider-agnostic** -- LLM integrations go through LiteLLM abstraction

## Questions?

- Open a [GitHub Discussion](https://github.com/gdlabs-io/EduSynapseOS/discussions)
- Check existing documentation in the `docs/` directory

Thank you for contributing!
