# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

If you discover a security vulnerability in EduSynapseOS, please report it responsibly:

1. **Email**: Send details to **security@gdlabs.io**
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge your report within **48 hours** and aim to provide a fix within **7 days** for critical issues.

## Security Considerations

### Authentication & Authorization
- JWT-based authentication with configurable token expiration
- Role-based access control (system admin, school admin, teacher, student, parent)
- API key authentication for service-to-service communication
- Per-tenant data isolation

### Data Protection
- Per-tenant isolated PostgreSQL databases
- Redis key prefixing for tenant isolation
- Qdrant collection namespacing
- Passwords hashed with bcrypt via passlib

### Secrets Management
- All secrets are managed via environment variables (`.env`)
- Never commit `.env` files or API keys to version control
- Rotate JWT secrets and database passwords regularly

### Rate Limiting
- API rate limiting via SlowAPI
- Configurable per-minute request limits
- Burst protection

## Best Practices for Deployment

- Always use HTTPS in production
- Set strong, unique passwords for all services
- Enable CORS only for trusted origins
- Keep all dependencies up to date
- Monitor application logs for suspicious activity
- Use network policies to restrict inter-service communication

## Acknowledgments

We appreciate the security research community's efforts in responsibly disclosing vulnerabilities. Contributors who report valid security issues will be acknowledged (with their permission) in release notes.
