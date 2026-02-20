# API Reference

> EduSynapseOS REST API v1

## Overview

EduSynapseOS exposes a RESTful API built with FastAPI. Once running, interactive documentation is available at:

- **Swagger UI**: `http://localhost:34000/docs`
- **ReDoc**: `http://localhost:34000/redoc`
- **OpenAPI JSON**: `http://localhost:34000/openapi.json`

## Authentication

### JWT Authentication

Most endpoints require a JWT Bearer token:

```
Authorization: Bearer <access_token>
```

#### Login

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "your_password",
  "tenant_code": "school_abc"
}
```

Response:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### Refresh Token

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJ..."
}
```

### API Key Authentication

For service-to-service integration (LMS, external systems):

```
X-API-Key: <api_key>
X-API-Secret: <api_secret>
```

## Tenant Context

All tenant-scoped endpoints require the tenant to be identified. The tenant is resolved from:
1. The JWT token's `tenant_code` claim
2. The `X-Tenant-Code` header (for API key auth)

## Core Endpoints

### Health & Monitoring

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |

### Authentication (`/api/v1/auth/`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/auth/login` | User login |
| `POST` | `/auth/refresh` | Refresh access token |
| `POST` | `/auth/logout` | Logout (invalidate tokens) |
| `GET` | `/auth/me` | Get current user info |

### Users (`/api/v1/users/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/users/` | List users (admin) |
| `POST` | `/users/` | Create user |
| `GET` | `/users/{id}` | Get user details |
| `PUT` | `/users/{id}` | Update user |
| `DELETE` | `/users/{id}` | Delete user |

### Companion (`/api/v1/companion/`)

The main AI companion interaction endpoint:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/companion/chat` | Send message to AI companion |
| `GET` | `/companion/sessions` | List chat sessions |
| `GET` | `/companion/sessions/{id}` | Get session history |

### Learning (`/api/v1/learning/`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/learning/sessions` | Start a learning session |
| `POST` | `/learning/sessions/{id}/message` | Send message in session |
| `GET` | `/learning/progress` | Get learning progress |

### Learning Tutor (`/api/v1/learning-tutor/`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/learning-tutor/chat` | Chat with subject tutor |
| `GET` | `/learning-tutor/subjects` | Get available subjects |

### Practice (`/api/v1/practice/`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/practice/sessions` | Start practice session |
| `POST` | `/practice/sessions/{id}/answer` | Submit answer |
| `GET` | `/practice/review-schedule` | Get spaced repetition schedule |

### Games (`/api/v1/games/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/games/` | List available games |
| `POST` | `/games/sessions` | Start game session |
| `POST` | `/games/sessions/{id}/move` | Make a move |
| `POST` | `/games/sessions/{id}/chat` | Chat with game coach |

### Content Creation (`/api/v1/content-creation/`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/content-creation/sessions` | Start content creation session |
| `POST` | `/content-creation/sessions/{id}/message` | Interact with content agents |
| `GET` | `/content-creation/content-types` | List H5P content types |

### Curriculum (`/api/v1/curriculum/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/curriculum/frameworks` | List curriculum frameworks |
| `GET` | `/curriculum/subjects` | List subjects |
| `GET` | `/curriculum/topics` | List topics |
| `GET` | `/curriculum/objectives` | List learning objectives |

### Diagnostics (`/api/v1/diagnostics/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/diagnostics/reports/{student_id}` | Get diagnostic report |
| `GET` | `/diagnostics/signals/{student_id}` | Get diagnostic signals |

### Analytics (`/api/v1/analytics/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/analytics/student/{id}` | Student analytics |
| `GET` | `/analytics/class/{id}` | Class analytics |
| `GET` | `/analytics/school/{id}` | School analytics |

### Schools (`/api/v1/schools/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/schools/` | List schools |
| `POST` | `/schools/` | Create school |
| `GET` | `/schools/{id}` | Get school details |

### Classes (`/api/v1/classes/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/classes/` | List classes |
| `POST` | `/classes/` | Create class |
| `GET` | `/classes/{id}` | Get class details |
| `GET` | `/classes/{id}/students` | List students in class |

### Teacher (`/api/v1/teacher/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/teacher/dashboard` | Teacher dashboard |
| `POST` | `/teacher/companion/chat` | Teacher AI companion |
| `GET` | `/teacher/students/{id}/progress` | Student progress report |

### Parent (`/api/v1/parent/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/parent/children` | List linked children |
| `GET` | `/parent/children/{id}/report` | Child progress report |

### Memory (`/api/v1/memory/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/memory/student/{id}` | Get student memory summary |
| `GET` | `/memory/student/{id}/search` | Search student memories |

### Tenant Management (`/api/v1/tenant/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/tenant/info` | Get current tenant info |
| `PUT` | `/tenant/settings` | Update tenant settings |

### Provisioning (`/api/v1/provisioning/`)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/provisioning/tenants` | Provision new tenant |
| `GET` | `/provisioning/tenants/{code}/status` | Check provisioning status |

### System Admin (`/api/v1/system/`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/system/tenants` | List all tenants |
| `GET` | `/system/stats` | System statistics |
| `POST` | `/system/auth/login` | System admin login |

## Rate Limiting

API requests are rate-limited per user:
- **Default**: 60 requests/minute
- **Burst**: 10 additional requests

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 55
X-RateLimit-Reset: 1234567890
```

## Error Responses

All errors follow a consistent format:

```json
{
  "detail": "Error description",
  "status_code": 400,
  "error_code": "VALIDATION_ERROR"
}
```

Common HTTP status codes:
- `400` -- Bad Request (validation errors)
- `401` -- Unauthorized (missing/invalid token)
- `403` -- Forbidden (insufficient permissions)
- `404` -- Not Found
- `429` -- Too Many Requests (rate limited)
- `500` -- Internal Server Error
