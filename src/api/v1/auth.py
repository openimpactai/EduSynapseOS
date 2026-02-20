# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Authentication API endpoints.

This module provides endpoints for user authentication:
- POST /exchange - Exchange API credentials for JWT tokens (LMS integration)
- POST /logout - User logout
- POST /refresh - Refresh access token
- GET /me - Get current user info
- GET /sessions - List active sessions
- DELETE /sessions/{session_id} - Revoke a session

Users are authenticated via LMS integration - the LMS authenticates users
and asserts their identity to EduSynapseOS using API credentials.

Example (LMS integration):
    POST /api/v1/auth/exchange
    Headers:
        X-API-Key: tk_abc123...
        X-API-Secret: ts_xyz789...
    Body:
        {
            "user": {
                "external_id": "lms_user_123",
                "email": "student@school.com",
                "first_name": "John",
                "last_name": "Doe",
                "user_type": "student"
            }
        }
"""

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.dependencies import (
    get_central_db,
    get_tenant_db,
    get_jwt_manager,
    require_auth,
    require_tenant,
)
from src.api.middleware.api_key_auth import (
    APIAuthContext,
    get_api_auth,
)
from src.api.middleware.auth import CurrentUser
from src.api.middleware.tenant import TenantContext
from src.api.middleware.rate_limit import limiter, RATE_LIMIT_AUTH, get_ip_only
from src.domains.auth.api_key_service import APIKeyService
from src.domains.auth.jwt import JWTManager
from src.domains.auth.service import (
    AuthService,
    DeviceInfo,
    TokenRefreshError,
)
from src.infrastructure.database.models.tenant.user import (
    User,
    UserRole,
    Role,
    RolePermission,
)
from src.infrastructure.database.models.tenant.session import (
    UserSession,
    RefreshToken,
)
from src.models.auth import (
    TokenResponse,
    RefreshTokenRequest,
    LogoutRequest,
    SessionInfo,
    SessionListResponse,
    TokenExchangeRequest,
    TokenExchangeResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_device_info(request: Request) -> DeviceInfo:
    """Extract device info from request.

    Args:
        request: HTTP request.

    Returns:
        DeviceInfo with client information.
    """
    # Get client IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else None

    # Get user agent
    user_agent = request.headers.get("User-Agent", "")

    # Detect device type from user agent
    device_type = "desktop"
    ua_lower = user_agent.lower()
    if "mobile" in ua_lower or "android" in ua_lower:
        device_type = "mobile"
    elif "tablet" in ua_lower or "ipad" in ua_lower:
        device_type = "tablet"

    return DeviceInfo(
        device_type=device_type,
        device_name=user_agent[:100] if user_agent else None,
        ip_address=ip,
        user_agent=user_agent[:500] if user_agent else None,
    )


@router.post(
    "/exchange",
    response_model=TokenExchangeResponse,
    summary="Exchange API credentials for JWT tokens",
    description="""
    Exchange API key/secret credentials for JWT tokens.

    This endpoint is used by LMS systems that authenticate users internally
    and need to assert user identity to EduSynapseOS.

    Requirements:
    - Valid API key and secret in headers (X-API-Key, X-API-Secret)
    - User assertion in request body

    The endpoint will:
    1. Validate API credentials
    2. Find or create the user in tenant database
    3. Generate JWT tokens for subsequent API calls

    Headers:
    - X-API-Key: Tenant's API key (tk_...)
    - X-API-Secret: Tenant's API secret (ts_...)
    """,
)
@limiter.limit(RATE_LIMIT_AUTH, key_func=get_ip_only)
async def exchange_token(
    request: Request,
    data: TokenExchangeRequest,
    tenant_db: AsyncSession = Depends(get_tenant_db),
    central_db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
) -> TokenExchangeResponse:
    """Exchange API credentials for JWT tokens.

    This endpoint validates API credentials and exchanges them for JWT tokens.
    The LMS asserts user identity, and EduSynapseOS trusts this assertion
    after validating the API credentials.

    Args:
        request: HTTP request with API credentials in headers.
        data: Token exchange request with user assertion.
        tenant_db: Tenant database session.
        central_db: Central database session.
        jwt_manager: JWT manager for token creation.

    Returns:
        TokenExchangeResponse with JWT tokens and user info.

    Raises:
        HTTPException: If API credentials are invalid or missing.
    """
    # Get API auth context (set by middleware)
    api_auth = get_api_auth(request)
    if not api_auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API authentication required. Provide X-API-Key and X-API-Secret headers.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    tenant = api_auth.tenant
    user_data = data.user

    # Find or create user in tenant database
    user, user_created = await _find_or_create_user(
        db=tenant_db,
        external_id=user_data.external_id,
        email=user_data.email,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        user_type=user_data.user_type,
    )

    # Get user roles and permissions
    roles, permissions, school_ids = await _get_user_roles_permissions(
        tenant_db, user.id
    )

    # Create token pair
    tokens = jwt_manager.create_token_pair(
        user_id=user.id,
        tenant_id=tenant.id,
        tenant_code=tenant.code,
        user_type=user.user_type,
        roles=roles,
        permissions=permissions,
        school_ids=school_ids,
        preferred_language=user.preferred_language,
    )

    # Create session
    device_info = DeviceInfo(
        device_type=data.device_type or "api",
        device_name=data.device_name,
        ip_address=_get_client_ip(request),
        user_agent=request.headers.get("User-Agent", "")[:500],
    )

    session = UserSession(
        user_id=user.id,
        access_token_hash=jwt_manager.hash_token(tokens.access_token),
        device_type=device_info.device_type,
        device_name=device_info.device_name,
        ip_address=device_info.ip_address,
        user_agent=device_info.user_agent,
        expires_at=datetime.now(timezone.utc) + timedelta(
            minutes=jwt_manager._settings.access_token_expire_minutes
        ),
    )
    tenant_db.add(session)

    # Store refresh token
    from uuid import uuid4
    refresh_token_record = RefreshToken(
        user_id=user.id,
        session_id=session.id,
        token_hash=jwt_manager.hash_token(tokens.refresh_token),
        family_id=str(uuid4()),
        generation=1,
        expires_at=datetime.now(timezone.utc) + timedelta(
            days=jwt_manager._settings.refresh_token_expire_days
        ),
    )
    tenant_db.add(refresh_token_record)

    # Update user activity
    user.last_login_at = datetime.now(timezone.utc)
    user.last_activity_at = datetime.now(timezone.utc)

    await tenant_db.commit()

    # Log exchange in central DB
    api_key_service = APIKeyService(central_db)
    await api_key_service.log_audit_event(
        credential=api_auth.credential,
        action="exchange_token",
        success=True,
        endpoint=request.url.path,
        method=request.method,
        user_id_asserted=user_data.external_id,
        ip_address=_get_client_ip(request),
        user_agent=request.headers.get("User-Agent"),
    )
    await central_db.commit()

    logger.info(
        "Token exchange successful: user=%s, tenant=%s, created=%s",
        user.id,
        tenant.code,
        user_created,
    )

    return TokenExchangeResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        token_type=tokens.token_type,
        expires_in=tokens.expires_in,
        refresh_expires_in=tokens.refresh_expires_in,
        user_id=user.id,
        user_created=user_created,
    )


async def _find_or_create_user(
    db: AsyncSession,
    external_id: str,
    email: str,
    first_name: str | None,
    last_name: str | None,
    user_type: str,
) -> tuple[User, bool]:
    """Find existing user or create new one.

    Looks up user by SSO external ID first, then by email.
    If not found, creates a new user.

    Args:
        db: Database session.
        external_id: External ID from LMS.
        email: User email.
        first_name: User's first name (optional).
        last_name: User's last name (optional).
        user_type: User type.

    Returns:
        Tuple of (user, created) where created is True if new user was made.
    """
    # First try to find by external ID (for returning users)
    stmt = select(User).where(
        User.sso_provider == "lms",
        User.sso_external_id == external_id,
        User.deleted_at.is_(None),
    )
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user:
        # Update user info if changed (only if provided)
        user.email = email
        if first_name:
            user.first_name = first_name
        if last_name:
            user.last_name = last_name
        if user.status == "pending":
            user.status = "active"
        return user, False

    # Try to find by email (for users who might exist from other sources)
    stmt = select(User).where(
        User.email == email,
        User.deleted_at.is_(None),
    )
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user:
        # Link existing user to LMS
        user.sso_provider = "lms"
        user.sso_external_id = external_id
        if first_name:
            user.first_name = first_name
        if last_name:
            user.last_name = last_name
        if user.status == "pending":
            user.status = "active"
        return user, False

    # Create new user - use defaults if names not provided
    user = User(
        email=email,
        first_name=first_name or "User",
        last_name=last_name or "",
        user_type=user_type,
        status="active",
        email_verified=True,  # LMS has verified the user
        sso_provider="lms",
        sso_external_id=external_id,
    )
    db.add(user)
    await db.flush()

    logger.info(
        "Created new user from LMS: %s (external_id=%s, email=%s)",
        user.id,
        external_id,
        email,
    )

    return user, True


async def _get_user_roles_permissions(
    db: AsyncSession,
    user_id: str,
) -> tuple[list[str], list[str], list[str]]:
    """Get user's roles, permissions, and accessible school IDs.

    Args:
        db: Database session.
        user_id: User ID.

    Returns:
        Tuple of (role_codes, permission_codes, school_ids).
    """
    stmt = (
        select(UserRole)
        .options(
            selectinload(UserRole.role)
            .selectinload(Role.role_permissions)
            .selectinload(RolePermission.permission)
        )
        .where(UserRole.user_id == user_id)
    )
    result = await db.execute(stmt)
    user_roles = result.scalars().all()

    roles: list[str] = []
    permissions: list[str] = []
    school_ids: list[str] = []

    for user_role in user_roles:
        if not user_role.is_active:
            continue

        role = user_role.role
        if role:
            roles.append(role.code)

            for rp in role.role_permissions:
                if rp.permission.code not in permissions:
                    permissions.append(rp.permission.code)

        if user_role.school_id and user_role.school_id not in school_ids:
            school_ids.append(user_role.school_id)

    return roles, permissions, school_ids


def _get_client_ip(request: Request) -> str | None:
    """Get client IP from request.

    Args:
        request: HTTP request.

    Returns:
        Client IP address or None.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
    description="Get new access token using refresh token. Implements token rotation.",
)
@limiter.limit(RATE_LIMIT_AUTH, key_func=get_ip_only)
async def refresh_token(
    request: Request,
    data: RefreshTokenRequest,
    tenant: TenantContext = Depends(require_tenant),
    db: AsyncSession = Depends(get_tenant_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
) -> TokenResponse:
    """Refresh access token.

    Args:
        request: HTTP request.
        data: Refresh token request.
        tenant: Tenant context.
        db: Database session.
        jwt_manager: JWT manager.

    Returns:
        TokenResponse with new tokens.

    Raises:
        HTTPException: If refresh fails.
    """
    auth_service = AuthService(db, jwt_manager)

    try:
        tokens = await auth_service.refresh_tokens(
            refresh_token=data.refresh_token,
            tenant_id=tenant.id,
            tenant_code=tenant.code,
        )

        return TokenResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type=tokens.token_type,
            expires_in=tokens.expires_in,
            refresh_expires_in=tokens.refresh_expires_in,
        )

    except TokenRefreshError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="User logout",
    description="Logout user and revoke tokens. Can revoke current or all sessions.",
)
async def logout(
    request: Request,
    data: LogoutRequest,
    current_user: CurrentUser = Depends(require_auth),
    db: AsyncSession = Depends(get_tenant_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
) -> None:
    """Logout user and revoke session.

    Args:
        request: HTTP request.
        data: Logout request.
        current_user: Authenticated user.
        db: Database session.
        jwt_manager: JWT manager.
    """
    auth_service = AuthService(db, jwt_manager)

    # Get current session from request header token
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else None

    if data.all_sessions:
        await auth_service.logout(
            session_id="",  # Not used when revoke_all=True
            revoke_all=True,
            user_id=current_user.id,
        )
    elif token:
        session = await auth_service.validate_session(token)
        if session:
            await auth_service.logout(session_id=session.id)


@router.get(
    "/me",
    summary="Get current user",
    description="Get information about the currently authenticated user.",
)
async def get_current_user_info(
    current_user: CurrentUser = Depends(require_auth),
) -> dict:
    """Get current user information.

    Args:
        current_user: Authenticated user.

    Returns:
        User information from token.
    """
    return {
        "id": current_user.id,
        "tenant_id": current_user.tenant_id,
        "tenant_code": current_user.tenant_code,
        "user_type": current_user.user_type,
        "roles": current_user.roles,
        "permissions": current_user.permissions,
        "school_ids": current_user.school_ids,
    }


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List active sessions",
    description="Get list of all active sessions for the current user.",
)
async def list_sessions(
    request: Request,
    current_user: CurrentUser = Depends(require_auth),
    db: AsyncSession = Depends(get_tenant_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
) -> SessionListResponse:
    """List active sessions for current user.

    Args:
        request: HTTP request.
        current_user: Authenticated user.
        db: Database session.
        jwt_manager: JWT manager.

    Returns:
        List of active sessions.
    """
    auth_service = AuthService(db, jwt_manager)
    sessions = await auth_service.get_active_sessions(current_user.id)

    # Mark current session
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else None

    if token:
        current_session = await auth_service.validate_session(token)
        if current_session:
            for session in sessions:
                if str(session.id) == current_session.id:
                    session.is_current = True
                    break

    return SessionListResponse(
        sessions=sessions,
        total=len(sessions),
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke session",
    description="Revoke a specific session by ID.",
)
async def revoke_session(
    session_id: UUID,
    current_user: CurrentUser = Depends(require_auth),
    db: AsyncSession = Depends(get_tenant_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
) -> None:
    """Revoke a specific session.

    Args:
        session_id: Session ID to revoke.
        current_user: Authenticated user.
        db: Database session.
        jwt_manager: JWT manager.

    Raises:
        HTTPException: If session not found.
    """
    auth_service = AuthService(db, jwt_manager)
    revoked = await auth_service.revoke_session(
        session_id=str(session_id),
        user_id=current_user.id,
    )

    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
