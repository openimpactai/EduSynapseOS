# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""System admin authentication endpoints.

This module provides authentication endpoints for platform administrators:
- POST /login - Authenticate system admin
- POST /refresh - Refresh access token
- POST /logout - Logout current session
- GET /me - Get current admin profile
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_central_db, get_jwt_manager, get_password_hasher
from src.domains.auth.jwt import JWTManager
from src.domains.auth.password import PasswordHasher
from src.domains.system.auth_service import (
    SystemAuthService,
    SystemInvalidCredentialsError,
    SystemAccountLockedError,
    SystemAccountInactiveError,
    SystemTokenRefreshError,
    DeviceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class SystemLoginRequest(BaseModel):
    """System admin login request."""

    email: EmailStr = Field(..., description="Admin email address")
    password: str = Field(..., min_length=1, description="Password")


class SystemLoginResponse(BaseModel):
    """System admin login response."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")
    user: "SystemUserResponse" = Field(..., description="Authenticated user")


class SystemUserResponse(BaseModel):
    """System user profile response."""

    id: str = Field(..., description="User ID")
    email: str = Field(..., description="Email address")
    name: str = Field(..., description="Display name")
    role: str = Field(..., description="Admin role (admin, super_admin)")
    is_active: bool = Field(..., description="Account status")
    mfa_enabled: bool = Field(..., description="MFA status")
    last_login_at: str | None = Field(None, description="Last login timestamp")

    class Config:
        from_attributes = True


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str = Field(..., description="Current refresh token")


class TokenResponse(BaseModel):
    """Token response after refresh."""

    access_token: str = Field(..., description="New JWT access token")
    refresh_token: str = Field(..., description="New JWT refresh token")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")


def _get_device_info(request: Request) -> DeviceInfo:
    """Extract device info from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else None

    return DeviceInfo(
        ip_address=ip,
        user_agent=request.headers.get("User-Agent"),
    )


async def get_system_auth_service(
    db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
) -> SystemAuthService:
    """Get SystemAuthService instance."""
    return SystemAuthService(db, jwt_manager, password_hasher)


@router.post(
    "/login",
    response_model=SystemLoginResponse,
    summary="System admin login",
    description="Authenticate a system administrator and get access tokens.",
)
async def login(
    data: SystemLoginRequest,
    request: Request,
    auth_service: SystemAuthService = Depends(get_system_auth_service),
) -> SystemLoginResponse:
    """Authenticate system admin.

    Args:
        data: Login credentials.
        request: HTTP request for device info.
        auth_service: System auth service.

    Returns:
        SystemLoginResponse with tokens and user info.

    Raises:
        HTTPException: If authentication fails.
    """
    device_info = _get_device_info(request)

    try:
        result = await auth_service.login(
            email=data.email,
            password=data.password,
            device_info=device_info,
        )

        return SystemLoginResponse(
            access_token=result.tokens.access_token,
            refresh_token=result.tokens.refresh_token,
            token_type="Bearer",
            expires_in=result.tokens.expires_in,
            user=SystemUserResponse(
                id=result.user.id,
                email=result.user.email,
                name=result.user.name,
                role=result.user.role,
                is_active=result.user.is_active,
                mfa_enabled=result.user.mfa_enabled,
                last_login_at=(
                    result.user.last_login_at.isoformat()
                    if result.user.last_login_at
                    else None
                ),
            ),
        )

    except SystemInvalidCredentialsError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    except SystemAccountLockedError as e:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=str(e),
        )
    except SystemAccountInactiveError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is not active",
        )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh tokens",
    description="Get new access token using refresh token.",
)
async def refresh_tokens(
    data: RefreshTokenRequest,
    auth_service: SystemAuthService = Depends(get_system_auth_service),
) -> TokenResponse:
    """Refresh access token.

    Args:
        data: Refresh token request.
        auth_service: System auth service.

    Returns:
        TokenResponse with new tokens.

    Raises:
        HTTPException: If refresh fails.
    """
    try:
        tokens = await auth_service.refresh_tokens(data.refresh_token)

        return TokenResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type="Bearer",
            expires_in=tokens.expires_in,
        )

    except SystemTokenRefreshError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Logout",
    description="Revoke current session.",
)
async def logout(
    request: Request,
    db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
) -> None:
    """Logout current session.

    Args:
        request: HTTP request with Authorization header.
        db: Central database session.
        jwt_manager: JWT manager.
        password_hasher: Password hasher.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return

    token = auth_header.split(" ")[1]
    auth_service = SystemAuthService(db, jwt_manager, password_hasher)

    session = await auth_service.validate_session(token)
    if session:
        await auth_service.logout(session.id)


@router.get(
    "/me",
    response_model=SystemUserResponse,
    summary="Get current user",
    description="Get the currently authenticated system admin profile.",
)
async def get_me(
    request: Request,
    db: AsyncSession = Depends(get_central_db),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    password_hasher: PasswordHasher = Depends(get_password_hasher),
) -> SystemUserResponse:
    """Get current admin profile.

    Args:
        request: HTTP request with Authorization header.
        db: Central database session.
        jwt_manager: JWT manager.
        password_hasher: Password hasher.

    Returns:
        SystemUserResponse with user profile.

    Raises:
        HTTPException: If not authenticated.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header.split(" ")[1]
    auth_service = SystemAuthService(db, jwt_manager, password_hasher)

    user = await auth_service.get_current_user(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return SystemUserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        is_active=user.is_active,
        mfa_enabled=user.mfa_enabled,
        last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
    )
