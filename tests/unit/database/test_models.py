# Copyright (C) 2025 Global Digital Labs (gdlabs.io)
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for database models.

Tests model definitions, relationships, and helper methods.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.infrastructure.database.models.base import Base, SoftDeleteMixin, TimestampMixin
from src.infrastructure.database.models.central import (
    License,
    SystemAuditLog,
    SystemSession,
    SystemUser,
    Tenant,
    TenantFeatureFlag,
)
from src.infrastructure.database.models.tenant import (
    AuditLog,
    Conversation,
    FeatureFlag,
    Language,
    Permission,
    PracticeSession,
    ReviewItem,
    Role,
    TenantSetting,
    Topic,
    User,
)


class TestBase:
    """Test base model functionality."""

    def test_base_inherits_declarative_base(self):
        """Verify Base is a declarative base."""
        assert hasattr(Base, "metadata")
        assert hasattr(Base, "registry")

    def test_timestamp_mixin_has_created_at(self):
        """Verify TimestampMixin has created_at field."""
        assert hasattr(TimestampMixin, "created_at")
        assert hasattr(TimestampMixin, "updated_at")

    def test_soft_delete_mixin_has_deleted_at(self):
        """Verify SoftDeleteMixin has deleted_at field."""
        assert hasattr(SoftDeleteMixin, "deleted_at")


class TestCentralModels:
    """Test central database models."""

    def test_license_model_exists(self):
        """Verify License model has required attributes."""
        assert hasattr(License, "__tablename__")
        assert License.__tablename__ == "licenses"
        assert hasattr(License, "id")
        assert hasattr(License, "license_key")
        assert hasattr(License, "license_type")
        assert hasattr(License, "max_students")
        assert hasattr(License, "features")

    def test_license_is_valid_property(self):
        """Test License.is_valid property."""
        now = datetime.now(timezone.utc)
        license_obj = License(
            license_key="TEST-001",
            license_type="trial",
            max_students=50,
            max_teachers=5,
            features={},
            valid_from=now - timedelta(days=1),
            valid_until=now + timedelta(days=30),
            is_active=True,
        )
        assert license_obj.is_valid is True

        license_obj.is_active = False
        assert license_obj.is_valid is False

    def test_license_days_remaining_property(self):
        """Test License.days_remaining property."""
        now = datetime.now(timezone.utc)
        license_obj = License(
            license_key="TEST-002",
            license_type="trial",
            max_students=50,
            max_teachers=5,
            features={},
            valid_from=now,
            valid_until=now + timedelta(days=15),
            is_active=True,
        )
        assert 14 <= license_obj.days_remaining <= 15

    def test_tenant_model_exists(self):
        """Verify Tenant model has required attributes."""
        assert hasattr(Tenant, "__tablename__")
        assert Tenant.__tablename__ == "tenants"
        assert hasattr(Tenant, "id")
        assert hasattr(Tenant, "name")
        assert hasattr(Tenant, "slug")
        assert hasattr(Tenant, "db_host")
        assert hasattr(Tenant, "container_id")

    def test_tenant_connection_string_property(self):
        """Test Tenant.connection_string property."""
        tenant = Tenant(
            name="Test Tenant",
            slug="test-tenant",
            license_id="fake-uuid",
            db_host="localhost",
            db_port=5432,
            db_name="test_db",
            db_user="test_user",
            db_password_encrypted="encrypted",
        )
        conn_str = tenant.connection_string
        assert "localhost" in conn_str
        assert "5432" in conn_str
        assert "test_db" in conn_str

    def test_system_user_model_exists(self):
        """Verify SystemUser model has required attributes."""
        assert hasattr(SystemUser, "__tablename__")
        assert SystemUser.__tablename__ == "system_users"
        assert hasattr(SystemUser, "id")
        assert hasattr(SystemUser, "email")
        assert hasattr(SystemUser, "password_hash")
        assert hasattr(SystemUser, "mfa_enabled")

    def test_system_user_is_locked_property(self):
        """Test SystemUser.is_locked property."""
        user = SystemUser(
            email="test@example.com",
            password_hash="hash",
            full_name="Test User",
            role="admin",
        )
        assert user.is_locked is False

        user.locked_until = datetime.now(timezone.utc) + timedelta(hours=1)
        assert user.is_locked is True

    def test_system_session_model_exists(self):
        """Verify SystemSession model has required attributes."""
        assert hasattr(SystemSession, "__tablename__")
        assert SystemSession.__tablename__ == "system_sessions"
        assert hasattr(SystemSession, "token_hash")
        assert hasattr(SystemSession, "expires_at")

    def test_system_audit_log_model_exists(self):
        """Verify SystemAuditLog model has required attributes."""
        assert hasattr(SystemAuditLog, "__tablename__")
        assert SystemAuditLog.__tablename__ == "system_audit_logs"
        assert hasattr(SystemAuditLog, "action")
        assert hasattr(SystemAuditLog, "entity_type")

    def test_tenant_feature_flag_model_exists(self):
        """Verify TenantFeatureFlag model has required attributes."""
        assert hasattr(TenantFeatureFlag, "__tablename__")
        assert TenantFeatureFlag.__tablename__ == "tenant_feature_flags"
        assert hasattr(TenantFeatureFlag, "feature_key")
        assert hasattr(TenantFeatureFlag, "is_enabled")


class TestTenantModels:
    """Test tenant database models."""

    def test_user_model_exists(self):
        """Verify User model has required attributes."""
        assert hasattr(User, "__tablename__")
        assert User.__tablename__ == "users"
        assert hasattr(User, "id")
        assert hasattr(User, "email")
        assert hasattr(User, "first_name")
        assert hasattr(User, "user_type")

    def test_user_full_name_property(self):
        """Test User.full_name property."""
        user = User(
            email="test@example.com",
            first_name="John",
            last_name="Doe",
            user_type="student",
        )
        assert user.full_name == "John Doe"

    def test_role_model_exists(self):
        """Verify Role model has required attributes."""
        assert hasattr(Role, "__tablename__")
        assert Role.__tablename__ == "roles"
        assert hasattr(Role, "name")
        assert hasattr(Role, "display_name")

    def test_permission_model_exists(self):
        """Verify Permission model has required attributes."""
        assert hasattr(Permission, "__tablename__")
        assert Permission.__tablename__ == "permissions"
        assert hasattr(Permission, "code")
        assert hasattr(Permission, "category")

    def test_topic_model_exists(self):
        """Verify Topic model has required attributes."""
        assert hasattr(Topic, "__tablename__")
        assert Topic.__tablename__ == "topics"
        assert hasattr(Topic, "name")
        assert hasattr(Topic, "code")
        assert hasattr(Topic, "difficulty_level")

    def test_practice_session_model_exists(self):
        """Verify PracticeSession model has required attributes."""
        assert hasattr(PracticeSession, "__tablename__")
        assert PracticeSession.__tablename__ == "practice_sessions"
        assert hasattr(PracticeSession, "student_id")
        assert hasattr(PracticeSession, "session_type")
        assert hasattr(PracticeSession, "status")

    def test_practice_session_accuracy_property(self):
        """Test PracticeSession.accuracy property."""
        session = PracticeSession(
            student_id="fake-uuid",
            session_type="practice",
            total_questions=10,
            correct_answers=7,
        )
        assert session.accuracy == 70.0

        session.total_questions = 0
        assert session.accuracy == 0.0

    def test_conversation_model_exists(self):
        """Verify Conversation model has required attributes."""
        assert hasattr(Conversation, "__tablename__")
        assert Conversation.__tablename__ == "conversations"
        assert hasattr(Conversation, "student_id")
        assert hasattr(Conversation, "persona")
        assert hasattr(Conversation, "status")

    def test_review_item_model_exists(self):
        """Verify ReviewItem model has required attributes (FSRS-5)."""
        assert hasattr(ReviewItem, "__tablename__")
        assert ReviewItem.__tablename__ == "review_items"
        assert hasattr(ReviewItem, "stability")
        assert hasattr(ReviewItem, "difficulty")
        assert hasattr(ReviewItem, "reps")
        assert hasattr(ReviewItem, "lapses")
        assert hasattr(ReviewItem, "state")
        assert hasattr(ReviewItem, "due")

    def test_review_item_is_due_property(self):
        """Test ReviewItem.is_due property."""
        now = datetime.now(timezone.utc)
        item = ReviewItem(
            student_id="fake-uuid",
            item_type="topic",
            item_id="fake-topic-uuid",
            due=now - timedelta(hours=1),
        )
        assert item.is_due is True

        item.due = now + timedelta(hours=1)
        assert item.is_due is False

    def test_language_model_exists(self):
        """Verify Language model has required attributes."""
        assert hasattr(Language, "__tablename__")
        assert Language.__tablename__ == "languages"
        assert hasattr(Language, "code")
        assert hasattr(Language, "name")
        assert hasattr(Language, "is_rtl")

    def test_language_display_name_property(self):
        """Test Language.display_name property."""
        lang = Language(
            code="tr",
            name="Turkish",
            native_name="Türkçe",
        )
        assert lang.display_name == "Türkçe"

        lang.native_name = None
        assert lang.display_name == "Turkish"

    def test_tenant_setting_model_exists(self):
        """Verify TenantSetting model has required attributes."""
        assert hasattr(TenantSetting, "__tablename__")
        assert TenantSetting.__tablename__ == "tenant_settings"
        assert hasattr(TenantSetting, "setting_key")
        assert hasattr(TenantSetting, "setting_value")

    def test_tenant_setting_get_value_method(self):
        """Test TenantSetting.get_value method."""
        setting = TenantSetting(
            setting_key="test.setting",
            setting_value={"value": "test_value"},
        )
        assert setting.get_value() == "test_value"

        setting.setting_value = {"other": "data"}
        assert setting.get_value() == {"other": "data"}

    def test_feature_flag_model_exists(self):
        """Verify FeatureFlag model has required attributes."""
        assert hasattr(FeatureFlag, "__tablename__")
        assert FeatureFlag.__tablename__ == "feature_flags"
        assert hasattr(FeatureFlag, "feature_key")
        assert hasattr(FeatureFlag, "is_enabled")
        assert hasattr(FeatureFlag, "rollout_percentage")

    def test_feature_flag_is_enabled_for_user_method(self):
        """Test FeatureFlag.is_enabled_for_user method."""
        flag = FeatureFlag(
            feature_key="test_feature",
            is_enabled=True,
            rollout_percentage=100,
            user_ids=[],
        )
        assert flag.is_enabled_for_user("any-user-id") is True

        flag.is_enabled = False
        assert flag.is_enabled_for_user("any-user-id") is False

        flag.is_enabled = True
        flag.rollout_percentage = 0
        flag.user_ids = ["specific-user"]
        assert flag.is_enabled_for_user("specific-user") is True
        assert flag.is_enabled_for_user("other-user") is False

    def test_audit_log_model_exists(self):
        """Verify AuditLog model has required attributes."""
        assert hasattr(AuditLog, "__tablename__")
        assert AuditLog.__tablename__ == "audit_logs"
        assert hasattr(AuditLog, "action")
        assert hasattr(AuditLog, "user_id")
        assert hasattr(AuditLog, "ip_address")

    def test_audit_log_log_action_classmethod(self):
        """Test AuditLog.log_action classmethod."""
        log = AuditLog.log_action(
            action="user.create",
            user_id="fake-uuid",
            user_email="test@example.com",
            entity_type="user",
            entity_id="new-user-uuid",
            new_values={"email": "new@example.com"},
            success=True,
        )
        assert log.action == "user.create"
        assert log.entity_type == "user"
        assert log.success is True


class TestModelTableNames:
    """Test that all models have correct table names."""

    def test_central_table_names(self):
        """Verify central database table names."""
        assert License.__tablename__ == "licenses"
        assert Tenant.__tablename__ == "tenants"
        assert SystemUser.__tablename__ == "system_users"
        assert SystemSession.__tablename__ == "system_sessions"
        assert SystemAuditLog.__tablename__ == "system_audit_logs"
        assert TenantFeatureFlag.__tablename__ == "tenant_feature_flags"

    def test_tenant_user_table_names(self):
        """Verify tenant user management table names."""
        from src.infrastructure.database.models.tenant import (
            Permission,
            Role,
            RolePermission,
            User,
            UserRole,
        )

        assert User.__tablename__ == "users"
        assert Role.__tablename__ == "roles"
        assert Permission.__tablename__ == "permissions"
        assert RolePermission.__tablename__ == "role_permissions"
        assert UserRole.__tablename__ == "user_roles"

    def test_tenant_session_table_names(self):
        """Verify tenant session management table names."""
        from src.infrastructure.database.models.tenant import (
            EmailVerification,
            PasswordResetToken,
            RefreshToken,
            UserSession,
        )

        assert UserSession.__tablename__ == "user_sessions"
        assert RefreshToken.__tablename__ == "refresh_tokens"
        assert PasswordResetToken.__tablename__ == "password_reset_tokens"
        assert EmailVerification.__tablename__ == "email_verifications"

    def test_tenant_curriculum_table_names(self):
        """Verify tenant curriculum table names."""
        from src.infrastructure.database.models.tenant import (
            Curriculum,
            GradeLevel,
            KnowledgeComponent,
            LearningObjective,
            Prerequisite,
            Subject,
            Topic,
            Unit,
        )

        assert Curriculum.__tablename__ == "curricula"
        assert GradeLevel.__tablename__ == "grade_levels"
        assert Subject.__tablename__ == "subjects"
        assert Unit.__tablename__ == "units"
        assert Topic.__tablename__ == "topics"
        assert LearningObjective.__tablename__ == "learning_objectives"
        assert KnowledgeComponent.__tablename__ == "knowledge_components"
        assert Prerequisite.__tablename__ == "prerequisites"

    def test_tenant_memory_table_names(self):
        """Verify tenant memory system table names."""
        from src.infrastructure.database.models.tenant import (
            AssociativeMemory,
            EpisodicMemory,
            ProceduralMemory,
            SemanticMemory,
        )

        assert EpisodicMemory.__tablename__ == "episodic_memories"
        assert SemanticMemory.__tablename__ == "semantic_memories"
        assert ProceduralMemory.__tablename__ == "procedural_memories"
        assert AssociativeMemory.__tablename__ == "associative_memories"
