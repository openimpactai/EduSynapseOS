-- ==============================================================================
-- EduSynapseOS Tenant Database Initialization Script
-- ==============================================================================
-- Reference script for tenant database schema. The actual schema is created
-- programmatically via SQLAlchemy models in TenantService._initialize_tenant_schema().
-- This script serves as documentation and fallback reference.
--
-- Source: src/infrastructure/database/models/tenant/*.py
-- Last Updated: 2026-01-08
-- ==============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==============================================================================
-- USER MANAGEMENT TABLES
-- ==============================================================================

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255),
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    display_name VARCHAR(200),
    avatar_url TEXT,
    user_type VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    email_verified_at TIMESTAMPTZ,
    mfa_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    mfa_secret_encrypted BYTEA,
    failed_login_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TIMESTAMPTZ,
    password_changed_at TIMESTAMPTZ,
    must_change_password BOOLEAN NOT NULL DEFAULT FALSE,
    sso_provider VARCHAR(50),
    sso_external_id VARCHAR(255),
    preferred_language VARCHAR(10) NOT NULL DEFAULT 'tr',
    timezone VARCHAR(50) NOT NULL DEFAULT 'Europe/Istanbul',
    last_login_at TIMESTAMPTZ,
    last_activity_at TIMESTAMPTZ,
    extra_data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    CONSTRAINT unique_email_active UNIQUE (email),
    CONSTRAINT valid_user_type CHECK (user_type IN ('student', 'teacher', 'parent', 'school_admin', 'tenant_admin')),
    CONSTRAINT valid_status CHECK (status IN ('pending', 'active', 'suspended', 'archived'))
);
CREATE INDEX IF NOT EXISTS ix_users_email ON users(email);
CREATE INDEX IF NOT EXISTS ix_users_user_type ON users(user_type);
CREATE INDEX IF NOT EXISTS ix_users_status ON users(status);

-- Roles table
CREATE TABLE IF NOT EXISTS roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    level INTEGER NOT NULL DEFAULT 0,
    is_system BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Permissions table
CREATE TABLE IF NOT EXISTS permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    category VARCHAR(50),
    is_system BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_permissions_category ON permissions(category);

-- Role permissions junction table
CREATE TABLE IF NOT EXISTS role_permissions (
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    permission_id UUID NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (role_id, permission_id)
);

-- ==============================================================================
-- SESSION MANAGEMENT TABLES
-- ==============================================================================

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    access_token_hash VARCHAR(64) NOT NULL,
    device_type VARCHAR(20),
    device_name VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    country_code VARCHAR(2),
    city VARCHAR(100),
    expires_at TIMESTAMPTZ NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    revoked_at TIMESTAMPTZ,
    revoked_reason VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_activity_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS ix_user_sessions_access_token_hash ON user_sessions(access_token_hash);
CREATE INDEX IF NOT EXISTS ix_user_sessions_expires_at ON user_sessions(expires_at);

-- Refresh tokens table
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES user_sessions(id) ON DELETE CASCADE,
    token_hash VARCHAR(64) NOT NULL,
    family_id UUID NOT NULL,
    generation INTEGER NOT NULL DEFAULT 1,
    expires_at TIMESTAMPTZ NOT NULL,
    used_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_refresh_tokens_user_id ON refresh_tokens(user_id);
CREATE INDEX IF NOT EXISTS ix_refresh_tokens_token_hash ON refresh_tokens(token_hash);
CREATE INDEX IF NOT EXISTS ix_refresh_tokens_family_id ON refresh_tokens(family_id);

-- Email verifications table
CREATE TABLE IF NOT EXISTS email_verifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    token_hash VARCHAR(64) NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    verified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_email_verifications_user_id ON email_verifications(user_id);
CREATE INDEX IF NOT EXISTS ix_email_verifications_token_hash ON email_verifications(token_hash);

-- ==============================================================================
-- SCHOOL STRUCTURE TABLES
-- ==============================================================================

-- Schools table
CREATE TABLE IF NOT EXISTS schools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    school_type VARCHAR(50),
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state_province VARCHAR(100),
    postal_code VARCHAR(20),
    country_code VARCHAR(2) NOT NULL DEFAULT 'TR',
    phone VARCHAR(50),
    email VARCHAR(255),
    website VARCHAR(255),
    settings JSONB NOT NULL DEFAULT '{}',
    timezone VARCHAR(50) NOT NULL DEFAULT 'Europe/Istanbul',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at TIMESTAMPTZ,
    CONSTRAINT unique_school_code UNIQUE (code)
);

-- Academic years table
CREATE TABLE IF NOT EXISTS academic_years (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(50) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    is_current BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ==============================================================================
-- CURRICULUM TABLES
-- ==============================================================================

-- Curricula table
CREATE TABLE IF NOT EXISTS curricula (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    country_code VARCHAR(2),
    organization VARCHAR(100),
    version VARCHAR(20),
    year INTEGER,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    extra_data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Grade levels table
CREATE TABLE IF NOT EXISTS grade_levels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    curriculum_id UUID NOT NULL REFERENCES curricula(id) ON DELETE CASCADE,
    code VARCHAR(20) NOT NULL,
    name VARCHAR(100) NOT NULL,
    sequence INTEGER NOT NULL,
    min_age INTEGER,
    max_age INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_grade_curriculum UNIQUE (curriculum_id, code)
);
CREATE INDEX IF NOT EXISTS ix_grade_levels_curriculum_id ON grade_levels(curriculum_id);

-- Subjects table
CREATE TABLE IF NOT EXISTS subjects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    curriculum_id UUID NOT NULL REFERENCES curricula(id) ON DELETE CASCADE,
    code VARCHAR(50) NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    icon VARCHAR(50),
    color VARCHAR(7),
    sequence INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_subject_curriculum UNIQUE (curriculum_id, code)
);
CREATE INDEX IF NOT EXISTS ix_subjects_curriculum_id ON subjects(curriculum_id);

-- Classes table (depends on schools, academic_years, grade_levels)
CREATE TABLE IF NOT EXISTS classes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    school_id UUID NOT NULL REFERENCES schools(id) ON DELETE CASCADE,
    academic_year_id UUID NOT NULL REFERENCES academic_years(id),
    code VARCHAR(50) NOT NULL,
    name VARCHAR(100) NOT NULL,
    grade_level_id UUID REFERENCES grade_levels(id),
    max_students INTEGER,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_class_code_year UNIQUE (school_id, academic_year_id, code)
);
CREATE INDEX IF NOT EXISTS ix_classes_school_id ON classes(school_id);
CREATE INDEX IF NOT EXISTS ix_classes_academic_year_id ON classes(academic_year_id);

-- Units table
CREATE TABLE IF NOT EXISTS units (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subject_id UUID NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    grade_level_id UUID NOT NULL REFERENCES grade_levels(id) ON DELETE CASCADE,
    code VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    sequence INTEGER NOT NULL,
    estimated_hours INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_unit UNIQUE (subject_id, grade_level_id, code)
);
CREATE INDEX IF NOT EXISTS ix_units_subject_id ON units(subject_id);
CREATE INDEX IF NOT EXISTS ix_units_grade_level_id ON units(grade_level_id);

-- Topics table
CREATE TABLE IF NOT EXISTS topics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    unit_id UUID NOT NULL REFERENCES units(id) ON DELETE CASCADE,
    code VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    sequence INTEGER NOT NULL,
    base_difficulty NUMERIC(3,2) NOT NULL DEFAULT 0.50,
    estimated_minutes INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_topic UNIQUE (unit_id, code)
);
CREATE INDEX IF NOT EXISTS ix_topics_unit_id ON topics(unit_id);

-- Learning objectives table
CREATE TABLE IF NOT EXISTS learning_objectives (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic_id UUID NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    code VARCHAR(50) NOT NULL,
    objective TEXT NOT NULL,
    bloom_level VARCHAR(20),
    sequence INTEGER NOT NULL,
    mastery_threshold NUMERIC(3,2) NOT NULL DEFAULT 0.80,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_objective UNIQUE (topic_id, code),
    CONSTRAINT valid_bloom_level CHECK (bloom_level IN ('remember', 'understand', 'apply', 'analyze', 'evaluate', 'create'))
);
CREATE INDEX IF NOT EXISTS ix_learning_objectives_topic_id ON learning_objectives(topic_id);

-- Knowledge components table
CREATE TABLE IF NOT EXISTS knowledge_components (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    learning_objective_id UUID NOT NULL REFERENCES learning_objectives(id) ON DELETE CASCADE,
    code VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    component_type VARCHAR(20) NOT NULL DEFAULT 'concept',
    difficulty NUMERIC(3,2) NOT NULL DEFAULT 0.50,
    sequence INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_kc UNIQUE (learning_objective_id, code),
    CONSTRAINT valid_component_type CHECK (component_type IN ('concept', 'skill', 'fact', 'procedure'))
);
CREATE INDEX IF NOT EXISTS ix_knowledge_components_learning_objective_id ON knowledge_components(learning_objective_id);

-- Prerequisites table
CREATE TABLE IF NOT EXISTS prerequisites (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type VARCHAR(20) NOT NULL,
    source_id UUID NOT NULL,
    target_type VARCHAR(20) NOT NULL,
    target_id UUID NOT NULL,
    strength NUMERIC(3,2) NOT NULL DEFAULT 1.00,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_prerequisite UNIQUE (source_type, source_id, target_type, target_id),
    CONSTRAINT valid_source_type CHECK (source_type IN ('topic', 'objective', 'kc')),
    CONSTRAINT valid_target_type CHECK (target_type IN ('topic', 'objective', 'kc'))
);

-- ==============================================================================
-- USER-ROLE ASSIGNMENT TABLES (depends on schools, classes)
-- ==============================================================================

-- User roles table
CREATE TABLE IF NOT EXISTS user_roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    school_id UUID REFERENCES schools(id) ON DELETE CASCADE,
    class_id UUID REFERENCES classes(id) ON DELETE CASCADE,
    valid_from TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_until TIMESTAMPTZ,
    granted_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_user_role_scope UNIQUE (user_id, role_id, school_id, class_id)
);
CREATE INDEX IF NOT EXISTS ix_user_roles_user_id ON user_roles(user_id);

-- Class students table
CREATE TABLE IF NOT EXISTS class_students (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    class_id UUID NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    enrolled_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    withdrawn_at TIMESTAMPTZ,
    student_number VARCHAR(20),
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    CONSTRAINT unique_student_class UNIQUE (class_id, student_id),
    CONSTRAINT valid_status CHECK (status IN ('active', 'withdrawn'))
);
CREATE INDEX IF NOT EXISTS ix_class_students_class_id ON class_students(class_id);
CREATE INDEX IF NOT EXISTS ix_class_students_student_id ON class_students(student_id);

-- Class teachers table
CREATE TABLE IF NOT EXISTS class_teachers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    class_id UUID NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    teacher_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    subject_id UUID REFERENCES subjects(id),
    is_homeroom BOOLEAN NOT NULL DEFAULT FALSE,
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ,
    CONSTRAINT unique_teacher_class_subject UNIQUE (class_id, teacher_id, subject_id)
);
CREATE INDEX IF NOT EXISTS ix_class_teachers_class_id ON class_teachers(class_id);
CREATE INDEX IF NOT EXISTS ix_class_teachers_teacher_id ON class_teachers(teacher_id);

-- Parent student relations table
CREATE TABLE IF NOT EXISTS parent_student_relations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    parent_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    relationship_type VARCHAR(20) NOT NULL DEFAULT 'parent',
    can_view_progress BOOLEAN NOT NULL DEFAULT TRUE,
    can_view_conversations BOOLEAN NOT NULL DEFAULT FALSE,
    can_receive_notifications BOOLEAN NOT NULL DEFAULT TRUE,
    can_chat_with_ai BOOLEAN NOT NULL DEFAULT TRUE,
    is_primary BOOLEAN NOT NULL DEFAULT FALSE,
    verified_at TIMESTAMPTZ,
    verified_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_parent_student UNIQUE (parent_id, student_id),
    CONSTRAINT valid_relationship_type CHECK (relationship_type IN ('parent', 'guardian', 'other'))
);
CREATE INDEX IF NOT EXISTS ix_parent_student_relations_parent_id ON parent_student_relations(parent_id);
CREATE INDEX IF NOT EXISTS ix_parent_student_relations_student_id ON parent_student_relations(student_id);

-- ==============================================================================
-- PRACTICE & ASSESSMENT TABLES
-- ==============================================================================

-- Practice sessions table
CREATE TABLE IF NOT EXISTS practice_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    topic_id UUID REFERENCES topics(id),
    session_type VARCHAR(30) NOT NULL DEFAULT 'quick',
    persona_id VARCHAR(50),
    questions_total INTEGER NOT NULL DEFAULT 0,
    questions_answered INTEGER NOT NULL DEFAULT 0,
    questions_correct INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ,
    time_spent_seconds INTEGER NOT NULL DEFAULT 0,
    score NUMERIC(5,2),
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    summary TEXT,
    checkpoint_data JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT valid_session_status CHECK (status IN ('active', 'completed', 'abandoned', 'paused')),
    CONSTRAINT valid_session_type CHECK (session_type IN ('quick', 'focused', 'review', 'assessment', 'diagnostic'))
);
CREATE INDEX IF NOT EXISTS ix_practice_sessions_student_id ON practice_sessions(student_id);
CREATE INDEX IF NOT EXISTS ix_practice_sessions_topic_id ON practice_sessions(topic_id);
CREATE INDEX IF NOT EXISTS ix_practice_sessions_status ON practice_sessions(status);
CREATE INDEX IF NOT EXISTS ix_practice_sessions_started_at ON practice_sessions(started_at);

-- Practice questions table
CREATE TABLE IF NOT EXISTS practice_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES practice_sessions(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    content TEXT NOT NULL,
    display_hint VARCHAR(30),
    data JSONB NOT NULL DEFAULT '{}',
    correct_answer JSONB NOT NULL,
    explanation TEXT,
    evaluation_config JSONB NOT NULL DEFAULT '{}',
    topic_id UUID REFERENCES topics(id),
    learning_objective_id UUID REFERENCES learning_objectives(id),
    knowledge_component_id UUID REFERENCES knowledge_components(id),
    difficulty NUMERIC(3,2),
    bloom_level VARCHAR(20),
    hints JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_practice_questions_session_id ON practice_questions(session_id);

-- Student answers table
CREATE TABLE IF NOT EXISTS student_answers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question_id UUID NOT NULL REFERENCES practice_questions(id) ON DELETE CASCADE,
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    answer JSONB NOT NULL,
    time_spent_seconds INTEGER NOT NULL DEFAULT 0,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    hints_viewed INTEGER NOT NULL DEFAULT 0,
    attempt_number INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_student_answers_question_id ON student_answers(question_id);
CREATE INDEX IF NOT EXISTS ix_student_answers_student_id ON student_answers(student_id);

-- Evaluation results table
CREATE TABLE IF NOT EXISTS evaluation_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    answer_id UUID UNIQUE NOT NULL REFERENCES student_answers(id) ON DELETE CASCADE,
    is_correct BOOLEAN NOT NULL,
    score NUMERIC(5,2) NOT NULL,
    feedback TEXT,
    detailed_feedback JSONB,
    evaluation_method VARCHAR(20) NOT NULL,
    confidence NUMERIC(3,2),
    reasoning TEXT,
    misconceptions JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT valid_evaluation_method CHECK (evaluation_method IN ('exact', 'semantic', 'hybrid', 'llm'))
);
CREATE INDEX IF NOT EXISTS ix_evaluation_results_answer_id ON evaluation_results(answer_id);

-- Assessment sessions table
CREATE TABLE IF NOT EXISTS assessment_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    assessment_type VARCHAR(30) NOT NULL,
    topic_id UUID REFERENCES topics(id),
    config JSONB NOT NULL DEFAULT '{}',
    total_score NUMERIC(5,2),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT valid_assessment_status CHECK (status IN ('pending', 'in_progress', 'completed', 'cancelled')),
    CONSTRAINT valid_assessment_type CHECK (assessment_type IN ('diagnostic', 'formative', 'summative', 'placement'))
);
CREATE INDEX IF NOT EXISTS ix_assessment_sessions_student_id ON assessment_sessions(student_id);
CREATE INDEX IF NOT EXISTS ix_assessment_sessions_status ON assessment_sessions(status);

-- Assessment results table
CREATE TABLE IF NOT EXISTS assessment_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES assessment_sessions(id) ON DELETE CASCADE,
    knowledge_component_id UUID REFERENCES knowledge_components(id),
    mastery_level NUMERIC(3,2),
    correct_count INTEGER NOT NULL DEFAULT 0,
    total_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_assessment_results_session_id ON assessment_results(session_id);

-- ==============================================================================
-- CONVERSATION TABLES
-- ==============================================================================

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    conversation_type VARCHAR(30) NOT NULL DEFAULT 'tutoring',
    topic_id UUID REFERENCES topics(id),
    persona_id VARCHAR(50),
    title VARCHAR(255),
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    message_count INTEGER NOT NULL DEFAULT 0,
    last_message_at TIMESTAMPTZ,
    workflow_state JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS ix_conversations_topic_id ON conversations(topic_id);
CREATE INDEX IF NOT EXISTS ix_conversations_status ON conversations(status);

-- Conversation messages table
CREATE TABLE IF NOT EXISTS conversation_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    attachments JSONB NOT NULL DEFAULT '[]',
    tokens_input INTEGER,
    tokens_output INTEGER,
    model VARCHAR(100),
    parent_id UUID REFERENCES conversation_messages(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_conversation_messages_conversation_id ON conversation_messages(conversation_id);
CREATE INDEX IF NOT EXISTS ix_conversation_messages_created_at ON conversation_messages(created_at);

-- Conversation summaries table
CREATE TABLE IF NOT EXISTS conversation_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    summary_type VARCHAR(20) NOT NULL,
    summary TEXT NOT NULL,
    key_points JSONB NOT NULL DEFAULT '[]',
    messages_from_id UUID REFERENCES conversation_messages(id),
    messages_to_id UUID REFERENCES conversation_messages(id),
    message_count INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_conversation_summaries_conversation_id ON conversation_summaries(conversation_id);

-- ==============================================================================
-- MEMORY TABLES
-- ==============================================================================

-- Episodic memories table
CREATE TABLE IF NOT EXISTS episodic_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    summary TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}',
    emotional_state VARCHAR(30),
    importance NUMERIC(3,2) NOT NULL DEFAULT 0.50,
    session_id UUID,
    conversation_id UUID,
    topic_id UUID REFERENCES topics(id),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_accessed_at TIMESTAMPTZ,
    access_count INTEGER NOT NULL DEFAULT 0,
    embedding_id VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_episodic_memories_student_id ON episodic_memories(student_id);
CREATE INDEX IF NOT EXISTS ix_episodic_memories_event_type ON episodic_memories(event_type);
CREATE INDEX IF NOT EXISTS ix_episodic_memories_topic_id ON episodic_memories(topic_id);

-- Semantic memories table
CREATE TABLE IF NOT EXISTS semantic_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    entity_type VARCHAR(30) NOT NULL,
    entity_id UUID NOT NULL,
    mastery_level NUMERIC(3,2) NOT NULL DEFAULT 0.00,
    attempts_total INTEGER NOT NULL DEFAULT 0,
    attempts_correct INTEGER NOT NULL DEFAULT 0,
    total_time_seconds INTEGER NOT NULL DEFAULT 0,
    confidence NUMERIC(3,2) NOT NULL DEFAULT 0.50,
    last_practiced_at TIMESTAMPTZ,
    last_correct_at TIMESTAMPTZ,
    last_incorrect_at TIMESTAMPTZ,
    current_streak INTEGER NOT NULL DEFAULT 0,
    best_streak INTEGER NOT NULL DEFAULT 0,
    fsrs_stability NUMERIC(10,4),
    fsrs_difficulty NUMERIC(5,4),
    fsrs_due_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_student_entity UNIQUE (student_id, entity_type, entity_id)
);
CREATE INDEX IF NOT EXISTS ix_semantic_memories_student_id ON semantic_memories(student_id);

-- Procedural memories table
CREATE TABLE IF NOT EXISTS procedural_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    strategy_type VARCHAR(50) NOT NULL,
    strategy_value VARCHAR(100) NOT NULL,
    effectiveness NUMERIC(3,2) NOT NULL DEFAULT 0.50,
    sample_size INTEGER NOT NULL DEFAULT 0,
    subject_id UUID REFERENCES subjects(id),
    topic_id UUID REFERENCES topics(id),
    last_observation_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_student_strategy UNIQUE (student_id, strategy_type, strategy_value, topic_id)
);
CREATE INDEX IF NOT EXISTS ix_procedural_memories_student_id ON procedural_memories(student_id);

-- Associative memories table
CREATE TABLE IF NOT EXISTS associative_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    association_type VARCHAR(30) NOT NULL,
    content TEXT NOT NULL,
    strength NUMERIC(3,2) NOT NULL DEFAULT 0.50,
    times_used INTEGER NOT NULL DEFAULT 0,
    times_effective INTEGER NOT NULL DEFAULT 0,
    tags JSONB NOT NULL DEFAULT '[]',
    last_used_at TIMESTAMPTZ,
    embedding_id VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_associative_memories_student_id ON associative_memories(student_id);

-- ==============================================================================
-- SPACED REPETITION TABLES
-- ==============================================================================

-- Review items table
CREATE TABLE IF NOT EXISTS review_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    item_type VARCHAR(30) NOT NULL,
    item_id UUID NOT NULL,
    topic_id UUID REFERENCES topics(id),
    stability NUMERIC(10,6) NOT NULL DEFAULT 1.0,
    difficulty NUMERIC(5,4) NOT NULL DEFAULT 0.3,
    elapsed_days NUMERIC(10,4) NOT NULL DEFAULT 0,
    scheduled_days NUMERIC(10,4) NOT NULL DEFAULT 1,
    reps INTEGER NOT NULL DEFAULT 0,
    lapses INTEGER NOT NULL DEFAULT 0,
    state INTEGER NOT NULL DEFAULT 0,
    last_review TIMESTAMPTZ,
    due TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_review_item UNIQUE (student_id, item_type, item_id)
);
CREATE INDEX IF NOT EXISTS ix_review_items_student_id ON review_items(student_id);
CREATE INDEX IF NOT EXISTS ix_review_items_due ON review_items(due);
CREATE INDEX IF NOT EXISTS ix_review_items_topic_id ON review_items(topic_id);

-- Review logs table
CREATE TABLE IF NOT EXISTS review_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id UUID NOT NULL REFERENCES review_items(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL,
    state INTEGER NOT NULL,
    scheduled_days NUMERIC(10,4) NOT NULL,
    elapsed_days NUMERIC(10,4) NOT NULL,
    stability_before NUMERIC(10,6) NOT NULL,
    stability_after NUMERIC(10,6) NOT NULL,
    difficulty_before NUMERIC(5,4) NOT NULL,
    difficulty_after NUMERIC(5,4) NOT NULL,
    review_duration_ms INTEGER,
    reviewed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_review_logs_item_id ON review_logs(item_id);
CREATE INDEX IF NOT EXISTS ix_review_logs_reviewed_at ON review_logs(reviewed_at);

-- ==============================================================================
-- DIAGNOSTIC TABLES
-- ==============================================================================

-- Diagnostic scans table
CREATE TABLE IF NOT EXISTS diagnostic_scans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    scan_type VARCHAR(30) NOT NULL DEFAULT 'full',
    trigger_reason VARCHAR(100),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    risk_score NUMERIC(3,2),
    findings_count INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_diagnostic_scans_student_id ON diagnostic_scans(student_id);
CREATE INDEX IF NOT EXISTS ix_diagnostic_scans_status ON diagnostic_scans(status);

-- Diagnostic indicators table
CREATE TABLE IF NOT EXISTS diagnostic_indicators (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id UUID NOT NULL REFERENCES diagnostic_scans(id) ON DELETE CASCADE,
    indicator_type VARCHAR(50) NOT NULL,
    risk_score NUMERIC(3,2) NOT NULL,
    confidence NUMERIC(3,2),
    evidence JSONB NOT NULL DEFAULT '[]',
    threshold_level VARCHAR(20),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_diagnostic_indicators_scan_id ON diagnostic_indicators(scan_id);

-- Diagnostic recommendations table
CREATE TABLE IF NOT EXISTS diagnostic_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scan_id UUID NOT NULL REFERENCES diagnostic_scans(id) ON DELETE CASCADE,
    indicator_id UUID REFERENCES diagnostic_indicators(id),
    recommendation_type VARCHAR(30) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    for_teacher BOOLEAN NOT NULL DEFAULT TRUE,
    for_parent BOOLEAN NOT NULL DEFAULT FALSE,
    disclaimer TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_diagnostic_recommendations_scan_id ON diagnostic_recommendations(scan_id);

-- ==============================================================================
-- NOTIFICATION TABLES
-- ==============================================================================

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'info',
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}',
    topic_id UUID REFERENCES topics(id),
    session_id UUID,
    suggested_actions JSONB NOT NULL DEFAULT '[]',
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    acknowledged_by UUID REFERENCES users(id) ON DELETE SET NULL,
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_alerts_student_id ON alerts(student_id);
CREATE INDEX IF NOT EXISTS ix_alerts_alert_type ON alerts(alert_type);
CREATE INDEX IF NOT EXISTS ix_alerts_status ON alerts(status);

-- Notifications table
CREATE TABLE IF NOT EXISTS notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    notification_type VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    data JSONB NOT NULL DEFAULT '{}',
    channels JSONB NOT NULL DEFAULT '["in_app"]',
    delivery_status JSONB NOT NULL DEFAULT '{}',
    read_at TIMESTAMPTZ,
    action_url TEXT,
    action_label VARCHAR(100),
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_notifications_user_id ON notifications(user_id);
CREATE INDEX IF NOT EXISTS ix_notifications_notification_type ON notifications(notification_type);

-- Notification preferences table
CREATE TABLE IF NOT EXISTS notification_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    notification_type VARCHAR(50) NOT NULL,
    in_app BOOLEAN NOT NULL DEFAULT TRUE,
    push BOOLEAN NOT NULL DEFAULT TRUE,
    email BOOLEAN NOT NULL DEFAULT FALSE,
    sms BOOLEAN NOT NULL DEFAULT FALSE,
    frequency VARCHAR(20) NOT NULL DEFAULT 'immediate',
    quiet_start TIME,
    quiet_end TIME,
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_user_notification_type UNIQUE (user_id, notification_type)
);
CREATE INDEX IF NOT EXISTS ix_notification_preferences_user_id ON notification_preferences(user_id);

-- Notification templates table
CREATE TABLE IF NOT EXISTS notification_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    notification_type VARCHAR(50) NOT NULL,
    language_code VARCHAR(10) NOT NULL,
    title_template TEXT NOT NULL,
    message_template TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_template UNIQUE (notification_type, language_code)
);

-- ==============================================================================
-- ANALYTICS TABLES
-- ==============================================================================

-- Analytics events table
CREATE TABLE IF NOT EXISTS analytics_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    student_id UUID REFERENCES users(id),
    event_type VARCHAR(100) NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    session_id UUID,
    conversation_id UUID,
    topic_id UUID REFERENCES topics(id),
    data JSONB NOT NULL DEFAULT '{}',
    device_type VARCHAR(20),
    client_version VARCHAR(20)
);
CREATE INDEX IF NOT EXISTS ix_analytics_events_user_id ON analytics_events(user_id);
CREATE INDEX IF NOT EXISTS ix_analytics_events_event_type ON analytics_events(event_type);
CREATE INDEX IF NOT EXISTS ix_analytics_events_occurred_at ON analytics_events(occurred_at);

-- Daily summaries table
CREATE TABLE IF NOT EXISTS daily_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    summary_date DATE NOT NULL,
    total_time_seconds INTEGER NOT NULL DEFAULT 0,
    sessions_count INTEGER NOT NULL DEFAULT 0,
    questions_attempted INTEGER NOT NULL DEFAULT 0,
    questions_correct INTEGER NOT NULL DEFAULT 0,
    messages_sent INTEGER NOT NULL DEFAULT 0,
    topics_practiced JSONB NOT NULL DEFAULT '[]',
    average_score NUMERIC(5,2),
    engagement_score NUMERIC(3,2),
    daily_streak INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_student_date UNIQUE (student_id, summary_date)
);
CREATE INDEX IF NOT EXISTS ix_daily_summaries_student_id ON daily_summaries(student_id);
CREATE INDEX IF NOT EXISTS ix_daily_summaries_summary_date ON daily_summaries(summary_date);

-- Mastery snapshots table
CREATE TABLE IF NOT EXISTS mastery_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    subject_mastery JSONB NOT NULL DEFAULT '{}',
    topic_mastery JSONB NOT NULL DEFAULT '{}',
    overall_mastery NUMERIC(3,2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_student_snapshot_date UNIQUE (student_id, snapshot_date)
);
CREATE INDEX IF NOT EXISTS ix_mastery_snapshots_student_id ON mastery_snapshots(student_id);
CREATE INDEX IF NOT EXISTS ix_mastery_snapshots_snapshot_date ON mastery_snapshots(snapshot_date);

-- Engagement metrics table
CREATE TABLE IF NOT EXISTS engagement_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    metric_date DATE NOT NULL,
    login_count INTEGER NOT NULL DEFAULT 0,
    session_duration_avg INTEGER NOT NULL DEFAULT 0,
    questions_per_session NUMERIC(5,2),
    accuracy_trend NUMERIC(3,2),
    streak_days INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_student_metric_date UNIQUE (student_id, metric_date)
);
CREATE INDEX IF NOT EXISTS ix_engagement_metrics_student_id ON engagement_metrics(student_id);
CREATE INDEX IF NOT EXISTS ix_engagement_metrics_metric_date ON engagement_metrics(metric_date);

-- ==============================================================================
-- SETTINGS TABLES
-- ==============================================================================

-- Tenant settings table
CREATE TABLE IF NOT EXISTS tenant_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value JSONB NOT NULL,
    description TEXT,
    value_type VARCHAR(20) NOT NULL DEFAULT 'string',
    allow_school_override BOOLEAN NOT NULL DEFAULT FALSE,
    allow_user_override BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- User preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    theme VARCHAR(20) NOT NULL DEFAULT 'system',
    language VARCHAR(10) NOT NULL DEFAULT 'tr',
    font_size VARCHAR(20) NOT NULL DEFAULT 'medium',
    high_contrast BOOLEAN NOT NULL DEFAULT FALSE,
    reduce_motion BOOLEAN NOT NULL DEFAULT FALSE,
    sound_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    default_session_type VARCHAR(30),
    default_persona VARCHAR(50),
    preferences JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Feature flags table
CREATE TABLE IF NOT EXISTS feature_flags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_key VARCHAR(100) UNIQUE NOT NULL,
    is_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    rollout_percentage INTEGER NOT NULL DEFAULT 100,
    user_ids JSONB NOT NULL DEFAULT '[]',
    description TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ==============================================================================
-- LOCALIZATION TABLES
-- ==============================================================================

-- Languages table
CREATE TABLE IF NOT EXISTS languages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    native_name VARCHAR(100),
    is_rtl BOOLEAN NOT NULL DEFAULT FALSE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    translation_progress INTEGER NOT NULL DEFAULT 0
);

-- Translations table
CREATE TABLE IF NOT EXISTS translations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(30) NOT NULL,
    entity_id UUID NOT NULL,
    field_name VARCHAR(50) NOT NULL,
    language_code VARCHAR(10) NOT NULL,
    translated_text TEXT NOT NULL,
    is_reviewed BOOLEAN NOT NULL DEFAULT FALSE,
    reviewed_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT unique_translation UNIQUE (entity_type, entity_id, field_name, language_code)
);
CREATE INDEX IF NOT EXISTS ix_translations_entity_type ON translations(entity_type);
CREATE INDEX IF NOT EXISTS ix_translations_entity_id ON translations(entity_id);
CREATE INDEX IF NOT EXISTS ix_translations_language_code ON translations(language_code);

-- ==============================================================================
-- AUDIT LOG TABLE
-- ==============================================================================

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    user_email VARCHAR(255),
    user_type VARCHAR(20),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50),
    entity_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    request_id UUID,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS ix_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS ix_audit_logs_entity_type ON audit_logs(entity_type);
CREATE INDEX IF NOT EXISTS ix_audit_logs_created_at ON audit_logs(created_at);

-- ==============================================================================
-- GAMING TABLES
-- ==============================================================================

-- Game sessions table
CREATE TABLE IF NOT EXISTS game_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    game_type VARCHAR(30) NOT NULL,
    game_mode VARCHAR(30) NOT NULL DEFAULT 'practice',
    difficulty VARCHAR(20) NOT NULL DEFAULT 'medium',
    player_color VARCHAR(20) NOT NULL DEFAULT 'white',
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    result VARCHAR(20),
    winner VARCHAR(20),
    total_moves INTEGER NOT NULL DEFAULT 0,
    time_spent_seconds INTEGER NOT NULL DEFAULT 0,
    hints_used INTEGER NOT NULL DEFAULT 0,
    mistakes_count INTEGER NOT NULL DEFAULT 0,
    excellent_moves_count INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ,
    initial_position JSONB,
    final_position JSONB,
    game_state JSONB NOT NULL DEFAULT '{}',
    checkpoint_data JSONB,
    coach_summary TEXT,
    learning_points JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT valid_game_type CHECK (game_type IN ('chess', 'connect4')),
    CONSTRAINT valid_game_mode CHECK (game_mode IN ('tutorial', 'practice', 'challenge', 'puzzle', 'analysis')),
    CONSTRAINT valid_game_difficulty CHECK (difficulty IN ('beginner', 'easy', 'medium', 'hard', 'expert')),
    CONSTRAINT valid_game_status CHECK (status IN ('active', 'paused', 'completed', 'abandoned')),
    CONSTRAINT valid_game_result CHECK (result IS NULL OR result IN ('win', 'loss', 'draw', 'timeout', 'resignation'))
);
CREATE INDEX IF NOT EXISTS ix_game_sessions_student_id ON game_sessions(student_id);
CREATE INDEX IF NOT EXISTS ix_game_sessions_game_type ON game_sessions(game_type);
CREATE INDEX IF NOT EXISTS ix_game_sessions_status ON game_sessions(status);
CREATE INDEX IF NOT EXISTS ix_game_sessions_created_at ON game_sessions(created_at);

-- Game moves table
CREATE TABLE IF NOT EXISTS game_moves (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,
    move_number INTEGER NOT NULL,
    player VARCHAR(20) NOT NULL,
    notation VARCHAR(20) NOT NULL,
    position_before JSONB NOT NULL,
    position_after JSONB NOT NULL,
    time_spent_seconds INTEGER NOT NULL DEFAULT 0,
    evaluation_before NUMERIC(7,2),
    evaluation_after NUMERIC(7,2),
    quality VARCHAR(20),
    is_best_move BOOLEAN NOT NULL DEFAULT FALSE,
    best_move VARCHAR(20),
    coach_comment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT valid_move_player CHECK (player IN ('player', 'ai')),
    CONSTRAINT valid_move_quality CHECK (quality IS NULL OR quality IN ('excellent', 'good', 'inaccuracy', 'mistake', 'blunder'))
);
CREATE INDEX IF NOT EXISTS ix_game_moves_session_id ON game_moves(session_id);

-- Game analyses table
CREATE TABLE IF NOT EXISTS game_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES game_sessions(id) ON DELETE CASCADE,
    move_number INTEGER,
    analysis_type VARCHAR(20) NOT NULL DEFAULT 'position',
    position JSONB NOT NULL,
    evaluation NUMERIC(7,2),
    evaluation_text VARCHAR(200),
    best_moves JSONB NOT NULL DEFAULT '[]',
    threats JSONB NOT NULL DEFAULT '[]',
    opportunities JSONB NOT NULL DEFAULT '[]',
    strategic_themes JSONB NOT NULL DEFAULT '[]',
    coach_explanation TEXT,
    learning_points JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT valid_analysis_type CHECK (analysis_type IN ('position', 'move', 'phase', 'game', 'opening', 'endgame'))
);
CREATE INDEX IF NOT EXISTS ix_game_analyses_session_id ON game_analyses(session_id);

-- ==============================================================================
-- SEED DATA
-- ==============================================================================

-- Insert default roles
INSERT INTO roles (code, name, description, level, is_system) VALUES
    ('tenant_admin', 'Tenant Administrator', 'Full access to tenant management', 100, true),
    ('school_admin', 'School Administrator', 'School-level administration', 80, true),
    ('teacher', 'Teacher', 'Teacher with class management', 50, true),
    ('parent', 'Parent', 'Parent/Guardian role', 30, true),
    ('student', 'Student', 'Student user', 10, true)
ON CONFLICT (code) DO NOTHING;

-- Insert default languages
INSERT INTO languages (code, name, native_name, is_default) VALUES
    ('tr', 'Turkish', 'Türkçe', true),
    ('en', 'English', 'English', false)
ON CONFLICT (code) DO NOTHING;

-- ==============================================================================
-- COMPLETED
-- ==============================================================================
-- Note: This is a reference script. Actual schema is created via SQLAlchemy models.
-- See TenantService._initialize_tenant_schema() for the authoritative implementation.
-- Total tables in this reference: ~57 (plus additional tables in SQLAlchemy models)
