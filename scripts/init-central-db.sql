-- ==============================================================================
-- EduSynapseOS Central Database Initialization
-- ==============================================================================
-- This script runs when the central-db container first starts.
-- It only initializes the central database which contains:
--   - Tenants table
--   - System users table
--   - System sessions table
--   - System audit logs table
--   - Licenses table
--   - Feature flags table
--
-- Tenant databases are created dynamically via Docker SDK when a new tenant
-- is provisioned. Each tenant gets its own PostgreSQL container.
-- ==============================================================================

-- Create required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Grant permissions to the application user
GRANT ALL PRIVILEGES ON DATABASE edusynapse_central TO edusynapse;
