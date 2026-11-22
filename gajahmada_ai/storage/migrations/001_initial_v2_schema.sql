-- GajahMada 2.0 — Primary PostgreSQL + pgvector Persistent Memory Schema
-- Migration: 001_initial_v2_schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS gm_memory_records (
    record_id VARCHAR(64) PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type VARCHAR(32) NOT NULL, -- working, semantic, episodic, procedural, collective
    entity_references JSONB NOT NULL DEFAULT '[]',
    source_provenance JSONB NOT NULL DEFAULT '{}',
    created_at DOUBLE PRECISION NOT NULL,
    modified_at DOUBLE PRECISION NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.75,
    verification_status VARCHAR(32) NOT NULL DEFAULT 'UNVERIFIED',
    visibility_policy VARCHAR(32) NOT NULL DEFAULT 'LOCAL',
    integrity_hash VARCHAR(64) NOT NULL,
    embedding vector(384),
    superseded_by VARCHAR(64),
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_gm_memory_type ON gm_memory_records(memory_type);
CREATE INDEX IF NOT EXISTS idx_gm_memory_verification ON gm_memory_records(verification_status);

CREATE TABLE IF NOT EXISTS gm_memory_versions (
    version_id VARCHAR(80) PRIMARY KEY,
    record_id VARCHAR(64) NOT NULL REFERENCES gm_memory_records(record_id),
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    verification_status VARCHAR(32) NOT NULL,
    integrity_hash VARCHAR(64) NOT NULL,
    modified_by VARCHAR(128) NOT NULL,
    reason TEXT,
    timestamp DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS gm_knowledge_contradictions (
    contradiction_id VARCHAR(96) PRIMARY KEY,
    record_a_id VARCHAR(64) NOT NULL,
    record_b_id VARCHAR(64) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    claim_a TEXT NOT NULL,
    claim_b TEXT NOT NULL,
    reason TEXT NOT NULL,
    severity VARCHAR(32) NOT NULL DEFAULT 'HIGH',
    resolution_status VARCHAR(32) NOT NULL DEFAULT 'OPEN',
    detected_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS gm_knowledge_relations (
    relation_id VARCHAR(96) PRIMARY KEY,
    source_entity VARCHAR(255) NOT NULL,
    predicate VARCHAR(128) NOT NULL,
    target_entity VARCHAR(255) NOT NULL,
    record_id VARCHAR(64) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.8,
    created_at DOUBLE PRECISION NOT NULL
);
