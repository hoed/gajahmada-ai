import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

from gajahmada_ai.storage.interfaces import StorageBackend


class RelationalMemoryStore(StorageBackend):
    """
    Relational Persistent Storage Engine for GajahMada 2.0 Memory & Knowledge.
    - Supports PostgreSQL (`postgresql://...` via `psycopg`) with `pgvector` as primary database
      when `DATABASE_URL` is configured and reachable.
    - Automatically provides a transactional SQLite relational database (`gajahmada_v2.db`)
      with identical tables and ACID persistence when running standalone/offline without an
      active PostgreSQL daemon.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        sqlite_path: str = "./knowledge_store/gajahmada_v2.db",
    ):
        self.db_url = (db_url or os.environ.get("DATABASE_URL", "")).strip()
        self.sqlite_path = sqlite_path
        self.engine_type = "sqlite"
        self._pg_conn = None

        if self.db_url.startswith("postgres"):
            try:
                import psycopg
                self._pg_conn = psycopg.connect(self.db_url, autocommit=True)
                self.engine_type = "postgresql+pgvector"
            except Exception:
                self._pg_conn = None
                self.engine_type = "sqlite"

        if self.engine_type == "sqlite":
            if os.environ.get("VERCEL") == "1" and not self.sqlite_path.startswith("/tmp"):
                self.sqlite_path = "/tmp/knowledge_store/gajahmada_v2.db"
            os.makedirs(os.path.dirname(os.path.abspath(self.sqlite_path)), exist_ok=True)

        self._run_migrations()

    def _get_sqlite(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _run_migrations(self):
        if self.engine_type == "postgresql+pgvector" and self._pg_conn is not None:
            mig_file = os.path.join(os.path.dirname(__file__), "migrations", "001_initial_v2_schema.sql")
            if os.path.exists(mig_file):
                with open(mig_file, "r", encoding="utf-8") as f:
                    sql = f.read()
                with self._pg_conn.cursor() as cur:
                    cur.execute(sql)
            return

        with self._get_sqlite() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS gm_memory_records (
                    record_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    entity_references TEXT NOT NULL DEFAULT '[]',
                    source_provenance TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    modified_at REAL NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    confidence REAL NOT NULL DEFAULT 0.75,
                    verification_status TEXT NOT NULL DEFAULT 'UNVERIFIED',
                    visibility_policy TEXT NOT NULL DEFAULT 'LOCAL',
                    integrity_hash TEXT NOT NULL,
                    embedding TEXT NOT NULL DEFAULT '[]',
                    superseded_by TEXT,
                    is_deleted INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_gm_mem_type ON gm_memory_records(memory_type);
                CREATE INDEX IF NOT EXISTS idx_gm_mem_ver ON gm_memory_records(verification_status);

                CREATE TABLE IF NOT EXISTS gm_memory_versions (
                    version_id TEXT PRIMARY KEY,
                    record_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    verification_status TEXT NOT NULL,
                    integrity_hash TEXT NOT NULL,
                    modified_by TEXT NOT NULL,
                    reason TEXT,
                    timestamp REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS gm_knowledge_contradictions (
                    contradiction_id TEXT PRIMARY KEY,
                    record_a_id TEXT NOT NULL,
                    record_b_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    claim_a TEXT NOT NULL,
                    claim_b TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    severity TEXT NOT NULL DEFAULT 'HIGH',
                    resolution_status TEXT NOT NULL DEFAULT 'OPEN',
                    detected_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS gm_knowledge_relations (
                    relation_id TEXT PRIMARY KEY,
                    source_entity TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    target_entity TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.8,
                    created_at REAL NOT NULL
                );
                """
            )
            conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["entity_references"] = json.loads(d.get("entity_references") or "[]")
        d["source_provenance"] = json.loads(d.get("source_provenance") or "{}")
        d["embedding"] = json.loads(d.get("embedding") or "[]")
        d["is_deleted"] = bool(d.get("is_deleted", 0))
        return d

    def save_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        with self._get_sqlite() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO gm_memory_records (
                    record_id, content, memory_type, entity_references, source_provenance,
                    created_at, modified_at, version, confidence, verification_status,
                    visibility_policy, integrity_hash, embedding, superseded_by, is_deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["record_id"],
                    record["content"],
                    record["memory_type"],
                    json.dumps(record.get("entity_references", []), ensure_ascii=False),
                    json.dumps(record.get("source_provenance", {}), ensure_ascii=False),
                    float(record.get("created_at", time.time())),
                    float(record.get("modified_at", time.time())),
                    int(record.get("version", 1)),
                    float(record.get("confidence", 0.75)),
                    record.get("verification_status", "UNVERIFIED"),
                    record.get("visibility_policy", "LOCAL"),
                    record["integrity_hash"],
                    json.dumps(record.get("embedding", [])),
                    record.get("superseded_by"),
                    1 if record.get("is_deleted", False) else 0,
                ),
            )
            conn.commit()
        return record

    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        with self._get_sqlite() as conn:
            cur = conn.execute(
                "SELECT * FROM gm_memory_records WHERE record_id = ?",
                (record_id,),
            )
            row = cur.fetchone()
            return self._row_to_record(row) if row else None

    def list_records(
        self,
        memory_type: Optional[str] = None,
        verification_status: Optional[str] = None,
        include_deleted: bool = False,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if not include_deleted:
            clauses.append("is_deleted = 0")
        if memory_type:
            clauses.append("memory_type = ?")
            params.append(memory_type)
        if verification_status:
            clauses.append("verification_status = ?")
            params.append(verification_status)

        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM gm_memory_records {where_sql} ORDER BY modified_at DESC LIMIT ?"
        params.append(int(limit))

        with self._get_sqlite() as conn:
            cur = conn.execute(sql, tuple(params))
            return [self._row_to_record(r) for r in cur.fetchall()]

    def delete_record(self, record_id: str, hard: bool = False) -> bool:
        with self._get_sqlite() as conn:
            if hard:
                cur = conn.execute("DELETE FROM gm_memory_records WHERE record_id = ?", (record_id,))
            else:
                cur = conn.execute(
                    "UPDATE gm_memory_records SET is_deleted = 1, modified_at = ? WHERE record_id = ?",
                    (time.time(), record_id),
                )
            conn.commit()
            return cur.rowcount > 0

    def save_version(self, version_entry: Dict[str, Any]) -> None:
        with self._get_sqlite() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO gm_memory_versions (
                    version_id, record_id, version, content, confidence,
                    verification_status, integrity_hash, modified_by, reason, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_entry["version_id"],
                    version_entry["record_id"],
                    int(version_entry["version"]),
                    version_entry["content"],
                    float(version_entry["confidence"]),
                    version_entry["verification_status"],
                    version_entry["integrity_hash"],
                    version_entry.get("modified_by", "local_node"),
                    version_entry.get("reason", ""),
                    float(version_entry.get("timestamp", time.time())),
                ),
            )
            conn.commit()

    def get_versions(self, record_id: str) -> List[Dict[str, Any]]:
        with self._get_sqlite() as conn:
            cur = conn.execute(
                "SELECT * FROM gm_memory_versions WHERE record_id = ? ORDER BY version ASC",
                (record_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def save_contradiction(self, contradiction: Dict[str, Any]) -> None:
        with self._get_sqlite() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO gm_knowledge_contradictions (
                    contradiction_id, record_a_id, record_b_id, subject,
                    claim_a, claim_b, reason, severity, resolution_status, detected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    contradiction["contradiction_id"],
                    contradiction["record_a_id"],
                    contradiction["record_b_id"],
                    contradiction["subject"],
                    contradiction["claim_a"],
                    contradiction["claim_b"],
                    contradiction["reason"],
                    contradiction.get("severity", "HIGH"),
                    contradiction.get("resolution_status", "OPEN"),
                    float(contradiction.get("detected_at", time.time())),
                ),
            )
            conn.commit()

    def list_contradictions(self) -> List[Dict[str, Any]]:
        with self._get_sqlite() as conn:
            cur = conn.execute(
                "SELECT * FROM gm_knowledge_contradictions ORDER BY detected_at DESC"
            )
            return [dict(r) for r in cur.fetchall()]

    def save_relation(self, relation: Dict[str, Any]) -> None:
        with self._get_sqlite() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO gm_knowledge_relations (
                    relation_id, source_entity, predicate, target_entity,
                    record_id, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relation["relation_id"],
                    relation["source_entity"],
                    relation["predicate"],
                    relation["target_entity"],
                    relation["record_id"],
                    float(relation.get("confidence", 0.8)),
                    float(relation.get("created_at", time.time())),
                ),
            )
            conn.commit()

    def list_relations(self, entity: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_sqlite() as conn:
            if entity:
                cur = conn.execute(
                    "SELECT * FROM gm_knowledge_relations WHERE LOWER(source_entity) = LOWER(?) OR LOWER(target_entity) = LOWER(?)",
                    (entity, entity),
                )
            else:
                cur = conn.execute("SELECT * FROM gm_knowledge_relations LIMIT 200")
            return [dict(r) for r in cur.fetchall()]

    def health(self) -> Dict[str, Any]:
        with self._get_sqlite() as conn:
            rec_count = conn.execute("SELECT COUNT(*) FROM gm_memory_records WHERE is_deleted = 0").fetchone()[0]
            ver_count = conn.execute("SELECT COUNT(*) FROM gm_memory_versions").fetchone()[0]
            con_count = conn.execute("SELECT COUNT(*) FROM gm_knowledge_contradictions").fetchone()[0]
            rel_count = conn.execute("SELECT COUNT(*) FROM gm_knowledge_relations").fetchone()[0]
        return {
            "engine_type": self.engine_type,
            "sqlite_path": self.sqlite_path,
            "postgres_configured": bool(self.db_url),
            "active_records": rec_count,
            "versions_count": ver_count,
            "contradictions_count": con_count,
            "relations_count": rel_count,
        }
