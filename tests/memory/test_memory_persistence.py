import os
import tempfile

from gajahmada_ai.memory.manager import MemoryManager
from gajahmada_ai.memory.models import MemoryType, VerificationStatus, VisibilityPolicy


def test_7_memory_survives_process_restarts():
    """TEST 7: Memory survives process restarts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "gajahmada_v2.db")
        obj_dir = os.path.join(tmpdir, "objects")

        # Instance 1 writes semantic and episodic memory records
        mgr1 = MemoryManager(sqlite_path=db_path, backup_dir=obj_dir)
        rec = mgr1.store(
            content="Majapahit didirikan oleh Raden Wijaya pada tahun 1293 di Trowulan.",
            memory_type=MemoryType.SEMANTIC,
            origin_node_id="node-alpha",
            source_type="sovereign_core_corpus",
            confidence=0.95,
            verification_status=VerificationStatus.VERIFIED,
            visibility_policy=VisibilityPolicy.COLLECTIVE,
        )
        rec_id = rec["record_id"]
        del mgr1

        # Instance 2 simulates a fresh process restart reading from the same database
        mgr2 = MemoryManager(sqlite_path=db_path, backup_dir=obj_dir)
        loaded = mgr2.storage.get_record(rec_id)
        assert loaded is not None
        assert loaded["content"] == "Majapahit didirikan oleh Raden Wijaya pada tahun 1293 di Trowulan."
        assert loaded["verification_status"] == "VERIFIED"
        assert mgr2.provenance.verify_record_integrity(loaded) is True

        # Test versioning and correction audit trail
        updated = mgr2.correct_memory(
            record_id=rec_id,
            new_content="Majapahit didirikan oleh Raden Wijaya pada tahun 1293 M di Trowulan, Jawa Timur.",
            reason="added_province_detail",
        )
        assert updated["version"] == 2
        versions = mgr2.versioning.get_history(rec_id)
        assert len(versions) == 2
        assert versions[0]["version"] == 1
        assert versions[1]["version"] == 2

        # Test content-addressed backup export
        backup = mgr2.export_backup()
        assert backup["records_count"] == 1
        assert os.path.exists(backup["backup_path"])


def test_8_contradictory_knowledge_preserved_and_identified():
    """TEST 8: Contradictory knowledge is preserved and identified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "contra.db")
        mgr = MemoryManager(sqlite_path=db_path, backup_dir=os.path.join(tmpdir, "obj"))

        rec_a = mgr.store(
            content="Kemaharajaan Majapahit didirikan pada tahun 1293 di Trowulan.",
            memory_type=MemoryType.SEMANTIC,
            verification_status=VerificationStatus.VERIFIED,
        )
        rec_b = mgr.store(
            content="Kemaharajaan Majapahit didirikan pada tahun 1450 di Trowulan.",
            memory_type=MemoryType.SEMANTIC,
            verification_status=VerificationStatus.UNVERIFIED,
        )

        # Both records must be preserved (never silently overwritten)
        assert mgr.storage.get_record(rec_a["record_id"]) is not None
        assert mgr.storage.get_record(rec_b["record_id"]) is not None

        # Contradiction must be automatically detected and persisted
        assert len(rec_b["detected_contradictions"]) >= 1
        persisted_contradictions = mgr.storage.list_contradictions()
        assert len(persisted_contradictions) >= 1
        assert persisted_contradictions[0]["record_a_id"] == rec_a["record_id"]
        assert persisted_contradictions[0]["record_b_id"] == rec_b["record_id"]
