"""
GajahMada AI — Knowledge Dataset Builder (Continued Pretraining & SFT).

Strictly separates:
1. Continued Pretraining Corpus (`mode="continued_pretraining"`): raw approved prose/documents
   teaching domain terminology, language patterns, and distributions.
2. Supervised Fine-Tuning (`mode="sft"`): approved `(instruction, response)` pairs teaching
   grounded answering, citation behavior, and instruction following.

Produces versioned JSONL datasets with SHA-256 checksums and full source provenance manifests.
"""
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional
import uuid

from gajahmada_ai.knowledge.interfaces import KnowledgeRecord, KnowledgeSource
from gajahmada_ai.learning.knowledge_filter import TrainingKnowledgeFilter


class KnowledgeDatasetBuilder:
    """Builds versioned, provenance-tracked Continued Pretraining and SFT datasets."""

    def __init__(self, knowledge_filter: Optional[TrainingKnowledgeFilter] = None):
        self.filter = knowledge_filter or TrainingKnowledgeFilter()

    def build_continued_pretraining_dataset(
        self,
        records: List[KnowledgeRecord],
        sources_by_id: Optional[Dict[str, KnowledgeSource]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        filtered = self.filter.filter_batch(records, sources_by_id=sources_by_id)
        approved: List[KnowledgeRecord] = filtered["approved"]

        dataset_id = f"ds-pretrain-{uuid.uuid4().hex[:10]}"
        items = [
            {
                "text": rec.content.strip(),
                "record_id": rec.record_id,
                "source_id": rec.source_id,
                "verification_status": rec.verification_status,
                "knowledge_layer": rec.knowledge_layer,
            }
            for rec in approved
        ]
        manifest = self._build_manifest(
            dataset_id=dataset_id,
            mode="continued_pretraining",
            items=items,
            rejected=filtered["rejected"],
            output_dir=output_dir,
        )
        return {"manifest": manifest, "items": items, "rejected": filtered["rejected"]}

    def build_sft_dataset(
        self,
        examples: List[Dict[str, Any]],
        records_by_id: Optional[Dict[str, KnowledgeRecord]] = None,
        sources_by_id: Optional[Dict[str, KnowledgeSource]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build an SFT instruction-response dataset. Each example must have
        `instruction` and `response` (and optional `source_id`/`record_id` for verification).
        """
        dataset_id = f"ds-sft-{uuid.uuid4().hex[:10]}"
        approved_items: List[Dict[str, Any]] = []
        rejected: List[Dict[str, str]] = []
        seen = set()

        for idx, ex in enumerate(examples):
            instr = str(ex.get("instruction", "")).strip()
            resp = str(ex.get("response", "")).strip()
            ex_id = str(ex.get("example_id", f"sft-{idx}"))
            if not instr or not resp:
                rejected.append({"example_id": ex_id, "reason": "missing_instruction_or_response"})
                continue

            # Validate through TrainingKnowledgeFilter
            rec_id = str(ex.get("record_id", ex_id))
            src_id = str(ex.get("source_id", ""))
            backing_rec = (records_by_id or {}).get(rec_id) or KnowledgeRecord(
                record_id=rec_id,
                content=f"{instr}\n{resp}",
                source_id=src_id,
                source_type=str(ex.get("source_type", "curated_sft")),
                verification_status=str(ex.get("verification_status", "UNVERIFIED")),
                provenance={"license": str(ex.get("license", ""))},
            )
            src = (sources_by_id or {}).get(src_id)
            ok, reason = self.filter.evaluate_record(backing_rec, source=src)
            if not ok:
                rejected.append({"example_id": ex_id, "reason": reason})
                continue

            dedup_key = hashlib.sha256(f"{instr.lower()}|{resp.lower()}".encode("utf-8")).hexdigest()
            if dedup_key in seen:
                rejected.append({"example_id": ex_id, "reason": "duplicate_sft_example"})
                continue
            seen.add(dedup_key)

            approved_items.append({
                "instruction": instr,
                "response": resp,
                "source_id": src_id,
                "record_id": rec_id,
                "verification_status": backing_rec.verification_status,
            })

        manifest = self._build_manifest(
            dataset_id=dataset_id,
            mode="sft",
            items=approved_items,
            rejected=rejected,
            output_dir=output_dir,
        )
        return {"manifest": manifest, "items": approved_items, "rejected": rejected}

    def _build_manifest(
        self,
        dataset_id: str,
        mode: str,
        items: List[Dict[str, Any]],
        rejected: List[Dict[str, str]],
        output_dir: Optional[str],
    ) -> Dict[str, Any]:
        serialized = "\n".join(json.dumps(it, sort_keys=True, ensure_ascii=False) for it in items)
        checksum = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        source_ids = sorted({str(it.get("source_id", "")) for it in items if it.get("source_id")})
        manifest: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "mode": mode,
            "created_at": round(time.time(), 3),
            "item_count": len(items),
            "rejected_count": len(rejected),
            "sha256": checksum,
            "source_ids": source_ids,
            "jsonl_path": None,
        }
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            jsonl_path = os.path.join(output_dir, f"{dataset_id}.jsonl")
            manifest_path = os.path.join(output_dir, f"{dataset_id}.manifest.json")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(serialized + ("\n" if serialized else ""))
            manifest["jsonl_path"] = jsonl_path
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        return manifest
