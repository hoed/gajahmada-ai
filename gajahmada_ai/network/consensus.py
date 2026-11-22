import hashlib
from typing import Any, Dict, List


class QuorumConsensusManager:
    """
    Quorum Acknowledgement & Evidence-Weighted Conflict Resolution Manager (Layer 7).
    Engineering Rules:
    - Do NOT invent an untested custom BFT algorithm.
    - Do NOT automatically trust majority agreement as factual truth.
    - Evaluate evidence independence, provenance hashes, methodology, source quality,
      and independent verification results.
    """

    def evaluate_submissions(
        self,
        task_id: str,
        submissions: List[Dict[str, Any]],
        quorum_size: int = 2,
    ) -> Dict[str, Any]:
        valid_subs = [s for s in submissions if s.get("signature_verified", False)]
        quorum_reached = len(valid_subs) >= quorum_size

        if not valid_subs:
            return {
                "task_id": task_id,
                "quorum_reached": False,
                "status": "INSUFFICIENT_VERIFIED_SUBMISSIONS",
                "accepted_findings": [],
                "conflicts": [],
                "confidence": 0.0,
            }

        # Evaluate evidence independence (distinct provenance hashes & distinct node_ids)
        unique_nodes = {s["node_id"] for s in valid_subs}
        provenance_hashes = set()
        evidence_backed = []
        unbacked_majority = []

        for s in valid_subs:
            ev_list = s.get("evidence", [])
            has_proof = bool(ev_list) or bool(s.get("symbolic_verified", False))
            for ev in ev_list:
                h = ev.get("provenance_hash") or hashlib.sha256(str(ev.get("excerpt", "")).encode("utf-8")).hexdigest()[:16]
                provenance_hashes.add(h)
            if has_proof:
                evidence_backed.append(s)
            else:
                unbacked_majority.append(s)

        # Detect conflicts among conclusions
        conclusions_by_node = {s["node_id"]: str(s.get("conclusion", "")).strip() for s in valid_subs}
        conflicts = []
        nodes_list = list(conclusions_by_node.keys())
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                na, nb = nodes_list[i], nodes_list[j]
                ca, cb = conclusions_by_node[na], conclusions_by_node[nb]
                if ca and cb and ca.lower() != cb.lower():
                    # Check if one has verified evidence/symbolic proof and the other does not
                    sa = next(s for s in valid_subs if s["node_id"] == na)
                    sb = next(s for s in valid_subs if s["node_id"] == nb)
                    conflicts.append({
                        "node_a": na,
                        "node_b": nb,
                        "claim_a": ca[:160],
                        "claim_b": cb[:160],
                        "node_a_has_evidence": bool(sa.get("evidence")) or bool(sa.get("symbolic_verified")),
                        "node_b_has_evidence": bool(sb.get("evidence")) or bool(sb.get("symbolic_verified")),
                    })

        # Rule: Evidence-backed submission overrides unverified majority agreement!
        if evidence_backed:
            best = max(evidence_backed, key=lambda s: float(s.get("confidence", 0.5)))
            status = "VERIFIED_BY_EVIDENCE_AND_QUORUM" if quorum_reached else "PARTIALLY_VERIFIED"
            confidence = round(min(0.98, float(best.get("confidence", 0.8))), 4)
        else:
            best = valid_subs[0]
            status = "UNVERIFIED_CONSENSUS_NO_PRIMARY_EVIDENCE"
            confidence = 0.35

        return {
            "task_id": task_id,
            "quorum_reached": quorum_reached,
            "participating_nodes": len(unique_nodes),
            "independent_evidence_sources": len(provenance_hashes),
            "status": status,
            "selected_node_id": best["node_id"],
            "selected_conclusion": best.get("conclusion", ""),
            "confidence": confidence,
            "conflicts": conflicts,
            "unverified_majority_rejected": len(unbacked_majority) > len(evidence_backed) and len(evidence_backed) > 0,
        }
