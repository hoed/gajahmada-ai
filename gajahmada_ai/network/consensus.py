import hashlib
from typing import Any, Dict, List, Optional

from gajahmada_ai.security.crypto import NodeIdentity, canonical_json_bytes


class QuorumConsensusManager:
    """
    Quorum Acknowledgement & Evidence-Weighted Conflict Resolution Manager (Layer 7).
    Engineering & Audit Rules (Round 3 Hardened):
    - NEVER trust a caller-supplied `signature_verified` boolean.
    - Cryptographically verify every submission's Ed25519 signature directly inside
      `evaluate_submissions()` (via signed GIP envelope or canonical submission signature).
    - Reject duplicate `node_id`s (Sybil/replay inflation) and revoked `node_id`s.
    - Enforce `quorum_size >= 2`.
    - Require genuinely independent evidence sources (distinct provenance hashes or
      independent symbolic proofs from >= 2 distinct verified nodes) before granting
      `VERIFIED_BY_EVIDENCE_AND_QUORUM`; otherwise downgrade to `CORROBORATED_SINGLE_SOURCE`
      or `PARTIALLY_VERIFIED`.
    """

    def __init__(self, verifier_identity: Optional[NodeIdentity] = None):
        self._default_verifier = verifier_identity or NodeIdentity(key_path=None)

    def _verify_submission_cryptographically(
        self,
        task_id: str,
        sub: Dict[str, Any],
        verifier: NodeIdentity,
    ) -> bool:
        """
        Verify a submission's Ed25519 signature directly.
        Supports two cryptographic proof formats:
        1. `gip_message`: full signed GIP/2.0 envelope dict (`sender_node_id == node_id`).
        2. `signature`: direct Ed25519 signature over canonical submission payload.
        Caller-supplied `signature_verified` booleans are strictly ignored.
        """
        node_id = str(sub.get("node_id", "")).strip()
        if not node_id or verifier.is_revoked(node_id):
            return False

        # Format 1: Full signed GIP/2.0 envelope
        gip_env = sub.get("gip_message")
        if isinstance(gip_env, dict):
            env_sender = str(gip_env.get("sender_node_id", "")).strip()
            env_sig = str(gip_env.get("signature", "")).strip()
            if env_sender != node_id or not env_sig:
                return False
            raw_mt = gip_env.get("message_type", "")
            mt_str = raw_mt.value if hasattr(raw_mt, "value") else str(raw_mt)
            signable = {
                "protocol_version": gip_env.get("protocol_version", "GIP/2.0"),
                "message_id": gip_env.get("message_id", ""),
                "message_type": mt_str,
                "sender_node_id": env_sender,
                "recipient_node_id": gip_env.get("recipient_node_id", "*"),
                "timestamp": gip_env.get("timestamp"),
                "nonce": gip_env.get("nonce", ""),
                "payload": gip_env.get("payload", {}),
            }
            if not verifier.verify_message(canonical_json_bytes(signable), env_sig, node_id):
                return False
            # Ensure submission conclusion matches signed envelope payload
            payload = gip_env.get("payload", {})
            if str(payload.get("task_id", task_id)) != task_id:
                return False
            if str(payload.get("conclusion", "")) != str(sub.get("conclusion", "")):
                return False
            return True

        # Format 2: Direct canonical submission Ed25519 signature
        sig = str(sub.get("signature", "")).strip()
        if not sig:
            return False
        canonical_sub = canonical_json_bytes({
            "task_id": task_id,
            "node_id": node_id,
            "conclusion": str(sub.get("conclusion", "")),
            "confidence": float(sub.get("confidence", 0.5)),
            "evidence": sub.get("evidence", []),
        })
        return verifier.verify_message(canonical_sub, sig, node_id)

    def evaluate_submissions(
        self,
        task_id: str,
        submissions: List[Dict[str, Any]],
        quorum_size: int = 2,
        verifier_identity: Optional[NodeIdentity] = None,
    ) -> Dict[str, Any]:
        verifier = verifier_identity or self._default_verifier
        enforced_quorum = max(2, int(quorum_size))

        valid_subs: List[Dict[str, Any]] = []
        seen_nodes = set()
        rejected_count = 0

        for s in submissions:
            if not isinstance(s, dict):
                rejected_count += 1
                continue
            node_id = str(s.get("node_id", "")).strip()
            if not node_id or node_id in seen_nodes:
                rejected_count += 1
                continue
            if not self._verify_submission_cryptographically(task_id, s, verifier):
                rejected_count += 1
                continue
            seen_nodes.add(node_id)
            valid_subs.append(s)

        quorum_reached = len(valid_subs) >= enforced_quorum

        if not valid_subs:
            return {
                "task_id": task_id,
                "quorum_reached": False,
                "enforced_quorum_size": enforced_quorum,
                "participating_nodes": 0,
                "rejected_submissions": rejected_count,
                "independent_evidence_sources": 0,
                "status": "INSUFFICIENT_VERIFIED_SUBMISSIONS",
                "accepted_findings": [],
                "conflicts": [],
                "confidence": 0.0,
            }

        # Evaluate evidence independence (distinct provenance hashes & distinct node_ids)
        unique_nodes = {s["node_id"] for s in valid_subs}
        provenance_hashes = set()
        symbolic_verified_nodes = set()
        evidence_backed = []
        unbacked_majority = []

        for s in valid_subs:
            ev_list = s.get("evidence", [])
            is_sym = bool(s.get("symbolic_verified", False))
            if is_sym:
                symbolic_verified_nodes.add(s["node_id"])
            has_proof = bool(ev_list) or is_sym
            for ev in ev_list:
                if isinstance(ev, dict):
                    h = ev.get("provenance_hash") or hashlib.sha256(
                        str(ev.get("excerpt", "")).encode("utf-8")
                    ).hexdigest()[:16]
                    provenance_hashes.add(h)
            if has_proof:
                evidence_backed.append(s)
            else:
                unbacked_majority.append(s)

        # Independent corroboration requires >= 2 distinct provenance hashes OR >= 2 independent
        # nodes performing deterministic symbolic verification.
        independent_source_count = max(len(provenance_hashes), len(symbolic_verified_nodes))
        has_independent_corroboration = (
            len(evidence_backed) >= enforced_quorum and independent_source_count >= 2
        )

        # Detect conflicts among conclusions
        conclusions_by_node = {s["node_id"]: str(s.get("conclusion", "")).strip() for s in valid_subs}
        conflicts = []
        nodes_list = list(conclusions_by_node.keys())
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                na, nb = nodes_list[i], nodes_list[j]
                ca, cb = conclusions_by_node[na], conclusions_by_node[nb]
                if ca and cb and ca.lower() != cb.lower():
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

        # Rule: Evidence-backed submission overrides unverified majority agreement,
        # but VERIFIED_BY_EVIDENCE_AND_QUORUM strictly requires independent corroboration.
        if evidence_backed:
            best = max(evidence_backed, key=lambda s: float(s.get("confidence", 0.5)))
            if quorum_reached and has_independent_corroboration and not conflicts:
                status = "VERIFIED_BY_EVIDENCE_AND_QUORUM"
                confidence = round(min(0.98, float(best.get("confidence", 0.8))), 4)
            elif quorum_reached and not has_independent_corroboration:
                status = "CORROBORATED_SINGLE_SOURCE"
                confidence = round(min(0.79, float(best.get("confidence", 0.75))), 4)
            else:
                status = "PARTIALLY_VERIFIED"
                confidence = round(min(0.75, float(best.get("confidence", 0.7))), 4)
        else:
            best = valid_subs[0]
            status = "UNVERIFIED_CONSENSUS_NO_PRIMARY_EVIDENCE"
            confidence = 0.35

        return {
            "task_id": task_id,
            "quorum_reached": quorum_reached,
            "enforced_quorum_size": enforced_quorum,
            "participating_nodes": len(unique_nodes),
            "rejected_submissions": rejected_count,
            "independent_evidence_sources": independent_source_count,
            "independent_corroboration": has_independent_corroboration,
            "status": status,
            "selected_node_id": best["node_id"],
            "selected_conclusion": best.get("conclusion", ""),
            "confidence": confidence,
            "conflicts": conflicts,
            "unverified_majority_rejected": len(unbacked_majority) > len(evidence_backed) and len(evidence_backed) > 0,
        }
