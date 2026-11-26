# GajahMada AI — Memory Security Model (`MEMORY_SECURITY_MODEL.md`)

## Maturity Classification
- **Canonical Immutable `AccessContext` (`gajahmada_ai/security/context.py`):** **IMPLEMENTED AND TESTED**
- **Direct `MemoryManager.retrieve()` Scope Authorization:** **IMPLEMENTED AND TESTED**
- **9 Visibility Scopes (`PRIVATE`, `SESSION`, `AGENT`, `ORGANIZATION`, `LOCAL`, `SHARED`, `DISTRIBUTED_SHARED`, `COLLECTIVE`, `PUBLIC`):** **IMPLEMENTED AND TESTED**
- **Privacy-Safe `MEMORY_ACCESS_DENIED` Audit Logging:** **IMPLEMENTED AND TESTED**

## Enforcement Boundaries
Authorization is enforced at **both** the API layer and directly inside `MemoryManager.retrieve()` and `HybridKnowledgeRetriever.retrieve()` via `MemoryAuthorizationService` and `KnowledgeAccessPolicy`:
- **Fail-Closed Default:** Missing `AccessContext` never grants access to `PRIVATE`, `SESSION`, `AGENT`, or `ORGANIZATION` records.
- **Cross-User Isolation:** `PRIVATE` records require matching `owner_actor_id` (or `admin`).
- **Cross-Session Isolation:** `SESSION` records require matching `session_id` (or `admin`).
- **Cross-Agent Isolation:** `AGENT` records require matching `agent_id` (or `admin`).
- **Cross-Organization Isolation:** `ORGANIZATION` records require matching `organization_id` (or `admin`).
- **Log Hygiene:** `MEMORY_ACCESS_DENIED` events and `LifecycleObservabilityLogger` scrub private content and prompts and hash non-role actor IDs.
