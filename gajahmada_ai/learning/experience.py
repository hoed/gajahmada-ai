from typing import Any, Dict, List

from gajahmada_ai.memory.manager import MemoryManager
from gajahmada_ai.memory.models import MemoryType, VerificationStatus


class ExperienceExtractor:
    """
    Level 1 & Level 2 Learning:
    - Level 1: Validated knowledge acquisition into Semantic Memory.
    - Level 2: Extracts high-confidence Episodic experiences into reusable
      Procedural Memory strategies.
    """

    def __init__(self, memory: MemoryManager):
        self.memory = memory

    def extract_procedure_from_episode(
        self,
        objective_pattern: str,
        validated_steps: List[str],
        confidence: float = 0.85,
    ) -> Dict[str, Any]:
        proc_text = f"PROCEDURE [{objective_pattern}]: " + " -> ".join(validated_steps)
        return self.memory.store(
            content=proc_text,
            memory_type=MemoryType.PROCEDURAL,
            source_type="evaluated_experience_level2",
            author="learning.experience",
            confidence=confidence,
            verification_status=VerificationStatus.VERIFIED,
        )
