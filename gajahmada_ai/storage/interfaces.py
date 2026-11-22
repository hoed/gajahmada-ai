from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class StorageBackend(ABC):
    """Abstract persistent storage interface for GajahMada 2.0 Memory & Knowledge."""

    @abstractmethod
    def save_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def list_records(
        self,
        memory_type: Optional[str] = None,
        verification_status: Optional[str] = None,
        include_deleted: bool = False,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_record(self, record_id: str, hard: bool = False) -> bool:
        pass

    @abstractmethod
    def save_version(self, version_entry: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_versions(self, record_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def save_contradiction(self, contradiction: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def list_contradictions(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def save_relation(self, relation: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def list_relations(self, entity: Optional[str] = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        pass
