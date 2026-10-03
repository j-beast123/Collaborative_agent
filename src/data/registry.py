from abc import ABC, abstractmethod
from .schema import NormalizedCase, AdapterMetadata
from src.common.config import IntegrationUnavailable

class DatasetAdapter(ABC):
    @property
    @abstractmethod
    def metadata(self) -> AdapterMetadata:
        pass

    @abstractmethod
    def load(self):
        """Return normalized cases; do not download implicitly."""
        raise NotImplementedError

class DatasetRegistry:
    def __init__(self):
        self._adapters = {}

    def register(self, name, adapter, *, synthetic=False):
        if name in self._adapters:
            raise ValueError("Adapter already registered: " + name)
        if not isinstance(adapter, DatasetAdapter):
            raise TypeError("Expected DatasetAdapter")
        self._adapters[name] = (adapter, synthetic)

    def load(self, name, *, production=True):
        if name not in self._adapters:
            raise IntegrationUnavailable("Dataset adapter unavailable: " + name)
        adapter, synthetic = self._adapters[name]
        if production and synthetic:
            raise ValueError("Synthetic adapter cannot be used in production")
        rows = list(adapter.load())
        if not rows or any(not isinstance(r, NormalizedCase) or r.synthetic != synthetic for r in rows):
            raise ValueError("Empty dataset or adapter/record provenance mismatch")
        return rows

class AnswerEvaluator(ABC):
    def evaluate(self, case, response):
        if case.correct_answer is None:
            raise IntegrationUnavailable("Final success unavailable: task reference is missing")
        return self.score(case, response)

    @abstractmethod
    def score(self, case, response) -> bool:
        raise NotImplementedError
