"""Explicit fixture adapter. Never registered implicitly for production."""
import json
from pathlib import Path
from .registry import DatasetAdapter
from .schema import AdapterMetadata, NormalizedCase

class SyntheticFixtureAdapter(DatasetAdapter):
    metadata = AdapterMetadata(source="synthetic_fixture", version="1", license_status="test fixture")
    def __init__(self, path):
        self.path = Path(path)
    def load(self):
        rows = [NormalizedCase(**row) for row in json.loads(self.path.read_text())]
        if any(not r.synthetic for r in rows):
            raise ValueError("Every fixture must explicitly declare synthetic=true")
        return rows
