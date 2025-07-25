from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

try:
    import polars as pl
except ImportError:  # pragma: no cover - polars may not be installed
    pl = None

try:
    from sqlalchemy import create_engine
except ImportError:  # pragma: no cover - sqlalchemy may not be installed
    create_engine = None

class Load:
    """Load data from various resources into a polars LazyFrame."""

    def __init__(self, resource: str, query: Optional[str] = None, **kwargs: Any) -> None:
        self.resource = resource
        self.query = query
        self.kwargs = kwargs

    def _read_local(self, path: Path):
        ext = path.suffix.lower()
        if pl is None:
            raise ImportError('polars is required for Load step')
        if ext == '.csv':
            return pl.read_csv(str(path), **self.kwargs).lazy()
        elif ext in {'.xls', '.xlsx'}:
            df = pd.read_excel(path, **{k: v for k, v in self.kwargs.items() if k != 'n_rows'})
            return pl.from_pandas(df).lazy()
        else:
            raise ValueError(f'Unsupported file extension: {ext}')

    def _read_url(self, url: str):
        ext = Path(url).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            resp = requests.get(url)
            resp.raise_for_status()
            tmp.write(resp.content)
            tmp.flush()
            return self._read_local(Path(tmp.name))

    def _read_db(self, uri: str):
        if create_engine is None:
            raise ImportError('sqlalchemy is required for database loading')
        engine = create_engine(uri)
        query = self.query or self.kwargs.get('query') or 'SELECT 1'
        if pl is None:
            raise ImportError('polars is required for Load step')
        return pl.read_database(query, engine).lazy()

    def run(self, data: Any = None):
        path = Path(self.resource)
        if path.exists():
            return self._read_local(path)
        if self.resource.startswith('http://') or self.resource.startswith('https://'):
            return self._read_url(self.resource)
        if '://' in self.resource:
            return self._read_db(self.resource)
        raise ValueError(f'Cannot load resource: {self.resource}')
