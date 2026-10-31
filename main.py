"""Compatibility shim — the canonical entry point is server.py (server:app)."""
import importlib.util as _ilu
from pathlib import Path as _P
_spec = _ilu.spec_from_file_location("_legitmate_server", _P(__file__).parent / "app.py")
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)
app = _m.app
