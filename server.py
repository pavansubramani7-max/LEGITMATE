"""Uvicorn entry point: uvicorn server:app --reload
Loads the FastAPI application from app.py without colliding with the app/ package.
"""
import importlib.util as _ilu
from pathlib import Path as _P

_spec = _ilu.spec_from_file_location("_legitmate_server", _P(__file__).parent / "app.py")
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)
app = _m.app
