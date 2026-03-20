"""Load a hand-implemented strategy class by JSON name or module name."""
from __future__ import annotations
import importlib
import logging
from pathlib import Path
from typing import Optional

from pipeline.strategies.base import BaseStrategy

log = logging.getLogger(__name__)

# Directory containing strategy modules
_STRATEGY_DIR = Path(__file__).parent

# Cache of loaded strategy instances (keyed by module name)
_cache: dict[str, BaseStrategy] = {}

# Mapping from strategy JSON name → module name (built lazily)
_json_name_to_module: dict[str, str] = {}
_name_map_built = False


def _build_name_map():
    """Scan all strategy modules and build JSON-name → module-name mapping."""
    global _name_map_built
    if _name_map_built:
        return
    for f in _STRATEGY_DIR.glob("*.py"):
        if f.name.startswith("_") or f.name in ("base.py", "loader.py", "smoke_test.py"):
            continue
        mod_name = f.stem
        try:
            mod = importlib.import_module(f"pipeline.strategies.{mod_name}")
            strategy = mod.Strategy()
            _cache[mod_name] = strategy
            # Map both the strategy's .name attribute and the module name
            _json_name_to_module[strategy.name] = mod_name
            _json_name_to_module[mod_name] = mod_name
        except Exception:
            pass  # Skip broken modules during discovery
    _name_map_built = True


def load_strategy(name: str) -> Optional[BaseStrategy]:
    """Load a strategy by module name OR JSON name.

    Tries direct module import first (fast path), then falls back to
    the JSON-name → module-name mapping.

    Returns a BaseStrategy instance, or None if not found.
    """
    # Fast path: already cached
    if name in _cache:
        return _cache[name]

    # Try direct module import (module name matches)
    module_path = f"pipeline.strategies.{name}"
    try:
        mod = importlib.import_module(module_path)
        strategy = mod.Strategy()
        _cache[name] = strategy
        _json_name_to_module[strategy.name] = name
        _json_name_to_module[name] = name
        return strategy
    except (ModuleNotFoundError, AttributeError):
        pass

    # Slow path: build full name map and look up by JSON name
    _build_name_map()
    mod_name = _json_name_to_module.get(name)
    if mod_name and mod_name in _cache:
        return _cache[mod_name]

    return None


def list_available() -> list[str]:
    """Return sorted list of available strategy module names."""
    modules = []
    for f in _STRATEGY_DIR.glob("*.py"):
        if f.name.startswith("_") or f.name in ("base.py", "loader.py", "smoke_test.py"):
            continue
        modules.append(f.stem)
    return sorted(modules)
