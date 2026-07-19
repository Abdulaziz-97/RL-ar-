from __future__ import annotations
import importlib
import importlib.util
from pathlib import Path
from typing import Any

def load_external_module(module_path: str, attr: str='build_backend'):
    path = Path(module_path)
    if path.exists() and path.suffix == '.py':
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot load external module: {module_path}')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(module_path)
    if not hasattr(mod, attr):
        raise AttributeError(f'{module_path} missing {attr}()')
    return getattr(mod, attr)

class ExternalModuleBackend:

    def __init__(self, module_path: str, config: dict[str, Any] | None=None):
        factory = load_external_module(module_path)
        backend = factory(config or {})
        self.problem_generator = backend.problem_generator
        self.trace_teacher = backend.trace_teacher
        self.verifier = backend.verifier
        self.arabic_editor = backend.arabic_editor
