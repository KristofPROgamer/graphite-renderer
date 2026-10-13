# Wrapper to preload stdlib 'types' module and run demo_module1
import sys
import sysconfig
import importlib.util
from pathlib import Path
import runpy

try:
    stdlib = sysconfig.get_paths()['stdlib']
    types_path = Path(stdlib) / 'types.py'
    if types_path.exists():
        spec = importlib.util.spec_from_file_location('types', str(types_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules['types'] = mod
except Exception:
    pass

# Ensure workspace on sys.path
import os
proj = os.path.abspath(os.path.dirname(__file__))
if proj not in sys.path:
    sys.path.insert(0, proj)

runpy.run_path(os.path.join(proj, 'scripts', 'demo_module1.py'), run_name='__main__')
