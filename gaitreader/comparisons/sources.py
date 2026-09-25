"""Load original modules without collisions between their models/layers/utils packages."""
import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "benchmark_sources.json").read_text(encoding="utf-8"))


def fetch_sources():
    for name, source in MANIFEST.items():
        path = ROOT / ".benchmark_sources" / name
        if not path.exists():
            subprocess.run(["git", "clone", source["url"], str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", "--detach", source["commit"]], check=True)
        # Never reset a user's existing checkout.
        verify_source(name)


def verify_source(name):
    path = ROOT / ".benchmark_sources" / name
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != MANIFEST[name]["commit"]:
        raise ValueError(f"{name}: expected {MANIFEST[name]['commit']}, found {revision}")
    return path


def modules(name, *names, subdir=""):
    path = verify_source(name) / subdir
    roots = {p.stem if p.is_file() else p.name for p in path.iterdir()
             if p.suffix == ".py" or p.is_dir()}
    saved = {key: value for key, value in sys.modules.copy().items() if key.split('.')[0] in roots}
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, str(path))
    try:
        return tuple(importlib.import_module(name) for name in names)
    finally:
        sys.path.remove(str(path))
        for key in list(sys.modules):
            if key.split('.')[0] in roots:
                del sys.modules[key]
        sys.modules.update(saved)
