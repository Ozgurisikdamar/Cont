"""Record the machine and library versions the results were produced on.

    python scripts/hardware_info.py      # writes reports/hardware.json

Reported in docs/FINAL_REPORT.md (hardware section). Nothing is estimated: every
field is read from the running system.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlens.config import PATHS

PACKAGES = ("torch", "transformers", "sentence-transformers", "scikit-learn", "numpy", "pandas", "skops", "requests")


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def ram_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return None


def main() -> int:
    import torch

    info = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": ram_gb(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "packages": {p: metadata.version(p) for p in PACKAGES},
    }
    out = PATHS.reports / "hardware.json"
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
