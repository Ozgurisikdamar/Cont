"""Freeze the final configuration before the locked evaluation (decisions.md D-36).

    python scripts/freeze.py

Writes reports/locked/FREEZE.json: the git commit, the time, the runtime
settings and the SHA-256 of every file that defines the system (model config,
taxonomy, runtime config, training passages, locked-set manifest, artifact
metadata - which itself carries the checksums of every artifact file and the
encoder manifest). ``python evaluate.py --stage locked`` refuses to run if any
of these changed. Commit the freeze file before running the locked evaluation.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlens.config import PATHS, load_settings
from evaluate import LOCKED_DIR, freeze_fingerprint


def main() -> int:
    settings = load_settings()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 - git from PATH
        cwd=PATHS.root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607 - git from PATH
        cwd=PATHS.root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(f"commit your changes before freezing:\n{dirty}")
    if (LOCKED_DIR / "results.json").exists():
        raise SystemExit("the locked holdout was already evaluated; a new freeze would hide that")
    freeze = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        "settings": {k: str(v) for k, v in dataclasses.asdict(settings).items()},
        "files": freeze_fingerprint(settings.model_dir),
    }
    LOCKED_DIR.mkdir(parents=True, exist_ok=True)
    (LOCKED_DIR / "FREEZE.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps(freeze, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
