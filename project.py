"""ContextLens interactive console - entry point.

    python project.py                 # interactive session
    python project.py --no-web        # fully local (no network calls)
    python project.py --once "Quantum processors use qubits to speed up algorithms."

Type ``help`` inside the console for commands (history, reset, exit/q).
"""

from contextlens.cli.app import main

if __name__ == "__main__":
    raise SystemExit(main())
