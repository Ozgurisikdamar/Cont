"""Interactive console for ContextLens.

Commands: ``exit`` / ``quit`` / ``q`` (leave), ``history`` (this session's
messages and classifications), ``reset`` (new conversation context - stored
records are kept), ``help``. Technical detail goes to the log (``--debug``),
not to the screen.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from contextlens.config import PATHS, load_settings
from contextlens.database.db import Database, DatabaseError
from contextlens.logging_setup import setup_logging
from contextlens.models.artifact import ArtifactError, load_artifact, load_vocabulary
from contextlens.services.pipeline import ConversationSession, TurnResult
from contextlens.services.websearch import WebSearcher
from contextlens.taxonomy import load_taxonomy

log = logging.getLogger("contextlens")

EXIT_COMMANDS = {"exit", "quit", "q"}
HELP = """Commands:
  history   show this conversation's messages and their classifications
  reset     start a new conversation (saved records are kept)
  help      show this help
  exit, q   quit"""
SEARCH_STATUS = {
    "offline": "Web search is unavailable right now (offline or provider error); analysis was still saved.",
    "no_results": "No web results found for this query.",
    "disabled": "Web search is disabled (--no-web).",
}


def pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def render_turn(result: TurnResult, tax_display: Callable[[str], str], print_: Callable[[str], None]) -> None:
    pred = result.prediction
    if pred.status == "uninformative":
        print_("\nNothing topical to analyse - please type a sentence about a subject.")
        return
    print_("\n[Text analysis]")
    print_(f"General topic : {tax_display(pred.general or '')}")
    print_(f"Confidence    : {pct(pred.confidence)}")
    print_("Subtopics     :")
    for s in pred.subtopics:
        print_(f"  - {tax_display(s.id)} ({pct(s.probability)})")
    if pred.uncertain:
        print_("Note          : uncertain - " + "; ".join(pred.reasons) + ".")
        print_("                This message was not added to the conversation theme.")
    print_("\n[Conversation]")
    print_(f"Theme         : {result.theme.label}")
    if result.theme.phrase:
        print_(f"In words      : {result.theme.phrase}")
    if result.search is not None:
        print_(f'Search query  : "{result.search.query}"')
        print_("\n[Web results]")
        if result.search.results:
            for r in result.search.results:
                print_(f"{r.rank}. {r.title} ({r.source})")
                if r.summary:
                    print_(f"   {r.summary}")
                print_(f"   {r.url}")
        else:
            print_(SEARCH_STATUS.get(result.search.status, "No results."))
    for w in result.warnings:
        print_(f"Warning: {w}")
    if not any("database" in w for w in result.warnings):
        print_("\nSaved to the database.")


def render_history(
    session: ConversationSession, tax_display: Callable[[str], str], print_: Callable[[str], None]
) -> None:
    records = session.history()
    if not records:
        print_("No messages in this conversation yet.")
        return
    for r in records:
        subs = ", ".join(f"{tax_display(s['id'])} {pct(s['probability'])}" for s in r.subtopics)
        topic = tax_display(r.general_topic) if r.general_topic else "-"
        conf = pct(r.general_confidence or 0.0)
        flag = " [uncertain]" if r.uncertain else ""
        print_(f"{r.turn:>3}. {r.content}\n     -> {topic} ({conf}){flag}; {subs}")
    print_(f"Current theme: {session.current_theme().label}")


def run_loop(session: ConversationSession, lines: Iterable[str] | None, print_: Callable[[str], None]) -> int:
    tax = session.tax
    display = tax.display
    source = iter(lines) if lines is not None else None
    while True:
        try:
            if source is None:
                text = input("\nEnter text (or 'help'): ")
            else:
                text = next(source)
                print_(f"\n> {text}")
        except (EOFError, StopIteration):
            print_("\nGoodbye.")
            return 0
        except KeyboardInterrupt:
            print_("\nInterrupted - goodbye.")
            return 0
        command = text.strip().lower()
        if command in EXIT_COMMANDS:
            print_("Goodbye.")
            return 0
        if command == "help":
            print_(HELP)
            continue
        if command == "history":
            render_history(session, display, print_)
            continue
        if command == "reset":
            session.reset()
            print_("Conversation reset. Previous records remain in the database.")
            continue
        if not command:
            print_("Please type a sentence (or 'help').")
            continue
        render_turn(session.process(text), display, print_)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="project.py", description="ContextLens - conversation-aware topic analysis")
    p.add_argument("--model-dir", type=Path, default=None, help="model artifact directory")
    p.add_argument("--db", type=Path, default=None, help="SQLite database path")
    p.add_argument("--no-web", action="store_true", help="disable web search (fully local)")
    p.add_argument("--debug", action="store_true", help="verbose logging to stderr and logs/contextlens.log")
    p.add_argument("--script", type=Path, help="read messages from a file (one per line) instead of the keyboard")
    p.add_argument("--once", metavar="TEXT", help="analyse a single message and exit")
    return p


def main(argv: list[str] | None = None, print_: Callable[[str], None] = print) -> int:
    args = build_arg_parser().parse_args(argv)
    overrides: dict[str, object] = {}
    if args.model_dir:
        overrides["model_dir"] = args.model_dir
    if args.db:
        overrides["db_path"] = args.db
    if args.no_web:
        overrides["web_enabled"] = False
    settings = load_settings(**overrides)
    setup_logging(
        "DEBUG" if args.debug else settings.log_level, PATHS.root / "logs" / "contextlens.log" if args.debug else None
    )

    print_("ContextLens - English conversation topic analysis")
    print_("Loading model...")
    t0 = time.perf_counter()
    try:
        model = load_artifact(
            settings.model_dir, min_confidence=settings.min_confidence, max_subtopics=settings.max_subtopics
        )
    except ArtifactError as exc:
        print_(f"Error: {exc}")
        return 2
    print_(f"Model ready (v{model.version}, {time.perf_counter() - t0:.1f}s).")

    db: Database | None
    try:
        db = Database(settings.db_path)
    except DatabaseError as exc:
        print_(f"Warning: database unavailable ({exc}); continuing without saving.")
        db = None
    searcher = WebSearcher(
        cache=db,
        enabled=settings.web_enabled,
        timeout=settings.web_timeout,
        retries=settings.web_retries,
        backoff=settings.web_backoff,
        max_results=settings.max_results,
        cache_ttl_hours=settings.cache_ttl_hours,
    )
    session = ConversationSession(model, load_taxonomy(), settings, db, searcher, load_vocabulary(settings.model_dir))
    try:
        if args.once is not None:
            return run_loop(session, [args.once], print_)
        lines = None
        if args.script is not None:
            try:
                raw = args.script.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                print_(f"Error: cannot read script file {args.script}: {exc}")
                return 2
            lines = [ln for ln in raw.splitlines() if ln.strip()]
        if lines is None:
            print_(HELP)
        return run_loop(session, lines, print_)
    finally:
        session.close()
        if db is not None:
            db.close()


if __name__ == "__main__":
    sys.exit(main())
