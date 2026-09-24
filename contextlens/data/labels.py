"""Turn category-graph membership into training labels.

Rules (documented in docs/DATASET_CARD.md and decisions.md):

1. An article's *general topic* is the general topic of the subtopic crawl
   that reached it at the smallest category depth. If two different general
   topics tie at that depth, the article is ambiguous and excluded from
   training (it is kept in the manifest for analysis).
2. Its *subtopics* (multi-label) are the subtopics of that general topic that
   reached it at that minimum depth, plus any other subtopic of the same
   general topic that reached it at depth <= 1. Deeper secondary memberships
   are ignored because the category graph gets noisy with depth.
3. Each subtopic contributes at most ``cap`` articles, preferring shallow
   (more on-topic) articles; ties are broken by a stable hash of the title so
   the selection is deterministic.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

from contextlens.data.dbpedia import SparqlClient, articles_in, crawl_categories
from contextlens.taxonomy import Subtopic, Taxonomy

SECONDARY_LABEL_MAX_DEPTH = 1


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()


@dataclass
class LabelledArticle:
    title: str
    general: str
    subtopics: list[str]
    depth: int  # minimum depth over the article's subtopics


def exclusion_patterns(taxonomy: Taxonomy, sub: Subtopic) -> list[re.Pattern[str]]:
    raw = taxonomy.raw
    patterns = list(raw["global_exclude_category_patterns"]) + list(sub.exclude_patterns)
    for group in sub.exclude:
        patterns += raw[f"{group}_exclude_patterns"]
    return [re.compile(p) for p in patterns]


def crawl_membership(taxonomy: Taxonomy, client: SparqlClient) -> tuple[dict[str, dict[str, int]], dict[str, dict]]:
    """Return ``membership[title][subtopic] = depth`` and per-subtopic crawl stats."""
    membership: dict[str, dict[str, int]] = defaultdict(dict)
    stats: dict[str, dict] = {}
    for general in taxonomy.generals:
        for sub in general.subtopics:
            patterns = exclusion_patterns(taxonomy, sub)
            n_cats = 0
            excluded: dict[str, str] = {}
            found: dict[str, int] = {}
            for seed, max_depth in sub.seeds:
                tree = crawl_categories(client, seed, max_depth, patterns)
                n_cats += len(tree.depth_of)
                excluded.update(tree.excluded)
                for title, depth in articles_in(client, tree).items():
                    found[title] = min(depth, found.get(title, depth))
            for title, depth in found.items():
                membership[title][sub.id] = depth
            stats[sub.id] = {
                "categories": n_cats,
                "excluded_categories": len(excluded),
                "articles_reached": len(found),
                "by_depth": {str(d): sum(1 for v in found.values() if v == d) for d in range(3)},
            }
    return dict(membership), stats


def assign_labels(taxonomy: Taxonomy, membership: dict[str, dict[str, int]]) -> tuple[list[LabelledArticle], list[str]]:
    """Apply rules 1-2. Returns (labelled articles, ambiguous titles)."""
    labelled: list[LabelledArticle] = []
    ambiguous: list[str] = []
    for title in sorted(membership):
        subs = membership[title]
        best_by_general: dict[str, int] = {}
        for sid, depth in subs.items():
            gid = taxonomy.parent_of(sid)
            best_by_general[gid] = min(depth, best_by_general.get(gid, depth))
        best = min(best_by_general.values())
        winners = [g for g, d in best_by_general.items() if d == best]
        if len(winners) > 1:
            ambiguous.append(title)
            continue
        general = winners[0]
        chosen = sorted(
            s
            for s, d in subs.items()
            if taxonomy.parent_of(s) == general and (d == best or d <= SECONDARY_LABEL_MAX_DEPTH)
        )
        labelled.append(LabelledArticle(title, general, chosen, best))
    return labelled, ambiguous


def select_balanced(
    taxonomy: Taxonomy,
    labelled: list[LabelledArticle],
    membership: dict[str, dict[str, int]],
    cap: int,
    exclude_titles: set[str] | None = None,
) -> list[LabelledArticle]:
    """Apply rule 3: cap each subtopic, shallow articles first, deterministic."""
    exclude_titles = exclude_titles or set()
    by_sub: dict[str, list[LabelledArticle]] = defaultdict(list)
    for art in labelled:
        if art.title in exclude_titles:
            continue
        for sid in art.subtopics:
            by_sub[sid].append(art)
    selected: dict[str, LabelledArticle] = {}
    for sid in taxonomy.subtopic_ids:
        ranked = sorted(by_sub[sid], key=lambda a: (membership[a.title][sid], stable_hash(a.title)))
        for art in ranked[:cap]:
            selected[art.title] = art
    return [selected[t] for t in sorted(selected)]
