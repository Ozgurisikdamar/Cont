"""Compose the active conversation topics into one human-readable theme.

Instead of a hand-written table of topic pairs ("Books + Science -> science
books", "Books + History -> history books", ...), composition is driven by
three properties declared once per topic in ``configs/taxonomy.json``:

* ``role: format`` - a *medium* (Books) that wraps whatever domain is being
  discussed: ``{domain} books``.
* ``broader`` - a domain that is a specialisation of another active domain
  (Biology is narrower than Science). The narrower one becomes the focus:
  ``science books about biology``.
* ``perspective_template`` - a domain that frames another one
  (History: ``history of {topic}``).

Adding a new topic therefore needs only its metadata, not new pair rules.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextlens.services.tracker import Theme, ThemeTopic
from contextlens.taxonomy import Taxonomy

# A single general topic is represented by its top subtopic when that subtopic
# dominates the topic's subtopic mass.
SUBTOPIC_FOCUS_SHARE = 0.5


@dataclass(frozen=True)
class ComposedTheme:
    label: str  # e.g. "Books + Science + Biology" or "Technology > Quantum Computing"
    phrase: str  # natural-language theme, used for the search query
    generals: tuple[str, ...]
    subtopics: tuple[str, ...]


EMPTY_THEME = ComposedTheme(label="(no theme yet)", phrase="", generals=(), subtopics=())


def _focus_subtopic(theme: Theme, gid: str) -> ThemeTopic | None:
    subs = theme.subtopics.get(gid, [])
    if subs and subs[0].share >= SUBTOPIC_FOCUS_SHARE:
        return subs[0]
    return None


def _specific_phrase(tax: Taxonomy, theme: Theme, gid: str) -> str:
    sub = _focus_subtopic(theme, gid)
    return tax.subtopic(sub.id).phrase if sub else tax.general(gid).phrase


def compose(theme: Theme, tax: Taxonomy) -> ComposedTheme:
    if theme.is_empty:
        return EMPTY_THEME
    tpl = tax.raw["composition"]
    active = [g.id for g in theme.generals]
    formats = [g for g in active if tax.general(g).role == "format"]
    domains = [g for g in active if tax.general(g).role == "domain"]
    focus_subs = tuple(s.id for g in active if (s := _focus_subtopic(theme, g)) is not None)

    # Narrowing: a domain whose broader topic is also active.
    narrow = next((d for d in domains if tax.general(d).broader in domains), None)
    broader = tax.general(narrow).broader if narrow else None

    if len(active) == 1:
        gid = active[0]
        sub = _focus_subtopic(theme, gid)
        label = tax.general(gid).name + (f" > {tax.subtopic(sub.id).name}" if sub else "")
        return ComposedTheme(label, _specific_phrase(tax, theme, gid), (gid,), focus_subs)

    if formats and domains:
        fmt = tax.general(formats[0]).phrase
        if narrow and broader:
            phrase = tpl["format_narrow_template"].format(
                broader=tax.general(broader).phrase, format=fmt, narrow=tax.general(narrow).phrase
            )
        else:
            domain_phrase = (
                tpl["pair_template"].format(a=tax.general(domains[0]).phrase, b=tax.general(domains[1]).phrase)
                if len(domains) > 1
                else tax.general(domains[0]).phrase
            )
            phrase = tpl["format_template"].format(domain=domain_phrase, format=fmt)
    elif narrow and broader:
        phrase = tpl["narrow_template"].format(
            narrow=_specific_phrase(tax, theme, narrow), broader=tax.general(broader).phrase
        )
    else:
        perspective = next((d for d in domains if tax.general(d).perspective_template), None)
        others = [d for d in domains if d != perspective]
        if perspective and others:
            template = tax.general(perspective).perspective_template or "{topic}"
            phrase = template.format(topic=_specific_phrase(tax, theme, others[0]))
        else:
            phrase = tpl["pair_template"].format(
                a=_specific_phrase(tax, theme, domains[0]), b=_specific_phrase(tax, theme, domains[1])
            )
    label = " + ".join(tax.general(g).name for g in active)
    return ComposedTheme(label, phrase, tuple(active), focus_subs)
