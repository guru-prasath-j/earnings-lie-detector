"""
Rule-based guidance features for one earnings press release — self-contained (``re`` only),
so this file can be shipped next to the model on the Hugging Face Hub.

Sentences that mention guidance / outlook / forecast (minus boilerplate) are found, and in
each one the direction verb *closest* to that noun (within ``MAX_VERB_GAP`` characters)
decides whether guidance was raised, cut, maintained, narrowed, (re)introduced or withdrawn.
``outlook_hedge_rate`` counts hedging words per 1,000 words of outlook sentences.

    from guidance_rules import GUIDANCE_COLS, guidance_vector
    feats = guidance_vector(release_text)            # {"has_outlook": 1.0, "g_raise": 2.0, ...}

Used by ``text_features.guidance_features`` for training, so training and inference share one copy.
"""

from __future__ import annotations

import re
from typing import Dict

GUIDANCE_COLS = [
    "has_outlook", "n_outlook", "g_raise", "g_cut", "g_maintain", "g_narrow", "g_initiate", "g_withdraw",
    "g_net", "outlook_hedge_rate",
]
_WORD_RE = re.compile(r"[a-z]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"(•▪-])")
_OUTLOOK_RE = re.compile(r"\b(guidance|outlook|forecast)\b", re.IGNORECASE)
# Direction verbs; the one *closest* to the outlook noun decides the sentence.  Verb forms only:
# bare "increase" / "lower" are mostly nouns/adjectives ("an increase of 2%", "lower than expected").
_DIRECTION_RES = {
    "raise": re.compile(r"\b(rais(e|es|ed|ing)|increas(ed|ing)|increases? (its|our|the)|boost\w*|lift(s|ed|ing)"
                        r"|above (the )?(previous|prior))\b", re.I),
    "cut": re.compile(r"\b(lower(s|ed|ing)|lower (its|our|the)|reduc(es|ed|ing)|cut(s|ting)?"
                      r"|below (the )?(previous|prior))\b", re.I),
    "maintain": re.compile(r"\b(reaffirm\w*|maintain\w*|reiterat\w*|confirm\w*|unchanged|on track)\b", re.I),
    "narrow": re.compile(r"\bnarrow\w*\b", re.I),
    "initiate": re.compile(r"\b(initiat\w*|reinstat\w*|introduc\w*|issu(e|es|ed|ing))\b", re.I),
    "withdraw": re.compile(r"\b(withdr\w*|suspend\w*|no longer provid\w*|not provid\w*)\b", re.I),
}
_BOILERPLATE_RE = re.compile(r"accounting guidance|forward-looking|forward looking|undertake|not necessarily update|"
                             r"form 10-k|safe harbor|risks and uncertainties|reconciliation|unreasonable effort|"
                             r"tax rate", re.IGNORECASE)
HEDGE_WORDS = {"uncertain", "uncertainty", "uncertainties", "volatile", "volatility", "headwind", "headwinds",
               "challenging", "cautious", "softness", "softer", "pressure", "pressures", "slowdown", "weak",
               "weaker", "weakness", "macroeconomic", "tariff", "tariffs", "risk", "risks"}
MAX_VERB_GAP = 40  # characters between the direction verb and "guidance/outlook/forecast"


def guidance_vector(text: str) -> Dict[str, float]:
    """``GUIDANCE_COLS`` for one document.  ``g_net`` = sign(raise − cut)."""
    counts = {k: 0 for k in _DIRECTION_RES}
    n_outlook, hedge, words = 0, 0, 0
    for sent in _SENT_RE.split(text):
        if len(sent) > 800 or _BOILERPLATE_RE.search(sent):
            continue
        nouns = [m.start() for m in _OUTLOOK_RE.finditer(sent)]
        if not nouns:
            continue
        n_outlook += 1
        toks = _WORD_RE.findall(sent.lower())
        words += len(toks)
        hedge += sum(t in HEDGE_WORDS for t in toks)
        best = None  # (distance, direction)
        for direction, pat in _DIRECTION_RES.items():
            for m in pat.finditer(sent):
                dist = min(abs(m.start() - n) for n in nouns)
                if dist <= MAX_VERB_GAP and (best is None or dist < best[0]):
                    best = (dist, direction)
        if best:
            counts[best[1]] += 1
    diff = counts["raise"] - counts["cut"]
    return {
        "has_outlook": float(n_outlook > 0), "n_outlook": float(n_outlook),
        **{f"g_{k}": float(v) for k, v in counts.items()},
        "g_net": float((diff > 0) - (diff < 0)),
        "outlook_hedge_rate": 1000 * hedge / words if words else 0.0,
    }
