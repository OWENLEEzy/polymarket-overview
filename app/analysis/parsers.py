import re
from dataclasses import dataclass
from datetime import datetime

from dateutil import parser as _dateparser

UNIT_TO_BILLIONS = {"B": 1.0, "T": 1000.0, "M": 0.001}

_VERBOSE_UNIT = {
    "t": "T",
    "trillion": "T",
    "usd trillion": "T",
    "b": "B",
    "bn": "B",
    "billion": "B",
    "usd billion": "B",
    "m": "M",
    "million": "M",
    "usd million": "M",
}

_NUMBER = r"\$?\s*(\d+(?:\.\d+)?)"
_UNIT = r"\s*(T|B|M|trillion|billion|million)?"
_RANGE_RE = re.compile(_NUMBER + r"\s*[-–—]\s*" + _NUMBER + _UNIT, re.IGNORECASE)
_OPEN_UPPER_RE = re.compile(_NUMBER + r"\s*(T|B|M|trillion|billion|million)?\s*\+", re.IGNORECASE)
_OPEN_LOWER_RE = re.compile(r"[<≤]\s*" + _NUMBER + _UNIT, re.IGNORECASE)
_SINGLE_RE = re.compile(_NUMBER + _UNIT, re.IGNORECASE)


@dataclass(frozen=True)
class MoneyBracket:
    lower: float | None
    upper: float | None
    unit: str


def canonical_unit(raw: str | None) -> str:
    if not raw:
        return "B"
    return _VERBOSE_UNIT.get(raw.strip().casefold(), "B")


def to_billions(value: float, unit: str) -> float:
    return value * UNIT_TO_BILLIONS.get(canonical_unit(unit), 1.0)


def parse_money_bracket(text: str | None) -> MoneyBracket | None:
    if not text:
        return None
    cleaned = text.strip()
    range_match = _RANGE_RE.search(cleaned)
    if range_match:
        unit = canonical_unit(range_match.group(3))
        return MoneyBracket(float(range_match.group(1)), float(range_match.group(2)), unit)
    open_upper = _OPEN_UPPER_RE.search(cleaned)
    if open_upper:
        return MoneyBracket(float(open_upper.group(1)), None, canonical_unit(open_upper.group(2)))
    open_lower = _OPEN_LOWER_RE.search(cleaned)
    if open_lower:
        return MoneyBracket(None, float(open_lower.group(1)), canonical_unit(open_lower.group(2)))
    single = _SINGLE_RE.search(cleaned)
    # Require a currency/unit cue so bare numbers in prose (e.g. the "31, 2027"
    # in "No IPO by December 31, 2027") are NOT mistaken for a money bracket.
    if single and ("$" in cleaned or single.group(2)):
        return MoneyBracket(
            float(single.group(1)), float(single.group(1)), canonical_unit(single.group(2))
        )
    return None


# Locate a date inside free-text questions/labels, then let dateutil parse only
# that substring. dateutil's fuzzy mode mis-assigns the year on full sentences
# (it drops "2026" from "Will ... by October 31, 2026?"), so we anchor on an
# explicit date shape first and parse the clean match without fuzzy. The regex
# only locates; dateutil still owns month-name semantics and day/year assembly,
# and rejects non-month words (e.g. "Ryan 2026" -> no date).
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DATE_PHRASE_RE = re.compile(
    r"[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}"  # October 31, 2026
    r"|\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{4}"  # 30 June 2026
    r"|[A-Za-z]{3,9}\.?\s+\d{4}",  # June 2026 (month bucket -> day 1)
    re.IGNORECASE,
)
# Missing day defaults deterministically to the 1st, never to the wall clock.
_DATE_DEFAULT = datetime(2000, 1, 1)


def parse_deadline_date(text: str | None) -> str | None:
    if not text:
        return None
    iso = _ISO_DATE_RE.search(text)
    if iso:
        year, month, day = (int(group) for group in iso.groups())
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None
    phrase = _DATE_PHRASE_RE.search(text)
    if not phrase:
        return None
    try:
        parsed = _dateparser.parse(phrase.group(0), default=_DATE_DEFAULT)
    except (ValueError, OverflowError, TypeError):
        return None
    return parsed.strftime("%Y-%m-%d")
