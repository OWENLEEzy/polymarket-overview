from app.analysis.parsers import (
    MoneyBracket,
    canonical_unit,
    parse_deadline_date,
    parse_money_bracket,
    to_billions,
)


def test_parses_closed_range() -> None:
    assert parse_money_bracket("$1.25–$1.5T") == MoneyBracket(lower=1.25, upper=1.5, unit="T")


def test_parses_open_upper_tail() -> None:
    assert parse_money_bracket("$3.0T+") == MoneyBracket(lower=3.0, upper=None, unit="T")


def test_parses_open_lower_tail() -> None:
    assert parse_money_bracket("<$1.25T") == MoneyBracket(lower=None, upper=1.25, unit="T")


def test_parses_billions_with_hyphen() -> None:
    assert parse_money_bracket("600B+") == MoneyBracket(lower=600.0, upper=None, unit="B")


def test_returns_none_for_non_numeric() -> None:
    assert parse_money_bracket("No IPO") is None
    assert parse_money_bracket("No IPO by December 31, 2027") is None


def test_canonical_unit_maps_verbose_forms() -> None:
    assert canonical_unit("USD billion") == "B"
    assert canonical_unit("USD trillion") == "T"
    assert canonical_unit("USD million") == "M"


def test_to_billions_converts_units() -> None:
    assert to_billions(1.5, "T") == 1500.0
    assert to_billions(600.0, "B") == 600.0
    assert to_billions(500.0, "M") == 0.5


def test_parses_full_month_name() -> None:
    assert parse_deadline_date("Will Anthropic IPO by October 31, 2026?") == "2026-10-31"


def test_parses_abbreviated_month() -> None:
    assert parse_deadline_date("IPO by Oct 31, 2026") == "2026-10-31"


def test_parses_no_ipo_deadline() -> None:
    assert parse_deadline_date("No IPO by December 31, 2027") == "2027-12-31"


def test_returns_none_without_date() -> None:
    assert parse_deadline_date("Anthropic valuation above $100B") is None


def test_parses_iso_format() -> None:
    assert parse_deadline_date("Resolves 2026-06-30") == "2026-06-30"


def test_parses_day_first_format() -> None:
    assert parse_deadline_date("by 30 June 2026") == "2026-06-30"


def test_returns_none_for_empty() -> None:
    assert parse_deadline_date(None) is None
    assert parse_deadline_date("") is None


def test_returns_none_for_money_without_year() -> None:
    # The 4-digit-year guard stops "$100B" being fuzzy-parsed as a date.
    assert parse_deadline_date("Will Anthropic IPO above $100B?") is None
