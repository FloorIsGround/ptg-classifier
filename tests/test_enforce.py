from ptg.classify.enforce import parse_yes_no


def test_yes_lowercase():
    assert parse_yes_no("yes") == "yes"


def test_yes_uppercase():
    assert parse_yes_no("YES") == "yes"


def test_yes_mixed_case():
    assert parse_yes_no("Yes") == "yes"


def test_no_lowercase():
    assert parse_yes_no("no") == "no"


def test_no_uppercase():
    assert parse_yes_no("NO") == "no"


def test_yes_trailing_period():
    assert parse_yes_no("yes.") == "yes"


def test_no_trailing_period():
    assert parse_yes_no("no.") == "no"


def test_yes_with_whitespace():
    assert parse_yes_no("  yes  ") == "yes"


def test_invalid_returns_none():
    assert parse_yes_no("maybe") is None


def test_empty_returns_none():
    assert parse_yes_no("") is None


def test_prose_returns_none():
    assert parse_yes_no("Yes, this line exhibits the label.") is None
