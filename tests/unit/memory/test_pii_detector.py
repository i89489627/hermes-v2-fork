"""P17.T4 acceptance: PII detection patterns.

Covers :class:`hermes.memory.pii_detector.PIIDetector.scan`/``redact``
for email, phone, api-key and credit-card (+Luhn) detection, SSN checks,
false-positive guards (dates, asset names, weak secrets) and caller-
supplied extra patterns.
"""

from __future__ import annotations

import pytest

from hermes.memory.pii_detector import PIIDetector, PIIPattern


@pytest.mark.unit
def test_detects_email() -> None:
    """Emails are found with exact spans, masked samples and redaction."""
    detector = PIIDetector()
    text = "contact john.doe@example.com about the contract"
    result = detector.scan(text)

    assert len(result.detected) == 1
    finding = result.detected[0]
    assert finding["type"] == "email"
    assert finding["match_span"] == (8, 28)
    assert text[8:28] == "john.doe@example.com"
    assert finding["sample_redacted"].startswith("jo")
    assert finding["sample_redacted"].endswith("om")
    assert "*" in finding["sample_redacted"]
    assert "john.doe" not in finding["sample_redacted"]  # never leaks the secret
    assert result.risk == pytest.approx(0.25)
    assert result.types == ["email"]

    # redaction replaces the span, not the surrounding text
    assert detector.redact(text) == "contact [EMAIL] about the contract"

    # multiple emails -> multiple findings, risk accumulates (capped at 1.0)
    two = detector.scan("a@corp.example and b@corp.example")
    assert [item["type"] for item in two.detected] == ["email", "email"]
    assert len(two.detected) == 2

    # false-positive guard: retina-style asset names are not emails
    assert detector.scan("use image@2x.png for the banner").detected == []

    # clean text: empty result, zero risk
    empty = detector.scan("nothing to see here")
    assert empty.detected == [] and empty.risk == 0.0


@pytest.mark.unit
def test_detects_phone() -> None:
    """Dashed, country-coded and parenthesised phone numbers are detected."""
    detector = PIIDetector()
    dashed = detector.scan("call me at 555-123-4567 tomorrow")
    assert [item["type"] for item in dashed.detected] == ["phone"]
    span = dashed.detected[0]["match_span"]
    assert "555-123-4567" in "call me at 555-123-4567 tomorrow"[span[0] : span[1]]

    assert detector.scan("ring +1 415 555 2671").detected[0]["type"] == "phone"
    assert detector.scan("dial (415) 555-2671").detected[0]["type"] == "phone"

    # redaction
    assert detector.redact("phone: 555-123-4567") == "phone: [PHONE]"

    # false-positive guards: ISO dates and short digit runs are not phones
    assert detector.scan("meeting moved from 2026-11-18 to 2026-11-25").detected == []
    assert detector.scan("version 1.2.3 released").detected == []


@pytest.mark.unit
def test_detects_api_key() -> None:
    """Prefixed tokens and key/value assignments are detected as api-key."""
    detector = PIIDetector()
    result = detector.scan("api_key: sk-abcdef1234567890abcdef123")
    assert [item["type"] for item in result.detected] == ["api-key"]
    assert result.risk == pytest.approx(0.5)
    sample = result.detected[0]["sample_redacted"]
    assert sample.startswith("sk") and sample.endswith("23")
    assert "abcdef1234567890" not in sample  # the secret itself is masked

    assert detector.scan("token: AKIAABCDEFGHIJKLMNOP").detected[0]["type"] == "api-key"
    assert (
        detector.scan("deploy key ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef123456").detected[0][
            "type"
        ]
        == "api-key"
    )

    # generic assignment with a long opaque value
    assert detector.scan("secret: Zx9-abcdef123456_+").detected[0]["type"] == "api-key"

    # weak/non-secret values are not flagged (false-positive guard)
    assert detector.scan("password: hunter2 is weak").detected == []
    assert detector.scan("the token count is 42").detected == []

    assert detector.redact("key sk-abcdef1234567890abcdef123 embedded") == (
        "key [API_KEY] embedded"
    )


@pytest.mark.unit
def test_detects_credit_card() -> None:
    """Luhn-valid card numbers are detected; invalid ones and SSNs sort out."""
    detector = PIIDetector()
    valid = detector.scan("card 4111 1111 1111 1111 on file")
    assert [item["type"] for item in valid.detected] == ["credit-card"]
    assert valid.risk == pytest.approx(0.6)
    assert detector.redact("card 4111 1111 1111 1111 on file") == "card [CREDIT_CARD] on file"

    # dashed form works too
    assert detector.scan("4111-1111-1111-1111").detected[0]["type"] == "credit-card"

    # Luhn failure (last digit off by one) is NOT a credit card
    assert detector.scan("card 4111 1111 1111 1112 on file").detected == []

    # SSNs are their own type, with structural validation
    ssn = detector.scan("ssn 078-05-1120 filed")
    assert [item["type"] for item in ssn.detected] == ["ssn"]
    assert detector.scan("ssn 900-05-1120 invalid area").detected == []  # 9xx area
    assert detector.scan("ssn 000-05-1120 invalid area").detected == []  # 000 area

    # a phone number is never mistaken for a card (length guard)
    assert detector.scan("call 555-123-4567").detected[0]["type"] == "phone"

    # configurable extra patterns merge into the scan
    custom = PIIDetector(
        extra_patterns=[PIIPattern(type="internal-host", pattern=r"\b[a-z]+-\d\d\.lan\b")]
    )
    found = custom.scan("server db-01.lan is down")
    assert [item["type"] for item in found.detected] == ["internal-host"]
    assert custom.redact("server db-01.lan is down") == "server [INTERNAL_HOST] is down"


@pytest.mark.unit
def test_redaction_preserves_non_pii_text() -> None:
    """Redaction touches ONLY the detected spans (P20.T6 boundary).

    Every byte of non-PII text around the redactions survives byte-exact:
    the redacted output is the input with exactly the detected spans replaced
    by their placeholders — asserted span-by-span and via full-string
    equality. Text without PII round-trips unchanged (identity).
    """
    detector = PIIDetector()
    pre = "ping alice at "
    mid1 = " (work), then call "
    mid2 = " or use key "
    post = " before 17:00."
    text = f"{pre}alice@example.com{mid1}555-123-4567{mid2}sk-abcdef1234567890abcdef123{post}"

    result = detector.scan(text)
    email_start = len(pre)
    email_end = email_start + len("alice@example.com")
    phone_start = email_end + len(mid1)
    phone_end = phone_start + len("555-123-4567")
    key_start = phone_end + len(mid2)
    key_end = len(text) - len(post)
    # the spans carve the text into exactly the expected secrets
    assert [
        (item["type"], text[item["match_span"][0] : item["match_span"][1]], item["match_span"])
        for item in result.detected
    ] == [
        ("email", "alice@example.com", (email_start, email_end)),
        ("phone", "555-123-4567", (phone_start, phone_end)),
        ("api-key", "sk-abcdef1234567890abcdef123", (key_start, key_end)),
    ]

    # redaction = span-for-placeholder substitution; every non-PII segment
    # (including punctuation, spacing and capitalisation) survives byte-exact
    redacted = detector.redact(text)
    expected = f"{pre}[EMAIL]{mid1}[PHONE]{mid2}[API_KEY]{post}"
    assert redacted == expected
    for segment in (pre, mid1, mid2, post):
        assert segment in redacted
        assert redacted.count(segment) == 1
    # the placeholders replaced exactly the secret bytes: lengths add up
    assert len(redacted) == (
        len(text)
        - len("alice@example.com")
        - len("555-123-4567")
        - len("sk-abcdef1234567890abcdef123")
        + len("[EMAIL]")
        + len("[PHONE]")
        + len("[API_KEY]")
    )
    # process() (the store pre-write hook) applies the same substitution
    stored, process_result = detector.process(text)
    assert stored == expected
    assert process_result.detected == result.detected

    # no PII at all: redaction is the identity function (byte-exact)
    clean = "nothing sensitive in this line at all"
    assert detector.redact(clean) == clean
    assert detector.scan(clean).detected == []

    # PII at the very start and very end: the adjacent segments are preserved
    edge = "jane@example.com or 555-123-4567"
    assert detector.redact(edge) == "[EMAIL] or [PHONE]"
    assert detector.redact("jane@example.com").startswith("[EMAIL]")
