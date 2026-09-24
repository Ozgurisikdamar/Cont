from contextlens.preprocessing.text import MAX_INPUT_CHARS, content_words, is_informative, normalize


def test_nfkc_and_casefold():
    assert normalize("ﬁnance", lowercase=True) == "finance"  # ligature folded
    assert normalize("ＱＵＡＮＴＵＭ", lowercase=True) == "quantum"  # full-width
    assert normalize("STRASSE Straße", lowercase=True) == "strasse strasse"  # casefold, not lower


def test_dotted_capital_i_is_handled_without_locale():
    # The Turkish I/ı issue from the original brief: casefold is locale-independent,
    # so English text containing these characters still normalises deterministically.
    assert normalize("İstanbul ISTANBUL", lowercase=True).startswith("i")
    assert "ı" not in normalize("INFO", lowercase=True)


def test_removes_urls_mentions_html_and_emoji_but_keeps_hashtag_words():
    raw = "Read <b>this</b> https://example.com/x?y=1 by @alice #quantum 🚀 now &amp; later"
    assert normalize(raw) == "Read this by quantum now & later"


def test_case_preserved_by_default():
    assert normalize("Quantum  Mechanics\n") == "Quantum Mechanics"


def test_empty_and_whitespace():
    assert normalize("") == ""
    assert normalize("   \t\n ") == ""
    assert not is_informative("")
    assert not is_informative("   ")


def test_uninformative_inputs():
    for text in ["and this is a", "!!!", "12345", "... ?? --", "🚀🚀🚀", "the of and"]:
        assert not is_informative(text), text


def test_informative_inputs():
    assert is_informative("Hello")
    assert is_informative("Qubits enable quantum algorithms")
    assert content_words("The qubits and the processors") == ["qubits", "processors"]


def test_long_input_is_truncated():
    assert len(normalize("word " * 5000)) <= MAX_INPUT_CHARS
