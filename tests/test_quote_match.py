from factshorts.verify.programmatic import quote_check

SOURCE = (
    "Mount Testa is the highest mountain in Fictland, with a summit elevation of 6,412 m "
    "(21,037 ft) above sea level. It lies on the border between Northia and Southia.\n"
    "Temperatures at the summit can fall to −45 °C in winter. The town’s "
    "“old bridge” – built in 1802 – still stands."
)


def test_exact_quote_passes_and_returns_paragraph() -> None:
    m = quote_check("It lies on the border between Northia and Southia.", SOURCE)
    assert m.result.status == "pass"
    assert m.paragraph is not None and m.paragraph.startswith("Mount Testa")


def test_curly_quotes_dashes_and_whitespace_are_normalized() -> None:
    quote = 'The town\'s "old bridge" - built in 1802 -   still stands.'
    assert quote_check(quote, SOURCE).result.status == "pass"
    assert (
        quote_check(
            "Temperatures at the summit can fall to -45 °C in winter.", SOURCE
        ).result.status
        == "pass"
    )


def test_prompt_labels_are_stripped() -> None:
    assert (
        quote_check("[2] It lies on the border between Northia and Southia.", SOURCE).result.status
        == "pass"
    )


def test_punctuation_only_difference_passes() -> None:
    m = quote_check("It lies on the border between Northia and Southia", SOURCE)
    assert m.result.status == "pass"
    m = quote_check("The town's old bridge built in 1802 still stands", SOURCE)
    assert m.result.status == "pass"


def test_changed_word_fails() -> None:
    m = quote_check("It lies on the border between Northia and Eastia.", SOURCE)
    assert m.result.status == "fail"
    assert m.paragraph is None


def test_changed_number_fails_even_if_close() -> None:
    assert (
        quote_check(
            "with a summit elevation of 6,413 m (21,037 ft) above sea level", SOURCE
        ).result.status
        == "fail"
    )
    assert (
        quote_check(
            "with a summit elevation of 6.412 m (21,037 ft) above sea level", SOURCE
        ).result.status
        == "fail"
    )


def test_too_short_quote_fails() -> None:
    assert quote_check("Northia", SOURCE).result.status == "fail"
