import itertools

import pytest

from factshorts.core.captions import caption_chunks, proportional_word_timings
from factshorts.render.layout import SafeZone, fit_text
from factshorts.render.themes import load_theme


def test_word_timings_cover_sentence_exactly() -> None:
    words = proportional_word_timings("Mount Everest is 8,849 meters tall.", 2.0, 3.0)
    assert [w.text for w in words] == ["Mount", "Everest", "is", "8,849", "meters", "tall."]
    assert words[0].start == pytest.approx(2.0)
    assert words[-1].end == pytest.approx(5.0)
    for a, b in itertools.pairwise(words):
        assert a.end == pytest.approx(b.start)
        assert a.end > a.start


def test_numbers_get_time_for_their_spoken_length() -> None:
    words = proportional_word_timings("It is 8,849 m.", 0.0, 2.0)
    dur = {w.text: w.end - w.start for w in words}
    assert dur["8,849"] > 3 * dur["is"]


def test_caption_chunks_are_short_and_break_at_sentence_end() -> None:
    words = proportional_word_timings("One two three four five six. Seven eight.", 0.0, 4.0)
    chunks = caption_chunks(words, max_words=4)
    assert [len(c) for c in chunks] == [4, 2, 2]
    assert all(len(" ".join(w.text for w in c)) <= 22 or len(c) == 1 for c in chunks)


def test_safe_zone_avoids_bottom_and_right_edge() -> None:
    z = SafeZone()
    assert z.bottom == 1536
    assert z.right <= 1080 - 150
    assert z.contains((z.left, z.top, z.right, z.bottom))
    assert not z.contains((z.left, 1500, z.right, 1700))
    assert not z.contains((900, 500, 1060, 600))


def test_long_text_fits_inside_safe_zone() -> None:
    theme = load_theme("default")
    z = SafeZone()
    text = "THE HIGHEST MOUNTAIN ON EARTH ABOVE SEA LEVEL IS MOUNT EVEREST IN THE HIMALAYAS"
    font, lines, line_h = fit_text(text, theme.display_font, z.inner_width, 560, 190, 40)
    assert all(font.getlength(line) <= z.inner_width for line in lines)
    assert line_h * len(lines) <= 560
