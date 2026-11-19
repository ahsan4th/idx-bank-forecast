import pytest

from forecaster.branding import css_string, watermark_css


def test_css_string_escapes_everything_that_could_break_out():
    quoted = css_string('a "b" \\ </style><script>x</script>\nbaris 2')
    assert quoted.startswith('"') and quoted.endswith('"')
    inner = quoted[1:-1]
    assert "<" not in inner  # so the text can never close the <style> element
    assert '\\"b\\"' in inner
    assert "\\\\" in inner
    assert "\\A " in inner  # newline becomes a CSS line break


def test_watermark_css_targets_the_keyed_container():
    css = watermark_css("Muhammad Ahsan · ITS", "header")
    assert css.startswith("<style>") and css.endswith("</style>")
    assert css.count("</style>") == 1
    assert ".st-key-header::after" in css
    assert 'content: "Muhammad Ahsan · ITS";' in css
    assert "position: fixed" in css
    assert "pointer-events: none" in css


def test_hostile_text_stays_inside_one_style_element():
    css = watermark_css("x</style><img src=y onerror=alert(1)>", "header")
    assert css.count("</style>") == 1
    assert "<img" not in css


@pytest.mark.parametrize(("text", "key"), [("", "header"), ("x", "bad key"), ("x", "1abc"), ("x", "")])
def test_watermark_css_rejects_empty_text_and_unusable_keys(text, key):
    with pytest.raises(ValueError):
        watermark_css(text, key)
