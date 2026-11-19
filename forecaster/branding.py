"""Tanda air aplikasi: teks kecil semi-transparan di pojok kanan bawah halaman.

CSS-nya menempel pada pseudo-elemen container yang diberi `key`, sesuai anjuran Streamlit
untuk menargetkan kelas `.st-key-<key>` alih-alih nama kelas internal atau `data-testid`.
`st.html` yang hanya berisi tag <style> masuk ke event container, jadi tidak memakan ruang.
"""

from __future__ import annotations

import re

_CSS_CLASS_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def css_string(text: str) -> str:
    """`text` as a double-quoted CSS string that cannot close the <style> element."""
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("<", "\\3C ")
        .replace("\n", "\\A ")
    )
    return f'"{escaped}"'


def watermark_css(text: str, container_key: str) -> str:
    """Style block that pins `text` to the page's lower right, above content, unclickable."""
    if not text:
        raise ValueError("Watermark text is empty; skip the watermark instead of styling nothing")
    if not _CSS_CLASS_KEY.fullmatch(container_key):
        raise ValueError(f"Container key {container_key!r} is not usable as a CSS class name")
    return (
        "<style>"
        f".st-key-{container_key}::after {{"
        f"content: {css_string(text)};"
        "position: fixed; right: 1.25rem; bottom: 0.75rem; z-index: 999990;"
        "pointer-events: none; user-select: none; white-space: pre;"
        "font-size: 0.8rem; font-weight: 600; letter-spacing: 0.02em;"
        "color: inherit; opacity: 0.45;"
        "}"
        "</style>"
    )
