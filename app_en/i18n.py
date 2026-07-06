"""
app_en/i18n.py
==============

Shared display translations for the English app. The dataset stores category
values in Spanish (they are the source of truth); these helpers translate them
for DISPLAY only — the underlying values sent to the models stay untouched.
"""

from __future__ import annotations

import re

MODALITY_EN = {"100% remoto": "100% remote", "100% presencial": "100% on-site",
               "híbrido": "Hybrid"}
GENDER_EN = {"masculino": "male", "femenino": "female",
             "otro / no especifica": "other / N.A."}
BOOL_EN = {"True": "Yes", "False": "No"}


def size_en(v: str) -> str:
    """'De 201 a 500 personas' -> '201–500 people' (display only)."""
    s = str(v)
    if s.startswith("1 (solamente"):
        return "Just me (1)"
    m = re.match(r"De\s*(\d+)\s*a\s*(\d+)", s)
    if m:
        return f"{m.group(1)}–{m.group(2)} people"
    m = re.match(r"Más de\s*(\d+)", s)
    if m:
        return f"{int(m.group(1)):,}+ people"
    return s


def value_en(v: str) -> str:
    """Best-effort display translation for any category value."""
    s = str(v)
    return MODALITY_EN.get(s, GENDER_EN.get(s, BOOL_EN.get(s, size_en(s))))
