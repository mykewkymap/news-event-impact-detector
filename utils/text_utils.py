"""
Utility functions for text preprocessing.

These helpers perform basic text cleaning to prepare headlines for
tokenization.  Cleaning is minimal because transformer models like
FinBERT are robust to casing and punctuation, but we remove extra
whitespace and optionally lower case the text.  Feel free to extend
this module with additional preprocessing steps (e.g. handling
special symbols or expanding contractions) as needed.
"""

from __future__ import annotations

import re
from typing import Iterable, List


def normalize_headline(text: str) -> str:
    """Normalize a single headline.

    Parameters
    ----------
    text : str
        Raw headline string.

    Returns
    -------
    str
        Normalized headline with collapsed whitespace and stripped
        leading/trailing spaces.
    """
    # Replace multiple whitespace characters with a single space
    normalized = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    normalized = normalized.strip()
    return normalized


def normalize_batch(texts: Iterable[str]) -> List[str]:
    """Normalize a batch of headlines.

    Parameters
    ----------
    texts : iterable of str
        Raw headline strings.

    Returns
    -------
    list of str
        Normalized headlines.
    """
    return [normalize_headline(t) for t in texts]