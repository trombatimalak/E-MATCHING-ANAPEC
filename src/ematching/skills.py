"""Normalisation et extraction des compétences à partir d'ESCO."""

import re
import unicodedata

import pandas as pd


def normalize_text(text: str) -> str:
    """Minuscules, sans accents ni ponctuation superflue."""
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    # Un point n'est conservé qu'entre deux caractères alphanumériques (node.js).
    text = re.sub(r"\.(?![a-z0-9])|(?<![a-z0-9])\.", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_lexicon(skills: pd.DataFrame) -> dict[str, str]:
    """Associe chaque libellé normalisé (préféré ou alternatif) à l'URI ESCO."""
    lexicon: dict[str, str] = {}
    rows = zip(skills["conceptUri"], skills["preferredLabel"], skills["altLabels"])
    for uri, preferred, alternatives in rows:
        labels = [preferred]
        if isinstance(alternatives, str):
            labels += alternatives.split("\n")
        for label in labels:
            key = normalize_text(label)
            if key:
                lexicon.setdefault(key, uri)
    return lexicon


def extract_skills(text: str, lexicon: dict[str, str]) -> set[str]:
    """Retourne les URI des compétences du lexique trouvées dans le texte."""
    normalized = f" {normalize_text(text)} "
    return {uri for label, uri in lexicon.items() if f" {label} " in normalized}
