"""Chargement des données (ESCO, profils, offres)."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ESCO_DIR = RAW_DIR / "esco"


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Charge un fichier CSV."""
    return pd.read_csv(path, **kwargs)


def load_esco(name: str, lang: str = "fr", **kwargs) -> pd.DataFrame:
    """Charge une table ESCO, par exemple 'skills' ou 'occupations'."""
    return load_csv(ESCO_DIR / f"{name}_{lang}.csv", **kwargs)
