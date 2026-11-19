"""Moteur de matching hybride : critères lisibles (règles) + sens (embeddings).

Pour chaque compétence demandée par l'offre, on cherche la compétence la plus
proche du profil :
- identique, ou équivalente par le sens (similarité >= skill_threshold) :
  compétence commune, affichée « demandée (≈ possédée) » si les mots diffèrent ;
- sinon : compétence manquante.

Score = moyenne pondérée des critères (pondérations dans config/matching.toml) :
  skills       part des compétences requises couvertes (identiques ou équivalentes)
  semantic     proximité globale des compétences du profil et de l'offre
  experience, education, location, nice_to_have : voir features.py
Chaque critère est renvoyé dans le résultat : le score est explicable.

Sans encodeur, le moteur ne compare que des libellés identiques et ignore le
critère « semantic » (sert de référence dans l'évaluation).
"""

import tomllib
from pathlib import Path

import numpy as np

from ematching.data import PROJECT_ROOT
from ematching.features import pair_features, split_skills
from ematching.matching.base import MatchingEngine, MatchResult

CONFIG = PROJECT_ROOT / "config" / "matching.toml"


def load_weights(path: str | Path = CONFIG) -> dict[str, float]:
    """Pondérations du fichier de configuration, normalisées (somme = 1)."""
    with open(path, "rb") as handle:
        weights = tomllib.load(handle)["weights"]
    total = sum(weights.values())
    if total <= 0 or any(value < 0 for value in weights.values()):
        raise ValueError("Les pondérations doivent être positives")
    return {name: value / total for name, value in weights.items()}


class SemanticEngine(MatchingEngine):
    """Moteur hybride ; encoder = objet de ematching.embeddings (ou None)."""

    name = "sémantique"

    def __init__(
        self, encoder=None, skill_threshold: float | None = None, weights=None
    ):
        if encoder is not None and skill_threshold is None:
            # Seuil connu pour les modèles du projet (voir embeddings.MODEL_SPECS).
            skill_threshold = getattr(encoder, "spec", {}).get("skill_threshold")
            if skill_threshold is None:
                raise ValueError("skill_threshold requis pour ce modèle")
        self.encoder = encoder
        self.skill_threshold = skill_threshold
        self.weights = weights or load_weights()
        # Vecteurs mis en cache : chaque libellé n'est encodé qu'une fois.
        # Compétences du profil = « requêtes », de l'offre = « documents », comme
        # lors du réglage du seuil sur les synonymes ESCO.
        self._query_cache: dict[str, np.ndarray] = {}
        self._doc_cache: dict[str, np.ndarray] = {}

    def warm_up(self, offers: list[dict]) -> None:
        """Encode à l'avance les compétences de toutes les offres (une seule fois),
        pour que chaque CV soit ensuite classé rapidement."""
        if self.encoder is not None:
            skills = [s for o in offers for s in split_skills(o.get("required_skills"))]
            if skills:
                self._vectors(skills, self._doc_cache, "documents")

    def score(self, profile: dict, offer: dict) -> MatchResult:
        owned = split_skills(profile.get("skills"))
        wanted = split_skills(offer.get("required_skills"))
        common, missing, semantic = self._compare_skills(owned, wanted)

        parts = pair_features(profile, offer)
        parts["skills"] = len(common) / len(wanted) if wanted else 1.0
        del parts["skills_exact"]  # remplacé par « skills » (identiques + équivalentes)
        if semantic is not None:
            parts["semantic"] = semantic

        # Moyenne pondérée sur les critères disponibles (poids renormalisés).
        used = {name: w for name, w in self.weights.items() if name in parts}
        total = sum(used.values())
        score = sum(w * parts[name] for name, w in used.items()) / total if total else 0
        return MatchResult(
            offer_id=offer.get("offer_id", ""),
            score=round(score, 4),
            parts={name: round(value, 3) for name, value in parts.items()},
            common_skills=common,
            missing_skills=missing,
        )

    def _compare_skills(self, owned: list[str], wanted: list[str]):
        """(compétences communes, manquantes, proximité globale ou None)."""
        exact = set(owned)
        if self.encoder is None or not owned or not wanted:
            common = [skill for skill in wanted if skill in exact]
            missing = [skill for skill in wanted if skill not in exact]
            return common, missing, None

        owned_vectors = self._vectors(owned, self._query_cache, "queries")
        wanted_vectors = self._vectors(wanted, self._doc_cache, "documents")
        similarities = wanted_vectors @ owned_vectors.T  # demandées x possédées
        common, missing = [], []
        for skill, row in zip(wanted, similarities):
            best = int(np.argmax(row))
            if skill in exact:
                common.append(skill)
            elif row[best] >= self.skill_threshold:
                common.append(f"{skill} (≈ {owned[best]})")
            else:
                missing.append(skill)
        # Proximité globale : cosinus entre les « centres » des deux ensembles.
        a, b = owned_vectors.mean(axis=0), wanted_vectors.mean(axis=0)
        semantic = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        return common, missing, max(0.0, semantic)

    def _vectors(self, labels: list[str], cache: dict, kind: str) -> np.ndarray:
        missing = [label for label in dict.fromkeys(labels) if label not in cache]
        if missing:
            encode = getattr(self.encoder, f"encode_{kind}")
            for label, vector in zip(missing, encode(missing)):
                cache[label] = vector
        return np.array([cache[label] for label in labels])
