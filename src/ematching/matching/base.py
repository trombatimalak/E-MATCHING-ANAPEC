"""Forme commune à tous les moteurs de matching (règles, sémantique, ML).

Chaque moteur implémente score(profil, offre). Le classement (rank) est commun.
L'API et l'évaluation utilisent n'importe quel moteur de la même façon, sans
savoir lequel tourne : on peut en changer ou en ajouter sans modifier le reste.

Profil et offre sont des dictionnaires avec les colonnes du dataset
(profiles.csv / offers.csv). Compétences : libellés ESCO séparés par « | ».
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MatchResult:
    """Résultat explicable d'une paire profil-offre."""

    offer_id: str
    score: float  # 0 à 1 : aide à la décision, pas une décision automatique
    parts: dict[str, float] = field(default_factory=dict)  # détail par critère
    common_skills: list[str] = field(default_factory=list)  # requises et possédées
    missing_skills: list[str] = field(default_factory=list)  # requises, absentes


class MatchingEngine(ABC):
    """Moteur de matching : note une paire, classe des offres pour un profil."""

    name = "abstrait"

    @abstractmethod
    def score(self, profile: dict, offer: dict) -> MatchResult:
        """Note une paire profil-offre."""

    def rank(self, profile: dict, offers: list[dict], k: int = 5) -> list[MatchResult]:
        """Les k meilleures offres pour un profil (par défaut : le Top 5)."""
        results = [self.score(profile, offer) for offer in offers]
        return sorted(results, key=lambda result: -result.score)[:k]
