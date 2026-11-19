"""Bout-en-bout : un CV (PDF ou Word) -> les 5 meilleures offres, expliquées.

Étapes :
  1. lire le CV (cv_parser) : texte, sections, études, expérience, ville, langues ;
  2. extraire les compétences ESCO (skills.SkillExtractor, méthode lexicale) ;
  3. construire le profil au format du dataset (profiles.csv) ;
  4. classer les offres (matching.semantic.SemanticEngine) et garder le Top k.

Utilisation (par exemple depuis l'API) :
    recommender = Recommender()                 # à créer une seule fois
    result = recommender.recommend("cv.pdf")    # puis pour chaque CV
    result = recommender.recommend_profile(profil_corrigé)  # après correction

Le modèle d'embeddings et le fichier de pondérations sont des paramètres :
l'équipe peut changer de configuration sans modifier le code. Par défaut, les
pondérations sont celles apprises pour le modèle choisi
(config/matching_learned_<modèle>.toml, créées par scripts/tune_weights.py) ;
sans modèle, ou si ce fichier n'existe pas : config/matching.toml (a priori).
Le score est une aide à la recommandation, pas une décision automatique.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from ematching.cv_parser import ParsedCV, parse_cv, parse_text
from ematching.data import PROCESSED_DIR, PROJECT_ROOT
from ematching.embeddings import DEFAULT_MODEL
from ematching.features import FORBIDDEN, split_skills
from ematching.matching.base import MatchResult
from ematching.matching.semantic import CONFIG, SemanticEngine, load_weights
from ematching.skills import SkillExtractor, SkillMatch

REFERENTIAL = PROCESSED_DIR / "skills_referential.csv"
SYNONYMS = PROJECT_ROOT / "data" / "lexicon" / "skills_synonyms.csv"
SEP = "|"


@dataclass
class Recommendation:
    """Résultat complet pour un CV, prêt à être affiché ou renvoyé par l'API."""

    profile: dict  # profil construit (format profiles.csv), corrigeable
    skills: list[SkillMatch] = field(
        default_factory=list
    )  # comment chacune a été trouvée
    results: list[MatchResult] = field(default_factory=list)  # Top k, meilleure d'abord
    offers: list[dict] = field(default_factory=list)  # détail des offres du Top k
    warnings: list[str] = field(default_factory=list)  # informations à vérifier


def default_weights_path(model: str | None) -> Path:
    """Pondérations apprises pour ce modèle si elles existent, sinon a priori."""
    if model and model.lower() != "none":
        learned = PROJECT_ROOT / "config" / f"matching_learned_{model}.toml"
        if learned.exists():
            return learned
    return CONFIG


class Recommender:
    """Assemble lecture du CV, extraction et matching (à créer une seule fois)."""

    def __init__(
        self,
        offers: list[dict] | None = None,
        model: str | None = DEFAULT_MODEL,
        weights_path: str | Path | None = None,
        encoder=None,
        extractor: SkillExtractor | None = None,
    ):
        """offers : liste d'offres (par défaut data/processed/offers.csv).
        model : nom d'un modèle d'embeddings, ou None / « none » pour s'en passer.
        weights_path : pondérations (par défaut : default_weights_path).
        encoder, extractor : à fournir pour les tests (sinon créés ici)."""
        if offers is None:
            table = pd.read_csv(PROCESSED_DIR / "offers.csv")
            offers = table.drop(columns=list(FORBIDDEN), errors="ignore").to_dict(
                "records"
            )
        # Par sécurité, les colonnes qui révèlent le métier source sont retirées.
        self.offers = [
            {k: v for k, v in o.items() if k not in FORBIDDEN} for o in offers
        ]
        self._offers_by_id = {o["offer_id"]: o for o in self.offers}

        self.extractor = extractor or SkillExtractor.from_files(REFERENTIAL, SYNONYMS)
        if encoder is None and model and model.lower() != "none":
            from ematching.embeddings import SentenceEncoder

            encoder = SentenceEncoder(model)
        self.weights_path = weights_path or default_weights_path(
            model if encoder is not None else None
        )
        self.engine = SemanticEngine(encoder, weights=load_weights(self.weights_path))
        self.engine.warm_up(self.offers)  # vecteurs des offres calculés une fois

    def recommend(
        self, cv_path: str | Path, k: int = 5, today: date | None = None
    ) -> Recommendation:
        """Top k des offres pour un fichier CV (PDF ou DOCX)."""
        return self._from_parsed(parse_cv(cv_path, today), k)

    def recommend_text(
        self, text: str, k: int = 5, today: date | None = None
    ) -> Recommendation:
        """Top k pour un texte de CV déjà extrait (ex. formulaire en ligne)."""
        return self._from_parsed(parse_text(text, today), k)

    def recommend_profile(self, profile: dict, k: int = 5) -> Recommendation:
        """Top k pour un profil saisi ou corrigé par le candidat ou le conseiller
        (mêmes clés que profiles.csv : skills, years_experience, education_level,
        city…)."""
        return self._rank(dict(profile), skills=[], warnings=self._check(profile), k=k)

    def _from_parsed(self, parsed: ParsedCV, k: int) -> Recommendation:
        matches = [
            m for m in self.extractor.extract(parsed.skill_text()) if m.status == "ok"
        ]
        profile = parsed.to_profile()
        profile["profile_id"] = "CV"
        profile["skills"] = SEP.join(dict.fromkeys(m.label for m in matches))
        return self._rank(profile, matches, self._check(profile), k)

    def _rank(self, profile, skills, warnings, k) -> Recommendation:
        results = self.engine.rank(profile, self.offers, k)
        return Recommendation(
            profile=profile,
            skills=skills,
            results=results,
            offers=[self._offers_by_id[r.offer_id] for r in results],
            warnings=warnings,
        )

    @staticmethod
    def _check(profile: dict) -> list[str]:
        """Informations manquantes, à faire vérifier ou compléter par l'utilisateur."""
        warnings = []
        if not split_skills(profile.get("skills")):
            warnings.append("Aucune compétence reconnue : compléter la liste.")
        for key, message in [
            ("years_experience", "Expérience non trouvée"),
            ("education_level", "Niveau d'études non trouvé"),
            ("city", "Ville non trouvée"),
        ]:
            if profile.get(key) is None:
                warnings.append(f"{message} : à compléter.")
        return warnings
