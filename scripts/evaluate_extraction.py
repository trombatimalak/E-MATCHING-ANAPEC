"""Mesure l'extraction des compétences sur les CV fictifs (data/samples/cv/).

Compare les compétences trouvées dans chaque CV au corrigé expected.csv :
- précision : part des compétences trouvées qui sont justes ;
- rappel    : part des compétences attendues qui ont été trouvées ;
- F1        : moyenne harmonique des deux.

Méthodes comparées :
- base     : extracteur d'origine (libellés ESCO exacts, tout le texte du CV) ;
- lexique  : SkillExtractor sans modèle (exact + synonymes + formes + fautes) ;
- hybride  : lexique + embeddings (option --model).

Réglage du seuil sémantique, SANS utiliser les CV (règles fixées à l'avance) :
- on prend les ~980 libellés alternatifs ESCO (« logiciels de gestion de
  versions ») ; pour chacun, le modèle propose la compétence la plus proche ;
- seuil = le plus petit score pour lequel au moins 80 % des propositions
  acceptées sont justes (précision proche de celle du lexique, 84 %) ;
- contrôle : part des lignes sans compétence (en-tête, contact, langues,
  loisirs des CV) qui dépasseraient ce seuil (fausses alertes).
Pourquoi pas les CV de développement : le lexique a été construit en les
regardant (rappel 92 %), il n'y reste presque rien à découvrir ; un réglage sur
eux désactive toujours la sémantique. Les CV de contrôle, eux, ne servent jamais
au réglage : leur chiffre reste le chiffre honnête.
Mode retenu (fixé à l'avance) : « unmatched » (lignes où le lexique n'a rien
trouvé) ; le mode « all » est affiché pour information.

Seules les compétences au statut « ok » comptent ; les « à revoir » sont comptées
à part. La comparaison se fait sur les libellés ESCO (le référentiel contient
quelques libellés en double, comme « SQL », avec deux URI différentes).

Usage :
  python scripts/evaluate_extraction.py [--details]
  python scripts/evaluate_extraction.py --model e5-small [--details]
"""

import argparse
import csv

import numpy as np
import pandas as pd

from ematching.cv_parser import parse_cv
from ematching.data import PROCESSED_DIR, PROJECT_ROOT
from ematching.skills import (
    SEMANTIC_MIN_WORDS,
    SkillExtractor,
    build_lexicon,
    choose_threshold,
    extract_skills,
    normalize_text,
)

CV_DIR = PROJECT_ROOT / "data" / "samples" / "cv"
REFERENTIAL = PROCESSED_DIR / "skills_referential.csv"
SYNONYMS = PROJECT_ROOT / "data" / "lexicon" / "skills_synonyms.csv"
SEP = "|"
TARGET_PRECISION = 0.8
# Sections sans compétences : servent à compter les fausses alertes.
NON_SKILL_SECTIONS = ("header", "contact", "languages", "interests", "qualities")


def scores(counts) -> tuple[float, float, float]:
    """Précision, rappel et F1 à partir de [trouvées, attendues, justes]."""
    n_found, n_expected, n_correct = counts
    precision = n_correct / n_found if n_found else 0.0
    recall = n_correct / n_expected if n_expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if n_correct else 0.0
    return precision, recall, f1


def load_cvs() -> list[dict]:
    """CV lisibles du corrigé, lus une seule fois."""
    with open(CV_DIR / "expected.csv", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["expected_skills"]]
    for row in rows:
        row["parsed"] = parse_cv(CV_DIR / row["cv_file"])
        row["expected"] = set(row["expected_skills"].split(SEP))
    return rows


def evaluate(method, rows) -> dict[str, list[int]]:
    """Totaux [trouvées, attendues, justes] par jeu (developpement / controle)."""
    totals: dict[str, list[int]] = {}
    for row in rows:
        found = method(row["parsed"])
        counts = [len(found), len(row["expected"]), len(found & row["expected"])]
        total = totals.setdefault(row["set"], [0, 0, 0])
        for i, value in enumerate(counts):
            total[i] += value
    return totals


def show(name: str, totals: dict, extra: str = "") -> None:
    for subset in ("developpement", "controle"):
        if subset in totals:
            p, r, f1 = scores(totals[subset])
            print(f"{name:28} {subset:14} {p:9.0%} {r:7.0%} {f1:6.2f}  {extra}")


def show_details(method, rows) -> None:
    for row in rows:
        found = method(row["parsed"])
        print(f"\n{row['cv_file']} ({row['set']})")
        print(f"    manquées        : {sorted(row['expected'] - found)}")
        print(f"    trouvées à tort : {sorted(found - row['expected'])}")


def esco_synonyms(referential: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(libellé alternatif, libellé préféré attendu), sans les libellés
    alternatifs identiques à un libellé préféré (trop faciles)."""
    known = {normalize_text(label) for label in referential["preferredLabel"]}
    queries, targets = [], []
    for preferred, alternatives in zip(
        referential["preferredLabel"], referential["altLabels"]
    ):
        if isinstance(alternatives, str):
            for alt in alternatives.split("\n"):
                if alt.strip() and normalize_text(alt) not in known:
                    queries.append(alt.strip())
                    targets.append(preferred)
    return queries, targets


def non_skill_lines(rows) -> list[str]:
    """Lignes des sections sans compétences (au moins 2 mots, comme la méthode
    sémantique)."""
    return [
        line.strip()
        for row in rows
        for name in NON_SKILL_SECTIONS
        for line in row["parsed"].sections.get(name, "").splitlines()
        if len(normalize_text(line).split()) >= SEMANTIC_MIN_WORDS
    ]


def hybrid(model_name: str, rows, referential, accepted):
    """Règle le seuil sur les synonymes ESCO, puis mesure sur les CV."""
    from ematching.embeddings import SentenceEncoder

    extractor = SkillExtractor.from_files(
        REFERENTIAL, SYNONYMS, encoder=SentenceEncoder(model_name), accept_threshold=1
    )
    queries, targets = esco_synonyms(referential)
    similarities = extractor.similarities(queries)
    top_scores = similarities.max(axis=1)
    proposed = [extractor.semantic_labels[i] for i in similarities.argmax(axis=1)]
    correct = [p == t for p, t in zip(proposed, targets)]
    threshold = choose_threshold(top_scores, correct, TARGET_PRECISION)
    if threshold is None:
        print(
            f"\n{model_name} : aucun seuil n'atteint {TARGET_PRECISION:.0%} de précision"
        )
        return accepted(extractor)
    negatives = extractor.similarities(non_skill_lines(rows)).max(axis=1)
    print(
        f"\nRéglage sur {len(queries)} synonymes ESCO : seuil={threshold:.2f}, "
        f"{np.mean(top_scores >= threshold):.0%} des synonymes acceptés (dont "
        f"{TARGET_PRECISION:.0%}+ justes) ; fausses alertes sur {len(negatives)} "
        f"lignes sans compétence : {np.mean(negatives >= threshold):.0%}\n"
    )
    extractor.accept_threshold = threshold
    for scope in ("all", "unmatched"):  # « unmatched » en dernier : mode retenu
        extractor.semantic_scope = scope
        show(f"hybride {model_name} ({scope})", evaluate(accepted(extractor), rows))

    # Ce que la sémantique ajoute sur les CV de contrôle (✓ = dans le corrigé).
    print("\nAjouts sémantiques sur les CV de contrôle (mode unmatched) :")
    for row in rows:
        if row["set"] != "controle":
            continue
        for m in extractor.extract(row["parsed"].skill_text()):
            if m.method == "sémantique":
                mark = "✓" if m.label in row["expected"] else "✗"
                print(f"  {mark} [{m.status}] {m.label}  <-  {m.source[:60]}")
    return accepted(extractor)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mesure de l'extraction")
    parser.add_argument("--model", help="modèle d'embeddings (ex. e5-small)")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    rows = load_cvs()

    referential = pd.read_csv(REFERENTIAL)
    label_of = dict(zip(referential["skillUri"], referential["preferredLabel"]))
    lexicon = build_lexicon(referential.rename(columns={"skillUri": "conceptUri"}))
    lexical = SkillExtractor.from_files(REFERENTIAL, SYNONYMS)

    def base(parsed) -> set:
        return {label_of[uri] for uri in extract_skills(parsed.text, lexicon)}

    def accepted(extractor):
        def method(parsed) -> set:
            matches = extractor.extract(parsed.skill_text())
            return {m.label for m in matches if m.status == "ok"}

        return method

    print(f"{'méthode':28} {'jeu':14} {'précision':>9} {'rappel':>7} {'F1':>6}")
    show("base", evaluate(base, rows))
    show("lexique", evaluate(accepted(lexical), rows))
    best_method = accepted(lexical)

    if args.model:
        best_method = hybrid(args.model, rows, referential, accepted)

    if args.details:
        print("\nDétail de la meilleure méthode :")
        show_details(best_method, rows)


if __name__ == "__main__":
    main()
