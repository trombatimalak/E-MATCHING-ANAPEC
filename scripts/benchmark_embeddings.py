"""Compare des modèles d'embeddings pré-entraînés pour choisir celui du projet.

Trois tests, sur nos propres données :
  1. Synonymes ESCO : à partir d'un libellé alternatif (« logiciels de gestion de
     versions »), retrouver la bonne compétence parmi toutes celles du référentiel.
     Mesures : Recall@1, Recall@5 (bonne réponse en 1re / dans les 5 premières).
  2. CV fictifs : pour chaque CV, classer toutes les compétences selon leur
     ressemblance avec les lignes du CV ; on regarde combien de compétences
     attendues (expected.csv) arrivent en tête (R-précision). Résultats séparés
     pour les CV de développement et les CV de contrôle (le chiffre honnête).
  3. Aperçu du matching profil -> offres (dataset synthétique, split test) :
     Precision@5 et NDCG@5. On compare uniquement les compétences, sans
     l'intitulé du poste : il est dérivé du métier qui sert à calculer le label,
     il donnerait la réponse. Test facile : il vérifie que tout fonctionne mais
     départage peu les modèles.

Règle de décision (fixée avant les résultats) : critère principal = test 2 sur les
CV de contrôle, complété par le test 1 ; on retient le plus petit modèle à moins
de 3 points du meilleur.

Installation : pip install -e ".[dev,nlp]"
Usage :
  python scripts/benchmark_embeddings.py                 # les 4 modèles par défaut
  python scripts/benchmark_embeddings.py --models e5-small minilm
  python scripts/benchmark_embeddings.py --models gemma  # option (compte HF requis)
Le premier lancement télécharge les modèles (environ 3 Go), ensuite ils sont en cache.
"""

import argparse
import csv
import math
import time

import numpy as np
import pandas as pd

from ematching.cv_parser import parse_cv
from ematching.data import PROCESSED_DIR, PROJECT_ROOT
from ematching.embeddings import MODEL_SPECS, SentenceEncoder
from ematching.skills import normalize_text

CV_DIR = PROJECT_ROOT / "data" / "samples" / "cv"
SEP = "|"
K = 5

# Les modèles et leurs préfixes sont définis dans ematching/embeddings.py.
DEFAULT_MODELS = ["minilm", "e5-small", "e5-base", "qwen3-0.6b"]


# ============================================================================
# Mesures
# ============================================================================


def ndcg_at_k(labels_in_rank_order: list[int], all_labels: list[int], k: int = K):
    """NDCG@k avec des labels gradués (0, 1, 2)."""

    def dcg(labels):
        return sum((2**label - 1) / math.log2(i + 2) for i, label in enumerate(labels))

    ideal = dcg(sorted(all_labels, reverse=True)[:k])
    return dcg(labels_in_rank_order[:k]) / ideal if ideal else 0.0


def test_esco_synonyms(encoder, referential: pd.DataFrame) -> dict:
    """Test 1 : libellé alternatif -> libellé préféré."""
    labels = sorted(set(referential["preferredLabel"]))
    index = {label: i for i, label in enumerate(labels)}
    known = {normalize_text(label) for label in labels}
    queries, targets = [], []
    for preferred, alternatives in zip(
        referential["preferredLabel"], referential["altLabels"]
    ):
        if not isinstance(alternatives, str):
            continue
        for alt in alternatives.split("\n"):
            # On ignore les libellés alternatifs identiques à un libellé préféré.
            if alt.strip() and normalize_text(alt) not in known:
                queries.append(alt.strip())
                targets.append(index[preferred])
    start = time.perf_counter()
    docs = encoder.encode_documents(labels)
    seconds = time.perf_counter() - start
    similarities = encoder.encode_queries(queries) @ docs.T
    target_scores = similarities[np.arange(len(targets)), targets]
    ranks = (similarities > target_scores[:, None]).sum(axis=1)  # 0 = 1re place
    return {
        "R@1": float(np.mean(ranks < 1)),
        "R@5": float(np.mean(ranks < 5)),
        "n_synonymes": len(queries),
        "sec_1009_competences": seconds,
        "docs": docs,
        "labels": labels,
    }


def test_cvs(encoder, docs: np.ndarray, labels: list[str]) -> dict:
    """Test 2 : R-précision par CV (compétences attendues dans le top k, k = nombre
    de compétences attendues), cumulée par jeu de CV."""
    index = {label: i for i, label in enumerate(labels)}
    with open(CV_DIR / "expected.csv", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["expected_skills"]]
    hits: dict[str, list[int]] = {}
    for row in rows:
        lines = [
            line.strip()
            for line in parse_cv(CV_DIR / row["cv_file"]).skill_text().splitlines()
            if len(normalize_text(line)) >= 3
        ]
        scores = (encoder.encode_queries(lines) @ docs.T).max(axis=0)
        expected = {index[s] for s in row["expected_skills"].split(SEP)}
        top = set(np.argsort(-scores)[: len(expected)])
        found, total = hits.setdefault(row["set"], [0, 0])
        hits[row["set"]] = [found + len(top & expected), total + len(expected)]
    return {name: found / total for name, (found, total) in hits.items()}


def test_matching(encoder) -> dict:
    """Test 3 : profils test -> offres test, classées par similarité des
    compétences ; labels provisoires (silver) de test_grid.csv."""
    profiles = pd.read_csv(PROCESSED_DIR / "profiles.csv").set_index("profile_id")
    offers = pd.read_csv(PROCESSED_DIR / "offers.csv").set_index("offer_id")
    grid = pd.read_csv(PROCESSED_DIR / "test_grid.csv")
    profile_ids, offer_ids = grid["profile_id"].unique(), grid["offer_id"].unique()

    def skills_text(*values) -> str:
        return ", ".join(s for v in values if isinstance(v, str) for s in v.split(SEP))

    p_vec = encoder.encode_queries(
        [skills_text(profiles.at[p, "skills"]) for p in profile_ids]
    )
    o_vec = encoder.encode_documents(
        [
            skills_text(
                offers.at[o, "required_skills"], offers.at[o, "nice_to_have_skills"]
            )
            for o in offer_ids
        ]
    )
    labels = grid.pivot(index="profile_id", columns="offer_id", values="silver_label")
    labels = labels.loc[profile_ids, offer_ids].to_numpy()
    precisions, ndcgs = [], []
    for i, similarities in enumerate(p_vec @ o_vec.T):
        order = np.argsort(-similarities)
        ranked = labels[i, order].tolist()
        precisions.append(np.mean([label == 2 for label in ranked[:K]]))
        ndcgs.append(ndcg_at_k(ranked, labels[i].tolist()))
    return {"P@5": float(np.mean(precisions)), "NDCG@5": float(np.mean(ndcgs))}


# ============================================================================
# Programme principal
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"noms ({', '.join(MODEL_SPECS)}) ou identifiant Hugging Face / dossier",
    )
    args = parser.parse_args()
    referential = pd.read_csv(PROCESSED_DIR / "skills_referential.csv")

    results = []
    for name in args.models:
        print(f"\n=== {name} ({MODEL_SPECS.get(name, {'id': name})['id']}) ===")
        try:
            start = time.perf_counter()
            encoder = SentenceEncoder(name)
            load_seconds = time.perf_counter() - start
        except Exception as error:  # modèle protégé, réseau, etc. : on continue
            print(f"  Impossible de charger le modèle : {error}")
            continue
        esco = test_esco_synonyms(encoder, referential)
        cvs = test_cvs(encoder, esco["docs"], esco["labels"])
        matching = test_matching(encoder)
        row = {
            "modèle": name,
            "params (M)": round(
                sum(p.numel() for p in encoder.model.parameters()) / 1e6
            ),
            "dim": encoder.dimension,
            "chargement (s)": round(load_seconds, 1),
            "1009 compétences (s)": round(esco["sec_1009_competences"], 1),
            "ESCO R@1": round(esco["R@1"], 3),
            "ESCO R@5": round(esco["R@5"], 3),
            "CV dév.": round(cvs.get("developpement", float("nan")), 3),
            "CV contrôle": round(cvs.get("controle", float("nan")), 3),
            "P@5": round(matching["P@5"], 3),
            "NDCG@5": round(matching["NDCG@5"], 3),
        }
        print("  " + "  ".join(f"{key}={value}" for key, value in row.items()))
        results.append(row)

    if results:
        print(f"\n{esco['n_synonymes']} synonymes ESCO testés.\n")
        print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
