"""Normalisation et extraction des compétences à partir d'ESCO."""

import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process


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


# ============================================================================
# Extraction améliorée (P2) : SkillExtractor
# ============================================================================
# Les fonctions ci-dessus restent inchangées (utilisées par le reste du projet).
# SkillExtractor les complète en cascade, de la méthode la plus sûre à la plus large :
#   1. exact     : libellé ESCO (préféré ou alternatif) écrit tel quel ;
#   2. synonyme  : terme du lexique local (data/lexicon/skills_synonyms.csv) ;
#   3. forme     : même compétence écrite sous une autre forme
#                  (« gestion des stocks » = « gérer des stocks ») ;
#   4. approché  : faute de frappe (« gestion des stoks »), via rapidfuzz ;
#   5. sémantique (optionnel, si un modèle d'embeddings est fourni) : compétence
#      ESCO la plus proche par le sens (« Encadrement d'une équipe » -> « gérer le
#      personnel »). Acceptée au-dessus d'un seuil, « à revoir » juste en dessous.
# Chaque compétence trouvée garde sa méthode, une confiance et la ligne du CV
# d'où elle vient : le résultat est explicable et vérifiable.

MIN_KEY_LENGTH = 3  # ignore les sigles ESCO de 1-2 lettres (« R », « C ») : ambigus
FUZZY_MIN_SCORE = 90  # similarité minimale (sur 100) pour une faute de frappe
FUZZY_MIN_LENGTH = 8  # pas de rapprochement approché sur les mots courts
FUZZY_MAX_WORDS = 6
# Pas de recherche approchée sur un mot isolé : « automatisé » n'est pas une
# faute de frappe pour « automatisme ». Sur 2 mots ou plus, le contexte limite
# ces confusions (« gestion des stoks » reste reconnu).
FUZZY_MIN_WORDS = 2
FUZZY_MIN_WORD_SCORE = 75  # chaque mot doit aussi ressembler à son correspondant
CONFIDENCE = {"exact": 1.0, "synonyme": 0.9, "forme": 0.85}  # « approché » : ≤ 0,8
# Méthode sémantique. Le seuil d'acceptation dépend du modèle (les similarités
# n'ont pas la même échelle d'un modèle à l'autre) : il est réglé sur les CV de
# développement avec scripts/evaluate_extraction.py, puis passé à l'extracteur.
SEMANTIC_MIN_WORDS = 2  # une ligne d'un seul mot est trop pauvre en sens
SEMANTIC_TOP_K = 2  # une ligne peut contenir deux compétences
REVIEW_MARGIN = 0.03  # zone « à revoir » : [seuil - 0,03 ; seuil[

# Mots vides ignorés quand on compare deux formulations.
STOPWORDS = {
    "a", "au", "aux", "avec", "d", "de", "des", "du", "en", "et", "l", "la",
    "le", "les", "pour", "sur", "un", "une",
}  # fmt: skip
# Noms d'action courants dans les CV -> verbe utilisé par les libellés ESCO.
NOUN_TO_VERB = {
    "administration": "administrer",
    "analyse": "analyser",
    "conception": "concevoir",
    "controle": "controler",
    "coordination": "coordonner",
    "developpement": "developper",
    "elaboration": "elaborer",
    "entretien": "entretenir",
    "etablissement": "etablir",
    "gestion": "gerer",
    "installation": "installer",
    "negociation": "negocier",
    "organisation": "organiser",
    "planification": "planifier",
    "preparation": "preparer",
    "prospection": "prospecter",
    "realisation": "realiser",
    "recrutement": "recruter",
    "redaction": "rediger",
    "reparation": "reparer",
    "saisie": "saisir",
    "suivi": "suivre",
    "supervision": "superviser",
    "utilisation": "utiliser",
    "verification": "verifier",
}


@dataclass(frozen=True)
class SkillMatch:
    """Une compétence trouvée dans un texte."""

    uri: str  # identifiant ESCO
    label: str  # libellé ESCO (pour l'affichage)
    method: str  # exact | synonyme | forme | approché
    confidence: float  # 1.0 = sûr ; sert à trier et à filtrer
    source: str  # ligne du texte d'où vient la compétence
    status: str = "ok"  # « ok » ou « à revoir » (sémantique sous le seuil)


def label_variants(label: str) -> set[str]:
    """Formes normalisées d'un libellé, avec et sans précision entre parenthèses :
    « Python (programmation informatique) » -> {« python programmation… », « python »}.
    """
    variants = {normalize_text(label), normalize_text(re.sub(r"\(.*?\)", " ", label))}
    return {variant for variant in variants if variant}


def content_words(key: str) -> str:
    """Retire les mots vides : « gerer des stocks » -> « gerer stocks »."""
    return " ".join(word for word in key.split() if word not in STOPWORDS)


def verbalize(key: str) -> str:
    """Remplace les noms d'action par leur verbe : « gestion des stocks » -> « gerer
    des stocks »."""
    return " ".join(NOUN_TO_VERB.get(word, word) for word in key.split())


def load_synonyms(path: str | Path) -> list[tuple[str, str]]:
    """Lit le lexique local : liste de (terme, libellé ESCO). Ignore les lignes #."""
    with open(path, encoding="utf-8", newline="") as handle:
        return [
            (row["term"].strip(), row["esco_label"].strip())
            for row in csv.DictReader(handle)
            if row["term"].strip() and not row["term"].startswith("#")
        ]


class SkillExtractor:
    """Extrait les compétences ESCO d'un texte (voir l'explication plus haut)."""

    def __init__(
        self,
        referential: pd.DataFrame,
        synonyms=(),
        encoder=None,
        accept_threshold: float | None = None,
        semantic_scope: str = "unmatched",
    ):
        """encoder : objet avec encode_queries / encode_documents (voir
        embeddings.py) ; sans encodeur, la méthode sémantique est désactivée.
        semantic_scope : « unmatched » (lignes où le lexique n'a rien trouvé) ou
        « all » (toutes les lignes)."""
        if encoder is not None and accept_threshold is None:
            raise ValueError("Un seuil d'acceptation est requis avec un encodeur")
        self.encoder = encoder
        self.accept_threshold = accept_threshold
        self.semantic_scope = semantic_scope
        self._lexical_cache: dict[str, dict[str, SkillMatch]] = {}
        self._line_vectors: dict[str, np.ndarray] = {}
        self._label_vectors = None  # calculés au premier besoin
        uri_column = "skillUri" if "skillUri" in referential else "conceptUri"
        rows = list(
            zip(
                referential[uri_column],
                referential["preferredLabel"],
                referential["altLabels"],
            )
        )
        self.labels = {uri: preferred for uri, preferred, _ in rows}
        self.semantic_labels = sorted(set(self.labels.values()))
        uris_by_label = defaultdict(set)  # un libellé peut désigner 2 URI (« SQL »)
        for uri, preferred, _ in rows:
            uris_by_label[preferred].add(uri)

        # Libellé normalisé -> URI. Un dictionnaire de *listes* d'URI, pour garder
        # toutes les compétences qui partagent un libellé.
        self.esco: dict[str, set[str]] = defaultdict(set)
        for uri, preferred, alternatives in rows:
            labels = [preferred]
            if isinstance(alternatives, str):
                labels += alternatives.split("\n")
            for label in labels:
                for key in label_variants(label):
                    if len(key) >= MIN_KEY_LENGTH:
                        self.esco[key].add(uri)
        # Même chose sans les mots vides (au moins 2 mots porteurs de sens).
        self.esco_content: dict[str, set[str]] = defaultdict(set)
        for key, uris in self.esco.items():
            content = content_words(key)
            if len(content.split()) >= 2:
                self.esco_content[content] |= uris

        self.synonyms: dict[str, set[str]] = defaultdict(set)
        for term, esco_label in synonyms:
            if esco_label not in uris_by_label:
                raise ValueError(f"Synonyme « {term} » : « {esco_label} » absent")
            self.synonyms[normalize_text(term)] |= uris_by_label[esco_label]

        # Clés assez longues pour la recherche approchée, rangées par nombre de mots.
        self.fuzzy_keys: dict[int, list[str]] = defaultdict(list)
        for key in set(self.esco) | set(self.synonyms):
            n_words = len(key.split())
            if (
                len(key) >= FUZZY_MIN_LENGTH
                and FUZZY_MIN_WORDS <= n_words <= FUZZY_MAX_WORDS
            ):
                self.fuzzy_keys[n_words].append(key)

    @classmethod
    def from_files(
        cls, referential_path, synonyms_path=None, **options
    ) -> "SkillExtractor":
        """Construit l'extracteur depuis les fichiers CSV du projet."""
        synonyms = load_synonyms(synonyms_path) if synonyms_path else ()
        return cls(pd.read_csv(referential_path), synonyms, **options)

    def extract(self, text: str) -> list[SkillMatch]:
        """Compétences trouvées, de la plus sûre à la moins sûre (les compétences
        « à revoir » sont incluses : filtrer sur match.status si besoin)."""
        best = dict(self._lexical(text))
        if self.encoder is not None:
            self._add_semantic(text, best)
        return sorted(best.values(), key=lambda m: (-m.confidence, m.label))

    def _lexical(self, text: str) -> dict[str, SkillMatch]:
        """Méthodes 1 à 4 (résultat mis en cache : utile pour régler le seuil)."""
        if text in self._lexical_cache:
            return self._lexical_cache[text]
        best: dict[str, SkillMatch] = {}

        def add(uris, method, confidence, line):
            for uri in uris:  # on garde, par compétence, la méthode la plus sûre
                if uri not in best or confidence > best[uri].confidence:
                    best[uri] = SkillMatch(
                        uri, self.labels[uri], method, round(confidence, 2), line
                    )

        for line in text.splitlines():
            norm = normalize_text(line)
            if not norm:
                continue
            line = line.strip()
            plain = f" {norm} "
            verbal = f" {verbalize(norm)} "
            content = f" {content_words(verbalize(norm))} "
            for key, uris in self.esco.items():
                if f" {key} " in plain:
                    add(uris, "exact", CONFIDENCE["exact"], line)
                elif f" {key} " in verbal:
                    add(uris, "forme", CONFIDENCE["forme"], line)
            for key, uris in self.synonyms.items():
                if f" {key} " in plain:
                    add(uris, "synonyme", CONFIDENCE["synonyme"], line)
            for key, uris in self.esco_content.items():
                if f" {key} " in content:
                    add(uris, "forme", CONFIDENCE["forme"], line)
            self._add_fuzzy(norm, line, add)
            self._add_fuzzy(verbalize(norm), line, add)

        self._lexical_cache[text] = best
        return best

    def _add_semantic(self, text: str, best: dict[str, SkillMatch]) -> None:
        """Méthode 5 : pour chaque ligne, les compétences ESCO les plus proches."""
        lines = [
            line.strip()
            for line in text.splitlines()
            if len(normalize_text(line).split()) >= SEMANTIC_MIN_WORDS
        ]
        if self.semantic_scope == "unmatched":  # le lexique a la priorité
            matched = {match.source for match in self._lexical(text).values()}
            lines = [line for line in lines if line not in matched]
        if not lines:
            return
        uris_of = defaultdict(set)
        for uri, label in self.labels.items():
            uris_of[label].add(uri)

        review_threshold = self.accept_threshold - REVIEW_MARGIN
        for line, scores in zip(lines, self.similarities(lines)):
            for index in np.argsort(-scores)[:SEMANTIC_TOP_K]:
                score = float(scores[index])
                if score < review_threshold:
                    break
                status = "ok" if score >= self.accept_threshold else "à revoir"
                label = self.semantic_labels[index]
                for uri in uris_of[label]:
                    # Une compétence déjà trouvée par le lexique garde sa méthode.
                    current = best.get(uri)
                    if current is None or (
                        current.method == "sémantique"
                        and 0.8 * score > current.confidence
                    ):
                        # Confiance plafonnée à 0,8 : toujours sous le lexique.
                        best[uri] = SkillMatch(
                            uri,
                            label,
                            "sémantique",
                            round(0.8 * score, 2),
                            line,
                            status,
                        )

    def similarities(self, lines: list[str]) -> np.ndarray:
        """Similarité de chaque ligne avec chaque compétence de semantic_labels
        (matrice lignes x compétences). Sert aussi à régler le seuil."""
        if self._label_vectors is None:  # calculés une seule fois
            self._label_vectors = self.encoder.encode_documents(self.semantic_labels)
        return self._encode_lines(lines) @ self._label_vectors.T

    def _encode_lines(self, lines: list[str]) -> np.ndarray:
        """Vecteurs des lignes, avec cache (une ligne n'est encodée qu'une fois)."""
        missing = [
            line for line in dict.fromkeys(lines) if line not in self._line_vectors
        ]
        if missing:
            for line, vector in zip(missing, self.encoder.encode_queries(missing)):
                self._line_vectors[line] = vector
        return np.array([self._line_vectors[line] for line in lines])

    def _add_fuzzy(self, norm: str, line: str, add) -> None:
        """Compare chaque groupe de 1 à 6 mots de la ligne aux clés connues."""
        words = norm.split()
        for n_words, keys in self.fuzzy_keys.items():
            for start in range(len(words) - n_words + 1):
                chunk = " ".join(words[start : start + n_words])
                if len(chunk) < FUZZY_MIN_LENGTH:
                    continue
                match = process.extractOne(
                    chunk, keys, scorer=fuzz.ratio, score_cutoff=FUZZY_MIN_SCORE
                )
                # score 100 : déjà trouvé par une méthode exacte.
                if match and match[1] < 100 and _same_words(chunk, match[0]):
                    key, score = match[0], match[1]
                    uris = self.esco.get(key) or self.synonyms.get(key)
                    add(uris, "approché", 0.8 * score / 100, line)


def _same_words(chunk: str, key: str) -> bool:
    """Vrai si chaque mot du morceau ressemble au mot correspondant de la clé.

    Évite qu'une phrase proche « en moyenne » soit acceptée alors qu'un mot entier
    diffère (« developpement de » n'est pas « developpement web »).
    """
    pairs = list(zip(chunk.split(), key.split()))
    return all(
        a == b if min(len(a), len(b)) <= 3 else fuzz.ratio(a, b) >= FUZZY_MIN_WORD_SCORE
        for a, b in pairs
    )


def choose_threshold(
    scores, correct, target_precision: float = 0.8, min_accepted: int = 50
) -> float | None:
    """Plus petit seuil tel que, parmi les réponses acceptées (score >= seuil), au
    moins `target_precision` soient justes, avec au moins `min_accepted` réponses
    (sinon la précision mesurée serait trop incertaine). None si aucun ne convient.

    Le plus petit seuil valable accepte le plus de réponses : meilleur rappel à
    précision garantie.
    """
    scores, correct = np.asarray(scores, dtype=float), np.asarray(correct, dtype=bool)
    for threshold in np.unique(np.round(scores, 2)):  # seuils croissants
        accepted = scores >= threshold
        if accepted.sum() < min_accepted:
            return None
        if correct[accepted].mean() >= target_precision:
            return float(threshold)
    return None
