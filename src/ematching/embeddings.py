"""Modèles d'embeddings : configuration et chargement (un seul endroit du projet).

Un « encodeur » transforme des textes en vecteurs normalisés : le produit scalaire
de deux vecteurs est alors leur similarité cosinus (1 = même sens).

Le reste du code ne dépend que de deux méthodes, encode_queries (textes du CV) et
encode_documents (compétences, offres). N'importe quel objet qui les fournit peut
être utilisé : c'est ce qui permet de changer de modèle sans réécrire le code, et
de tester avec un faux modèle sans rien télécharger.

Choix du projet : Qwen3-0.6B (meilleur sur nos 3 tests : +4,8 points de
NDCG@5 dans le moteur complet ; environ 1 s par CV sur processeur, ~2,5 Go de
mémoire). Solution de secours si une machine manque de mémoire : e5-small
(20 fois plus rapide, un peu moins précis). Voir scripts/benchmark_embeddings.py
et scripts/evaluate_matching.py.
Installation : pip install -e ".[dev,nlp]"
"""

import numpy as np

# Certains modèles attendent un préfixe ou une instruction devant le texte ;
# sans eux ils fonctionnent moins bien, sans message d'erreur.
QWEN_INSTRUCTION = (
    "Instruct: Given a sentence from a CV, retrieve the matching professional skill"
    "\nQuery: "
)
MODEL_SPECS = {
    "e5-small": {
        "id": "intfloat/multilingual-e5-small",
        "query_prompt": "query: ",
        "doc_prompt": "passage: ",
        "skill_threshold": 0.89,
    },
    "minilm": {"id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"},
    "e5-base": {
        "id": "intfloat/multilingual-e5-base",
        "query_prompt": "query: ",
        "doc_prompt": "passage: ",
    },
    "qwen3-0.6b": {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "query_prompt": QWEN_INSTRUCTION,
        "skill_threshold": 0.70,
    },
    # Options : compte Hugging Face requis (gemma) / code distant exécuté (gte).
    "gemma": {
        "id": "google/embeddinggemma-300m",
        "query_prompt_name": "query",
        "doc_prompt_name": "document",
    },
    "gte": {"id": "Alibaba-NLP/gte-multilingual-base", "trust_remote_code": True},
}
# skill_threshold : similarité à partir de laquelle deux libellés de compétences
# sont jugés équivalents. Réglé sur les ~980 synonymes ESCO pour que 80 % des
# équivalences acceptées soient justes (scripts/evaluate_extraction.py --model,
# mesuré le 19/11/2026). Il dépend du modèle : les échelles de similarité diffèrent.
DEFAULT_MODEL = "qwen3-0.6b"
FALLBACK_MODEL = "e5-small"


class SentenceEncoder:
    """Encodeur basé sur sentence-transformers, exécuté sur le processeur."""

    def __init__(self, name: str = DEFAULT_MODEL):
        import torch  # imports lourds : seulement si on utilise les embeddings
        from sentence_transformers import SentenceTransformer

        self.name = name
        # Nom inconnu : identifiant Hugging Face ou dossier local, sans préfixe.
        self.spec = MODEL_SPECS.get(name, {"id": name})
        # float32 : certains modèles (Qwen3) se chargent sinon en bfloat16, un
        # format très lent sur la plupart des processeurs d'ordinateur portable.
        self.model = SentenceTransformer(
            self.spec["id"],
            device="cpu",
            trust_remote_code=self.spec.get("trust_remote_code", False),
            model_kwargs={"dtype": torch.float32},
        )

    @property
    def dimension(self) -> int:
        # Méthode renommée dans sentence-transformers 6 : on accepte les deux noms.
        method = getattr(self.model, "get_embedding_dimension", None)
        return method() if method else self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str], kind: str) -> np.ndarray:
        options = {"batch_size": 32, "normalize_embeddings": True}
        if f"{kind}_prompt_name" in self.spec:  # prompts définis par le modèle
            options["prompt_name"] = self.spec[f"{kind}_prompt_name"]
        else:
            options["prompt"] = self.spec.get(f"{kind}_prompt", "")
        return np.asarray(self.model.encode(list(texts), **options))

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Textes à analyser (lignes de CV, profils)."""
        return self._encode(texts, "query")

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Textes de référence (compétences ESCO, offres)."""
        return self._encode(texts, "doc")
