# Guide d'annotation des paires profil-offre

## Objectif
Juger si une offre est une recommandation raisonnable pour un profil.
Fichier à remplir : `data/processed/annotation_sample.csv`
(colonnes `label_annotator_a` et `label_annotator_b`).

## Grille
| Label | Signification | Critères |
|---|---|---|
| 2 | Pertinent | Métier correspondant, au moins la moitié des compétences requises couvertes, expérience suffisante |
| 1 | Partiellement pertinent | Métier proche, compétences partielles ou expérience insuffisante |
| 0 | Non pertinent | Domaine différent ou compétences essentielles absentes |

## Critères
- **Obligatoires** : domaine du poste et compétences requises.
- **Souhaitables** : niveau d'études, ville, langues.

## Règles
- Ne pas utiliser le nom, la photo, l'âge, le genre ni d'autres attributs sensibles.
- Annoter sans regarder `silver_label` ni `pair_type` (volontairement absents du fichier).
- En cas de doute, choisir le label inférieur.

## Procédure
1. Deux annotateurs remplissent au moins 100 paires communes.
2. Calculer l'accord (kappa de Cohen) et lister les désaccords.
3. Trancher ensemble, puis répartir le reste de l'échantillon.
4. Comparer les labels humains aux labels provisoires pour mesurer leur fiabilité.