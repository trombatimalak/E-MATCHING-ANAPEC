# Data card : jeu de données E-Matching (v0)

## Objet
Données de **prototypage** pour le projet capstone E-Matching (inspiré de l'AO ANAPEC
n°13/2026). **Aucune donnée réelle de l'ANAPEC n'est utilisée.** Toutes les lignes
sont synthétiques (colonne `synthetic = True`).

## Sources
- **ESCO v1.2.1** (Commission européenne), version française, CSV, téléchargée le
  07/11/2026 : https://esco.ec.europa.eu/en/use-esco/download
  Fichiers : `occupations_fr.csv`, `skills_fr.csv`, `occupationSkillRelations_fr.csv`.
  Licence : contenu ESCO © Union européenne, réutilisable avec mention de la source
  (CC BY 4.0, sauf mention contraire ; voir la notice de copyright ESCO :
  https://esco.ec.europa.eu/en/copyright-notice-esco-skills-competences).
  Modifications apportées : sélection de 28 métiers, filtrage et assemblage des
  compétences dans des offres et profils synthétiques.
- **Génération synthétique** : scripts du dossier `scripts/` (graine fixe `42`,
  résultats reproductibles).

## Périmètre
6 métiers cibles : développeur, comptable, commercial, maintenance, ressources
humaines, logistique. Langue : français.

## Fichiers (`data/processed/`)
| Fichier | Contenu | Volume |
|---|---|---|
| `target_occupations.csv` | Métiers ESCO retenus par métier cible | 28 métiers |
| `skills_referential.csv` | Référentiel de compétences (ESCO) | 1 009 compétences |
| `offers.csv` | Offres synthétiques | 150 (25 par métier) |
| `profiles.csv` | Profils synthétiques (sans identité) | 300 (50 par métier) |
| `pairs.csv` | Paires profil-offre avec label provisoire | 1 742 |
| `test_grid.csv` | Toutes les paires du split test (évaluation du ranking) | 4 320 |
| `annotation_sample.csv` | Paires à annoter par deux personnes | 400 |

## Schémas
- `offers` : offer_id, title, target_job, sector, city, contract_type,
  education_level, min_experience_years, required_skills, nice_to_have_skills,
  description, source_occupation_uri, synthetic.
- `profiles` : profile_id, desired_job, education_level, years_experience, city,
  languages, skills, cv_text, source_occupation_uri, synthetic.
- Compétences séparées par `|`.
- Aucun nom, âge, genre ni photo : ces attributs ne servent pas au classement.

## Construction
1. Sélection de métiers ESCO par mots-clés, avec exclusion des métiers hors sujet.
2. Offres : 6 compétences essentielles + 3 optionnelles tirées au hasard pour le métier ESCO.
3. Profils : 3 à 8 compétences essentielles, 0 à 3 optionnelles et 0 à 2 compétences
   « bruit » d'un autre métier. Niveau d'études cohérent avec le poste.
4. Paires : séparation train/test **par profil et par offre** (70/30, stratifiée par
   métier) pour éviter les fuites. Pour chaque profil : 2 offres du même métier ESCO,
   1 du même domaine, 2 négatifs difficiles, 1 négatif facile.
5. Label provisoire (0 = non pertinent, 1 = partiel, 2 = pertinent) par règle, à valider
   sur l'échantillon annoté (voir `annotation_guide.md`).

## Répartition des paires
| Split | Label 0 | Label 1 | Label 2 | Total |
|---|---|---|---|---|
| train | 624 | 323 | 307 | 1 254 |
| test | 267 | 122 | 99 | 488 |

## Limites
- Profils et offres sont tirés du même référentiel ESCO : le matching est plus facile
  que sur des données réelles.
- Les CV sont très gabaritables (texte généré) : ils ne permettent pas de tester le
  parsing PDF/DOCX. Prévoir quelques CV fictifs rédigés à la main.
- Les labels provisoires sont construits à partir du métier source : les colonnes
  `source_occupation_uri`, `desired_job` et `target_job` ne doivent **jamais** servir
  de variables d'entrée du modèle.
- Certaines compétences ESCO sont génériques (par exemple « gérer les budgets ») et
  partagées entre métiers.
- Données ANAPEC non disponibles à ce jour : perspective.