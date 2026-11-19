import csv
from datetime import date

import pytest

from ematching.cv_parser import (
    UnsupportedCVError,
    education_level,
    heading_of,
    mask_contacts,
    parse_cv,
    parse_text,
    read_text,
    years_of_experience,
)
from ematching.data import PROJECT_ROOT

CV_DIR = PROJECT_ROOT / "data" / "samples" / "cv"
TODAY = date(2026, 11, 1)  # date fixe : « aujourd'hui » donne toujours le même calcul


def expected_rows():
    with open(CV_DIR / "expected.csv", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["education_level"]]


# --- Tests sur les CV fictifs (comparaison avec le corrigé expected.csv) ------


@pytest.mark.parametrize("row", expected_rows(), ids=lambda row: row["cv_file"])
def test_sample_cv_matches_expected(row):
    parsed = parse_cv(CV_DIR / row["cv_file"], today=TODAY)
    assert parsed.education_level == row["education_level"]
    assert parsed.years_experience == int(row["years_experience"])
    assert parsed.city == row["city"]
    assert parsed.languages == row["languages"]


@pytest.mark.parametrize("row", expected_rows(), ids=lambda row: row["cv_file"])
def test_sample_cv_has_no_contact_details(row):
    text = parse_cv(CV_DIR / row["cv_file"]).text
    assert "@exemple.ma" not in text
    assert "00 00 00" not in text


def test_scanned_cv_is_refused():
    with pytest.raises(UnsupportedCVError):
        read_text(CV_DIR / "cv_09_scanne.pdf")


def test_unsupported_format_is_refused(tmp_path):
    path = tmp_path / "cv.txt"
    path.write_text("Compétences : Python", encoding="utf-8")
    with pytest.raises(UnsupportedCVError):
        read_text(path)


def test_two_column_pdf_is_read_column_by_column():
    # CV type Canva : la barre latérale ne doit pas être mélangée au texte principal.
    lines = read_text(CV_DIR / "cv_03_comptable.pdf").splitlines()
    assert "COMPÉTENCES" in lines
    assert "HAMZA EXEMPLE" in lines


def test_right_aligned_dates_stay_with_their_job():
    sections = parse_cv(CV_DIR / "cv_01_developpeur_junior.pdf").sections
    assert "09/2024 – 08/2025" in sections["experience"]


def test_word_text_box_and_page_header_are_read():
    assert "Rabat, Maroc" in read_text(CV_DIR / "cv_02_developpeuse_fullstack.docx")
    assert "Casablanca" in read_text(CV_DIR / "cv_06_charge_rh.docx").splitlines()[1]


# --- Tests unitaires des règles ------------------------------------------------


def test_heading_detection_handles_typos_spacing_and_inline_titles():
    assert heading_of("Exprience") == "experience"
    assert heading_of("M O N   P A R C O U R S") == "experience"
    assert heading_of("COMPÉTENCES") == "skills"
    assert heading_of("Centres d'intérêt") == "interests"
    assert heading_of("LOGICIELS") == "skills"
    assert heading_of("Atouts") == "skills"
    assert heading_of("Développement d'une application web en Python") is None
    sections = parse_text("Compétences : Python, SQL").sections
    assert sections["skills"] == "Python, SQL"


def test_years_of_experience_reads_month_formats_and_merges_overlaps():
    assert years_of_experience("09/2017 - 09/2022\n10/2022 - 10/2025") == 8
    assert years_of_experience("Juin 2022 à Juin 2025") == 3
    assert years_of_experience("Mars 2022 – Aujourd'hui", today=TODAY) == 5
    assert years_of_experience("2018 – 2022\n2020 – 2023") == 5  # chevauchement
    assert years_of_experience("Licence — 2021") is None


def test_declared_years_are_used_when_no_dates():
    text = "Profil comptable. Formation : Bac+3. 6 an(s) d'expérience."
    assert parse_text(text).years_experience == 6


def test_education_level_keeps_the_highest_degree():
    assert education_level("DUT Génie informatique\nLicence professionnelle") == "Bac+3"
    assert education_level("Diplôme d'ingénieur d'État") == "Bac+5"
    assert education_level("Baccalauréat sciences") == "Bac"
    assert education_level("Formation : Bac+2") == "Bac+2"


def test_mask_contacts():
    text = mask_contacts("a.b@exemple.ma | +212 6 00 00 00 01 | 06.00.00.00.04")
    assert text == "[email] | [téléphone] | [téléphone]"


def test_skill_text_ignores_interests_and_education():
    parsed = parse_text(
        "Compétences\nPython\nFormation\nLicence en économie\n"
        "Centres d'intérêt\nélectronique (Arduino)"
    )
    assert parsed.skill_text() == "Python"


def test_skill_text_drops_contact_masks():
    parsed = parse_text("Salé — 06 00 00 00 13 — a.b@exemple.ma\nCompétences\nPython")
    assert "[téléphone]" in parsed.text
    assert "[téléphone]" not in parsed.skill_text()
    assert "[email]" not in parsed.skill_text()


# Structure fréquente (vue sur un vrai CV) : une ligne « Outils : … » sous chaque
# expérience et chaque projet, et des titres de projets qui nomment un sujet.
REAL_LAYOUT = """PROFIL
Ingénieure data.
EXPÉRIENCE PROFESSIONNELLE
Stagiaire data – Société Exemple 07/2026 – 08/2026
• Développé un tableau de bord
Outils : Python, Streamlit
Stagiaire développeuse – Administration Exemple 01/2025 – 06/2025
• Développé une plateforme web
PROJETS
Audit financier IA – Détection de fraude | Projet académique
• Construit un scoring explicable avec XGBoost
Outils : Python, LangChain
COMPÉTENCES
Outils : Git, Docker
Langues : Arabe, Français"""


def test_inline_labels_do_not_switch_section():
    sections = parse_text(REAL_LAYOUT).sections
    assert "Stagiaire développeuse" in sections["experience"]  # pas dans skills
    assert "Python, Streamlit" in sections["skills"]
    assert "Git, Docker" in sections["skills"]
    assert sections["languages"] == "Arabe, Français"
    assert parse_text(REAL_LAYOUT, today=TODAY).years_experience == 1  # 1 + 5 mois


def test_project_titles_are_ignored_for_skills():
    skill_text = parse_text(REAL_LAYOUT).skill_text()
    assert "Audit financier IA" not in skill_text  # sujet du projet
    assert "Construit un scoring explicable" in skill_text  # tâche réalisée
