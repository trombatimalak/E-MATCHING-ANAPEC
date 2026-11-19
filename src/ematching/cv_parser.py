"""Lecture des CV (PDF, DOCX) : texte, sections et informations clés.

Point d'entrée : parse_cv(chemin) -> ParsedCV.

Les valeurs extraites suivent le format du dataset (profiles.csv) :
niveau d'études « Bac+3 », villes « Fès », langues « Arabe|Français ».
Le nom, l'e-mail et le téléphone ne servent jamais au matching : e-mails et
téléphones sont masqués dans le texte renvoyé.

Limites connues : les CV scannés (images) ne sont pas pris en charge (pas d'OCR),
ni l'ancien format Word .doc.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pdfplumber
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from rapidfuzz import fuzz

from ematching.skills import normalize_text


class UnsupportedCVError(ValueError):
    """CV illisible : format non pris en charge, ou PDF scanné (sans texte)."""


# ============================================================================
# Paramètres (modifiables sans toucher au reste du code)
# ============================================================================

# Titres de sections reconnus, sous forme normalisée (minuscules, sans accents).
SECTION_TITLES = {
    "profile": ["profil", "profile", "a propos", "resume", "summary", "objectif"],
    "contact": ["contact", "coordonnees", "informations personnelles"],
    "experience": [
        "experience",
        "experiences",
        "experience professionnelle",
        "experiences professionnelles",
        "parcours",
        "mon parcours",
        "parcours professionnel",
        "work experience",
        "professional experience",
    ],
    "education": [
        "formation",
        "formations",
        "diplomes",
        "diplome",
        "etudes",
        "cursus",
        "education",
    ],
    "certifications": ["formations complementaires", "certifications", "certificats"],
    "skills": [
        "competences",
        "competences cles",
        "competences techniques",
        "savoir faire",
        "skills",
        "technical skills",
        "outils",
        "outils informatiques",
        "logiciels",
        "informatique",
        "atouts",
        "points forts",
    ],
    "languages": ["langues", "languages", "langues et outils"],
    "projects": ["projets", "projects", "realisations"],
    "qualities": ["qualites", "savoir etre", "soft skills"],
    "interests": ["centres d interet", "loisirs", "interets", "hobbies"],
}
# Sections où chercher les compétences. On ignore volontairement : formation
# (un nom de diplôme n'est pas une compétence démontrée), langues, contact,
# qualités et centres d'intérêt (« électronique (Arduino) » est un loisir).
SKILL_SECTIONS = (
    "header",
    "profile",
    "skills",
    "experience",
    "projects",
    "certifications",
)
HEADING_MAX_LENGTH = 40  # un titre de section est une ligne courte
HEADING_MIN_SIMILARITY = 85  # tolère une faute de frappe (« Exprience »)
COLUMN_GAP = 5  # PDF : vide minimal (en points) de part et d'autre d'une colonne
MIN_COLUMN_SHARE = 0.2  # PDF : part minimale des mots dans chacune des 2 colonnes
# Marqueur inséré entre deux blocs indépendants (colonnes, zones de texte) : la
# section en cours s'arrête, pour que le nom écrit en haut de la colonne de droite
# ne soit pas rangé dans la dernière section de la colonne de gauche.
BLOCK_BREAK = "\f"

# Du plus haut au plus bas : on garde le diplôme le plus élevé trouvé.
EDUCATION_PATTERNS = [
    ("Bac+5", r"\b(master|mastere|ingenieur|doctorat|bac\+5)\b"),
    ("Bac+3", r"\b(licence|bachelor|bac\+3)\b"),
    ("Bac+2", r"\b(bts|dut|deug|technicien specialise|bac\+2)\b"),
    ("Bac", r"\b(baccalaureat|bac)\b(?!\+)"),
]
# Villes du dataset en premier, puis d'autres villes (graphie canonique).
CITIES = [
    "Casablanca",
    "Rabat",
    "Marrakech",
    "Tanger",
    "Fès",
    "Agadir",
    "Oujda",
    "Meknès",
    "Kénitra",
    "Tétouan",
    "Salé",
    "Mohammedia",
    "El Jadida",
    "Safi",
    "Nador",
    "Béni Mellal",
    "Laâyoune",
]
LANGUAGES = {  # libellé du dataset -> mots reconnus (normalisés)
    "Arabe": ["arabe", "arabic"],
    "Français": ["francais", "french"],
    "Anglais": ["anglais", "english"],
    "Espagnol": ["espagnol", "spanish"],
    "Allemand": ["allemand", "german"],
    "Amazigh": ["amazigh", "tamazight", "berbere"],
}
MONTHS = {  # début des noms de mois (français et anglais, sans accents)
    "jan": 1,
    "fev": 2,
    "feb": 2,
    "mar": 3,
    "avr": 4,
    "apr": 4,
    "mai": 5,
    "may": 5,
    "juin": 6,
    "jun": 6,
    "juil": 7,
    "jul": 7,
    "aou": 8,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# --- Expressions régulières (appliquées sur du texte en minuscules, sans accents)
_MONTH = (
    r"(?:janv?|janvier|fevr?|fevrier|feb|mars?|avr|avril|apr|mai|may|juin|jun|"
    r"juil|juillet|jul|aout|aug|sept?|septembre|oct|octobre|nov|novembre|dec|"
    r"decembre)\.?"
)
_DATE = rf"(?:(?:{_MONTH}|\d{{1,2}})\s*[/.]?\s*)?(?:19|20)\d{{2}}"
_ONGOING = r"aujourd.?hui|present|ce jour|actuel|en cours|now|today"
DATE_RANGE = re.compile(
    rf"(?<![\d/])({_DATE})\s*(?:-|–|—|\ba\b|\bau\b|\bto\b)\s*({_DATE}|{_ONGOING})"
)
BULLET = re.compile(r"\s*[•\-–—▪›*○●]\s")
MASK_TOKENS = re.compile(r"\[(?:email|téléphone)\]")
CID = re.compile(r"\(cid:\d+\)")
DECLARED_YEARS = re.compile(r"(\d{1,2})\s*an(?:s|\(s\))?\s+d.?\s?experience")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
PHONE = re.compile(r"(?:\+212\s?(?:\(0\))?\s?|0)[5-7](?:[\s.-]?\d){8}")


@dataclass
class ParsedCV:
    """Résultat de la lecture d'un CV."""

    text: str  # texte complet (e-mails et téléphones masqués)
    sections: dict[str, str] = field(default_factory=dict)
    education_level: str | None = None  # « Bac », « Bac+2 », « Bac+3 », « Bac+5 »
    years_experience: int | None = None
    city: str | None = None
    languages: str = ""  # « Arabe|Français »

    def skill_text(self) -> str:
        """Texte des sections utiles pour extraire les compétences (tout le texte
        si aucune section de contenu n'a été reconnue).

        Dans « projects », seules les puces (tâches réalisées) sont gardées quand
        il y en a : le titre d'un projet décrit son sujet (« Audit financier IA »,
        « Analyste de la Bourse »), pas une compétence de la personne."""
        if not any(name in self.sections for name in SKILL_SECTIONS[1:]):
            text = self.text
        else:
            parts = []
            for name in SKILL_SECTIONS:
                if name in self.sections:
                    content = self.sections[name]
                    parts.append(
                        bullets_only(content) if name == "projects" else content
                    )
            text = "\n".join(parts)
        # Les marqueurs de masquage ne doivent pas être pris pour des compétences
        # (« [téléphone] » ressemblerait à « communiquer par téléphone »).
        return MASK_TOKENS.sub("", text)

    def to_profile(self) -> dict:
        """Mêmes colonnes que profiles.csv (les compétences sont ajoutées après)."""
        return {
            "education_level": self.education_level,
            "years_experience": self.years_experience,
            "city": self.city,
            "languages": self.languages,
            "cv_text": self.text,
        }


# ============================================================================
# 1. Lecture du fichier -> texte brut, dans l'ordre de lecture
# ============================================================================


def read_text(path: str | Path) -> str:
    """Retourne le texte d'un CV PDF ou DOCX."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix == ".docx":
        text = _read_docx(path)
    else:
        raise UnsupportedCVError(f"Format non pris en charge : {suffix} (PDF ou DOCX)")
    if not text.strip():
        raise UnsupportedCVError("Aucun texte trouvé : CV scanné non pris en charge")
    return text


# --- PDF ---------------------------------------------------------------------


def _read_pdf(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(_page_text(page) for page in pdf.pages)


def _page_text(page) -> str:
    """Texte d'une page, en lisant les zones à deux colonnes colonne par colonne
    (sinon pdfplumber mélange les deux colonnes sur chaque ligne)."""
    lines = _group_into_lines(page.extract_words())
    split = _find_column_split(lines, page.width)
    if split is None:  # une seule colonne
        return "\n".join(_join(line) for line in lines)
    x, start, end = split  # les lignes start..end-1 forment la zone à deux colonnes
    block = lines[start:end]
    left = [[w for w in line if w["x1"] <= x] for line in block]
    right = [[w for w in line if w["x0"] >= x] for line in block]
    text = [_join(line) for line in lines[:start] + left if line]
    text += [BLOCK_BREAK] + [_join(line) for line in right + lines[end:] if line]
    return "\n".join(text)


def _group_into_lines(words: list[dict], tolerance: float = 3) -> list[list[dict]]:
    """Regroupe les mots ayant (presque) la même position verticale."""
    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(lines[-1][0]["top"] - word["top"]) <= tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [sorted(line, key=lambda w: w["x0"]) for line in lines]


def _find_column_split(lines: list[list[dict]], width: float):
    """Cherche une séparation verticale entre deux colonnes de texte.

    Pour chaque position x (entre 30 % et 70 % de la largeur), on cherche la plus
    longue suite de lignes qu'aucun mot ne traverse, avec au moins 3 lignes de
    texte de chaque côté. Retourne (x, début, fin), ou None (une seule colonne).
    """
    best = None
    for x in range(int(width * 0.3), int(width * 0.7), 2):
        start = 0
        for index in range(len(lines) + 1):
            crosses = index == len(lines) or any(
                w["x0"] - COLUMN_GAP < x < w["x1"] + COLUMN_GAP for w in lines[index]
            )
            if not crosses:
                continue
            block = lines[start:index]
            if _is_two_columns(block, x):
                if best is None or len(block) > best[2] - best[1]:
                    best = (x, start, index)
            start = index + 1
    return best


def _is_two_columns(block: list[list[dict]], x: float) -> bool:
    """Vrai si le bloc contient deux vraies colonnes de texte de part et d'autre
    de x : au moins 3 lignes et 20 % des mots de chaque côté.

    Sinon, la partie droite n'est qu'un détail aligné sur la même ligne (dates
    alignées à droite, niveau en étoiles…) et il faut lire ligne par ligne.
    """
    left = [w for line in block for w in line if w["x1"] <= x]
    right = [w for line in block for w in line if w["x0"] >= x]
    n_left = sum(any(w["x1"] <= x for w in line) for line in block)
    n_right = sum(any(w["x0"] >= x for w in line) for line in block)
    total = len(left) + len(right)
    # Petits éléments à droite, toujours sur une ligne qui a aussi du texte à
    # gauche (ex. « WMS ★★★☆☆ ») : c'est un tableau à lire ligne par ligne.
    paired = sum(
        any(w["x1"] <= x for w in line) and any(w["x0"] >= x for w in line)
        for line in block
    )
    row_layout = paired >= 0.8 * n_right and len(right) <= 4 * n_right
    return (
        n_left >= 3
        and n_right >= 3
        and len(left) >= MIN_COLUMN_SHARE * total
        and len(right) >= MIN_COLUMN_SHARE * total
        and not row_layout
    )


def _join(line: list[dict]) -> str:
    # « (cid:127) » : symbole sans correspondance dans la police (souvent une puce).
    return CID.sub("•", " ".join(word["text"] for word in line))


# --- DOCX --------------------------------------------------------------------


def _read_docx(path: Path) -> str:
    """En-têtes de page, puis corps du document (paragraphes, tableaux et zones de
    texte). Les pieds de page sont ignorés."""
    document = Document(path)
    lines, seen_headers = [], set()
    for section in document.sections:
        header = section.header
        if not header.is_linked_to_previous and id(header.part) not in seen_headers:
            seen_headers.add(id(header.part))
            lines.extend(_docx_lines(header))
    lines.extend(_docx_lines(document))
    return "\n".join(lines)


def _docx_lines(container) -> list[str]:
    """Lignes d'un conteneur Word (document, cellule, en-tête), dans l'ordre."""
    lines = []
    for child in container.iter_inner_content():
        if isinstance(child, Paragraph):
            lines.append(child.text)
            lines.extend(_textbox_lines(child._p))
        elif isinstance(child, Table):
            seen = set()  # une cellule fusionnée apparaît plusieurs fois
            for row in child.rows:
                for cell in row.cells:
                    if id(cell._tc) not in seen:
                        seen.add(id(cell._tc))
                        lines.extend(_docx_lines(cell))
    return lines


def _textbox_lines(paragraph_element) -> list[str]:
    """Texte des zones de texte ancrées dans un paragraphe (python-docx ne les lit
    pas). Word enregistre souvent chaque zone deux fois (version moderne + version
    de secours « Fallback ») : on ignore la version de secours."""
    lines = []
    for box in paragraph_element.iter(qn("w:txbxContent")):
        ancestors = {element.tag for element in box.iterancestors()}
        if MC_FALLBACK in ancestors:
            continue
        lines.append(BLOCK_BREAK)
        for p in box.iter(qn("w:p")):
            lines.append("".join(t.text or "" for t in p.iter(qn("w:t"))))
        lines.append(BLOCK_BREAK)
    return lines


# ============================================================================
# 2. Texte brut -> sections et informations clés
# ============================================================================


def bullets_only(text: str) -> str:
    """Garde les lignes à puce (et leurs suites de phrase) ; tout le texte s'il
    n'y a aucune puce."""
    lines = text.splitlines()
    if not any(BULLET.match(line) for line in lines):
        return text
    kept, in_bullet = [], False
    for line in lines:
        if BULLET.match(line):
            in_bullet = True
        elif line[:1].isupper():  # nouvelle ligne de titre : fin de la puce
            in_bullet = False
        if in_bullet:
            kept.append(line)
    return "\n".join(kept)


def mask_contacts(text: str) -> str:
    """Masque les e-mails et numéros de téléphone (données personnelles)."""
    return PHONE.sub("[téléphone]", EMAIL.sub("[email]", text))


def _fold(text: str) -> str:
    """Minuscules sans accents, en gardant la ponctuation (tirets des dates)."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def heading_of(line: str) -> str | None:
    """Retourne le nom de la section si la ligne est un titre, sinon None.

    Les espaces sont ignorés dans la comparaison, pour reconnaître aussi les titres
    espacés lettre par lettre (« M O N   P A R C O U R S »).
    """
    compact = normalize_text(line).replace(" ", "")
    if not compact or len(compact) > HEADING_MAX_LENGTH:
        return None
    for section, titles in SECTION_TITLES.items():
        for title in titles:
            if fuzz.ratio(compact, title.replace(" ", "")) >= HEADING_MIN_SIMILARITY:
                return section
    return None


def split_sections(text: str) -> dict[str, str]:
    """Découpe le texte en sections. Ce qui précède le premier titre va dans
    « header ».

    Une ligne « étiquette : contenu » (« Outils : Python, SQL » sous une
    expérience, « Langues : Arabe ») range son contenu dans la section de
    l'étiquette SANS changer la section en cours : les lignes suivantes restent
    dans l'expérience ou le projet décrit."""
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    for raw in text.split("\n"):
        if raw == BLOCK_BREAK:  # nouveau bloc : on ne prolonge pas la section
            current = "header"
            continue
        line = raw.strip()
        if not line:
            continue
        head, colon, rest = line.partition(":")
        if colon and rest.strip() and heading_of(head):  # étiquette : contenu
            sections.setdefault(heading_of(head), []).append(rest.strip())
            continue
        section = heading_of(head if colon else line)
        if section:  # vrai titre de section : on change de section
            current = section
            sections.setdefault(current, [])
        else:
            sections[current].append(line)
    return {name: "\n".join(lines) for name, lines in sections.items() if lines}


def _to_month_index(value: str, today: date) -> int:
    """« 09/2024 », « mars 2022 », « 2023 » ou « aujourd'hui » -> année*12 + mois."""
    if re.fullmatch(_ONGOING, value):
        return today.year * 12 + today.month
    year = int(re.search(r"(?:19|20)\d{2}", value).group())
    month_number = re.match(r"(\d{1,2})\s*[/.]?\s*(?:19|20)\d{2}", value)
    month_name = re.match(r"([a-z]+)", value)
    if month_number and 1 <= int(month_number.group(1)) <= 12:
        month = int(month_number.group(1))
    elif month_name:
        month = next(
            (n for key, n in MONTHS.items() if month_name.group(1).startswith(key)), 1
        )
    else:  # année seule : « 2023 – 2025 » compte 2 ans
        month = 1
    return year * 12 + month


def years_of_experience(text: str, today: date | None = None) -> int | None:
    """Durée totale des périodes trouvées (« 09/2019 – 06/2022 », « Mars 2022 –
    aujourd'hui », « 2023-2025 »…), arrondie à l'année la plus proche.

    Les périodes qui se chevauchent ne sont comptées qu'une fois.
    """
    today = today or date.today()
    periods = []
    for start, end in DATE_RANGE.findall(_fold(text)):
        begin = _to_month_index(start, today)
        finish = _to_month_index(end, today)
        if begin <= finish:
            periods.append((begin, finish))
    if not periods:
        return None
    months, current = 0, None
    for begin, finish in sorted(periods):  # fusion des périodes qui se chevauchent
        if current is None or begin > current[1]:
            if current:
                months += current[1] - current[0]
            current = [begin, finish]
        else:
            current[1] = max(current[1], finish)
    months += current[1] - current[0]
    return int(months / 12 + 0.5)


def declared_years(text: str) -> int | None:
    """Expérience annoncée en toutes lettres : « 5 ans d'expérience »."""
    match = DECLARED_YEARS.search(_fold(text))
    return int(match.group(1)) if match else None


def education_level(text: str) -> str | None:
    """Plus haut niveau d'études mentionné, ramené à l'échelle du dataset."""
    key = normalize_text(text)
    for level, pattern in EDUCATION_PATTERNS:
        if re.search(pattern, key):
            return level
    return None


def find_city(text: str) -> str | None:
    """Première ville connue citée dans le texte."""
    key = f" {normalize_text(text)} "
    found = [(key.find(f" {normalize_text(c)} "), c) for c in CITIES]
    found = [(position, city) for position, city in found if position >= 0]
    return min(found)[1] if found else None


def find_languages(text: str) -> str:
    """Langues citées, au format du dataset (« Arabe|Français|Anglais »)."""
    words = set(normalize_text(text).split())
    return "|".join(label for label, keys in LANGUAGES.items() if words & set(keys))


def parse_text(text: str, today: date | None = None) -> ParsedCV:
    """Analyse un texte de CV déjà extrait.

    Chaque information est cherchée d'abord dans sa section (ex. l'expérience dans
    « experience »), puis dans tout le texte si la section est absente.
    """
    sections = split_sections(mask_contacts(text))
    text = mask_contacts(text.replace(BLOCK_BREAK + "\n", "").replace(BLOCK_BREAK, ""))

    def section(name: str) -> str:
        return sections.get(name, text)

    # La ville du candidat est près du nom (en-tête) ou dans le bloc « Contact » ;
    # les autres villes du CV sont souvent celles des employeurs ou des écoles.
    # Les périodes datées priment ; sinon « 5 ans d'expérience » dans le texte.
    years = years_of_experience(section("experience"), today)
    if years is None:
        years = declared_years(text)

    city = (
        find_city(sections.get("header", ""))
        or find_city(sections.get("contact", ""))
        or find_city(text)
    )
    return ParsedCV(
        text=text,
        sections=sections,
        education_level=education_level(section("education")),
        years_experience=years,
        city=city,
        languages=find_languages(section("languages")),
    )


def parse_cv(path: str | Path, today: date | None = None) -> ParsedCV:
    """Lit un CV PDF ou DOCX et en extrait les informations clés.

    `today` sert à calculer les périodes « … – aujourd'hui » (par défaut : la date
    du jour ; les tests passent une date fixe pour être reproductibles).
    """
    return parse_text(read_text(path), today)
