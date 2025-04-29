import spacy
from importlib import resources

PKG = resources.files("parzivai")
mhg_model_path = PKG.parents[1]


POS_DESCRIPTIONS = {
    "SYM": "Symbol",
    "PUNCT": "Punctuation",
    "ADJ": "Adjective",
    "ADP": "Adposition",
    "ADV": "Adverb",
    "NUM": "Numeral",
    "DET": "Determiner",
    "PRON": "Pronoun",
    "INTJ": "Interjection",
    "CCONJ": "Coordinating Conjunction",
    "SCONJ": "Subordinating Conjunction",
    "NOUN": "Noun",
    "PROPN": "Proper Noun",
    "PART": "Particle",
    "VERB": "Verb",
    "X": "Other",
}

TAG_TO_POS = {
    "$_": "SYM",
    "--": "PUNCT",
    "ADJA": "ADJ",
    "ADJD": "ADJ",
    "ADJN": "ADJ",
    "ADJS": "ADJ",
    "APPR": "ADP",
    "AVD": "ADV",
    "AVD-KO*": "ADV",
    "AVG": "ADV",
    "AVW": "ADV",
    "CARDA": "NUM",
    "CARDD": "NUM",
    "CARDN": "NUM",
    "CARDS": "NUM",
    "DDA": "DET",
    "DDART": "DET",
    "DDD": "DET",
    "DDN": "DET",
    "DDS": "DET",
    "DGA": "DET",
    "DGS": "DET",
    "DIA": "PRON",
    "DIART": "PRON",
    "DID": "PRON",
    "DIN": "PRON",
    "DIS": "PRON",
    "DPOSA": "PRON",
    "DPOSD": "PRON",
    "DPOSN": "PRON",
    "DPOSS": "PRON",
    "DRELS": "PRON",
    "DWA": "PRON",
    "DWD": "PRON",
    "DWS": "PRON",
    "FM": "X",
    "ITJ": "INTJ",
    "KO*": "CCONJ",
    "KOKOM": "CCONJ",
    "KON": "CCONJ",
    "KOUS": "SCONJ",
    "NA": "NOUN",
    "NE": "PROPN",
    "PAVAP": "ADV",
    "PAVD": "ADV",
    "PAVG": "ADV",
    "PAVW": "ADV",
    "PG": "PRON",
    "PI": "PRON",
    "PPER": "PRON",
    "PRF": "PRON",
    "PTK": "PART",
    "PTK*": "PART",
    "PTKA": "PART",
    "PTKANT": "PART",
    "PTKNEG": "PART",
    "PTKVZ": "PART",
    "PW": "PRON",
    "VAFIN": "VERB",
    "VAIMP": "VERB",
    "VAINF": "VERB",
    "VAPP": "VERB",
    "VAPS": "VERB",
    "VMFIN": "VERB",
    "VMIMP": "VERB",
    "VMINF": "VERB",
    "VMPP": "VERB",
    "VV": "VERB",
    "VVFIN": "VERB",
    "VVIMP": "VERB",
    "VVINF": "VERB",
    "VVPP": "VERB",
    "VVPS": "VERB",
}


def load_modern_model():
    try:
        return spacy.load("de_core_news_sm")
    except Exception as e:
        raise RuntimeError(f"Could not load modern German model: {e}")


def load_mhg_model():
    try:
        nlp_mhg = spacy.load(
            mhg_model_path
            / "Spacy-Model-for-Middle-High-German"
            / "models"
            / "model-best"
        )
        nlp_mhg.add_pipe("sentencizer")
        return nlp_mhg
    except Exception as e:
        raise RuntimeError(f"Could not load Middle High German model: {e}")


def pos_tagging_modern(nlp_modern, text):
    doc = nlp_modern(text)
    return doc


def pos_tagging_mhg(nlp_mhg, text):
    if not nlp_mhg:
        raise RuntimeError("Middle High German model is not available.")

    doc = nlp_mhg(text)
    for token in doc:
        token.pos_ = TAG_TO_POS.get(token.tag_, "X")
    return doc


def check_attributes(doc):
    for token in doc:
        print(f"Text: {token.text}, POS: {token.pos_}, TAG: {token.tag_}")
