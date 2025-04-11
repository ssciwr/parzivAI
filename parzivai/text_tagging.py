import streamlit as st
import pandas as pd
import spacy
from langchain_core.messages import HumanMessage
from datetime import datetime


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

with st.sidebar.expander("POS Tag Descriptions"):
    for pos, desc in POS_DESCRIPTIONS.items():
        st.write(f"{pos}: {desc}")


def append_message_to_history(role, message):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = pd.concat(
            [
                st.session_state.chat_history,
                pd.DataFrame(
                    {
                        "Chat_Timestamp": [datetime.now()],
                        "Chat_Role": [role],
                        "Chat_Message": [message],
                    }
                ),
            ],
            ignore_index=True,
        )


for message in st.session_state["messages"]:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    # content = message.content
    content = message[1]
    avatar = "parzivai.png" if role == "assistant" else None
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
    append_message_to_history(role, content)

####
try:
    nlp_modern = spacy.load("de_core_news_sm")
except Exception as e:
    st.error(f"Could not load modern German model: {e}")
    nlp_modern = None

try:
    nlp_mhg = spacy.load(
        "/mnt/data/rx262/chatbot/Spacy-Model-for-Middle-High-German/models/model-best"
    )
    nlp_mhg.add_pipe("sentencizer")
except Exception as e:
    st.error(f"Could not load Middle High German model: {e}")
    nlp_mhg = None

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


def pos_tagging_modern(text):
    doc = nlp_modern(text)
    return doc


def pos_tagging_mhg(text):
    if nlp_mhg:
        doc = nlp_mhg(text)
        for token in doc:
            token.pos_ = TAG_TO_POS.get(token.tag_, "X")
        return doc
    else:
        st.error("Middle High German model is not available.")
        return None


def check_attributes(doc):
    for token in doc:
        print(f"Text: {token.text}, POS: {token.pos_}, TAG: {token.tag_}")
