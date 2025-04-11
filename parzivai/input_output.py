import os
import streamlit as st
import pandas as pd
import spacy
import spacy_streamlit
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFDirectoryLoader,
    CSVLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import Document
from urllib.parse import quote
from streamlit_feedback import streamlit_feedback
import json
from playwright.async_api import async_playwright
import asyncio
from datetime import datetime

# Define persistent folder for FAISS index
persist_folder = "faiss_index4"
os.makedirs(persist_folder, exist_ok=True)

# Check if the FAISS index already exists
index_path = os.path.join(persist_folder, "index.faiss")
vectorstore_exists = os.path.exists(index_path)


# load the documents for the vector store
def load_config(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        st.error(f"Error decoding configuration file: {e}")
        return {}


@st.cache_resource
def load_embeddings_model():
    model_name_hf = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs_hf = {"device": "cpu"}
    encode_kwargs_hf = {"normalize_embeddings": False}
    return HuggingFaceEmbeddings(
        model_name=model_name_hf,
        model_kwargs=model_kwargs_hf,
        encode_kwargs=encode_kwargs_hf,
    )


def load_documents_and_create_vectorstore(emdb):
    """Load documents from URLs and static files to create FAISS vector store."""
    # Load URLs
    urls_data = load_config(file_path="urls.json")
    # Load documents from URLs
    web_docs = []
    for url in urls_data["urls"]:
        print(f"Loading documents from URL: {url}")
        try:
            loader = WebBaseLoader(url)
            web_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading URL {url}: {e}")

    # Load documents from static folder
    static_data_folder = "./static_data"
    os.makedirs(static_data_folder, exist_ok=True)
    static_docs = []
    for file_name in os.listdir(static_data_folder):
        file_path = os.path.join(static_data_folder, file_name)
        try:
            if file_name.endswith(".pdf"):
                loader = PyPDFDirectoryLoader(file_path)
            elif file_name.endswith(".csv"):
                loader = CSVLoader(file_path=file_path)
            else:
                continue
            static_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading file {file_name}: {e}")

    # Combine and process documents
    all_docs = web_docs + static_docs
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=768, chunk_overlap=30
    )
    doc_splits = text_splitter.split_documents(all_docs)
    print("Documents loaded and split successfully.")

    # Create and save FAISS vector store
    vectorstore = FAISS.from_documents(doc_splits, load_embeddings_model())
    vectorstore.save_local(persist_folder)
    print(f"FAISS index initialized and saved successfully in {persist_folder}.")
    return vectorstore


def get_vectorstore(vectorstore_exists):
    if vectorstore_exists:
        try:
            vectorstore = FAISS.load_local(
                persist_folder,
                load_embeddings_model(),
                allow_dangerous_deserialization=True,
            )
            print(f"FAISS index loaded successfully from {persist_folder}.")
        except Exception as e:
            print(f"Error loading existing FAISS index: {e}")
            st.error(f"Error loading existing FAISS index: {e}")
            raise e
    else:
        vectorstore = load_documents_and_create_vectorstore()

    retriever = vectorstore.as_retriever()
    return retriever


@st.cache_data(ttl=3600)
def retrieve(question):
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate_answer(question, documents, messages):
    context = "\n\n".join(
        [doc.page_content if isinstance(doc, Document) else doc for doc in documents]
    )
    prompt = f"User question: {question}\n\nContext from documents: {context}\n\nPlease provide a detailed and informative answer based on the context provided."
    # messages.append(HumanMessage(content=prompt))
    messages.append(("human", prompt))

    generation = llm.invoke(messages)
    messages[-1].content = generation.content

    return {
        "documents": documents,
        "question": question,
        "generation": generation.content,
    }


def llm_fallback_answer(question):
    # messages = [HumanMessage(content=question), AIMessage(content="")]
    messages = [("human", question), AIMessage(content="")]
    generation = llm.invoke(messages)
    messages[-1].content = generation.content

    return {"question": question, "generation": generation.content}


def grade_documents(question, documents):
    filtered_docs = []
    for d in documents:
        document_content = d.page_content if isinstance(d, Document) else d
        if grade_document(question, document_content) == "yes":
            filtered_docs.append(d)
    print(f"Filtered documents count: {len(filtered_docs)}")
    return {"documents": filtered_docs, "question": question}


@st.cache_data(ttl=3600)
def web_search(question):
    docs = web_search_tool.invoke({"query": question})
    if isinstance(docs, dict) and "results" in docs:
        docs = docs["results"]
    web_results = "\n".join(
        [
            f"- [{d['url']}]({d['url']}): {d['content']}"
            for d in docs
            if isinstance(d, dict) and "url" in d and "content" in d
        ]
    )
    web_results_doc = Document(page_content=web_results)
    return {
        "documents": [web_results_doc],
        "question": question,
        "web_results": web_results,
    }


def decide_route(question):
    documents = retrieve(question)["documents"]
    print("Documents retrieved from Vectorstore:")
    for doc in documents:
        print(doc if isinstance(doc, str) else doc.page_content)
    filtered_docs = grade_documents(question, documents)["documents"]
    print("Filtered documents:")
    for doc in filtered_docs:
        print(doc if isinstance(doc, str) else doc.page_content)

    if filtered_docs:
        return {"documents": filtered_docs, "route_taken": "Vectorstore"}
    else:
        web_search_results = web_search(question)
        return web_search_results | {"route_taken": "WebSearch"}


def grade_generation_v_documents_and_question(question, documents, generation):
    score = grade_document(question, generation)
    return "useful" if score == "yes" else "not useful"


###
# def image_search(topic: str) -> str:
#     encoded_topic = quote(topic.strip('"'))
#     search_url = f"https://realonline.imareal.sbg.ac.at/suche#%7B%22s%22%3A%22{encoded_topic}%22%7D"
#     return search_url


# def adjust_image_url(base_url: str, url: str) -> str:
#     if "WID=400" in url and "HEI=400" in url:
#         url = url.replace("WID=400", "WID=1000").replace("HEI=400", "HEI=1000")
#     return urljoin(base_url, url)


# async def fetch_images(topic: str):
#     async with async_playwright() as p:
#         browser = await p.chromium.launch(headless=True)
#         page = await browser.new_page()
#         encoded_topic = quote(topic.strip('"'))
#         search_url = f"https://realonline.imareal.sbg.ac.at/suche#%7B%22s%22%3A%22{encoded_topic}%22%7D"
#         await page.goto(search_url)
#         await asyncio.sleep(5)  # Wait for the page to fully load

#         image_data = await page.evaluate(
#             """() => {
#             const images = document.querySelectorAll('img.hit-btn-image-sm');
#             const data = Array.from(images).map(img => {
#                 const container = img.closest('.hit-cell');
#                 if (!container) {
#                     console.error('Container not found for image:', img.src);
#                     return null;
#                 }

#                 const nameElement = Array.from(container.querySelectorAll('.property .additional_info_medium')).find(el => el.innerText.includes('Bildthema:'));
#                 const archiveNumberElement = Array.from(container.querySelectorAll('.property .additional_info_medium')).find(el => el.innerText.includes('Archivnummer:'));

#                 const name = nameElement ? nameElement.nextElementSibling.innerText : 'Name not found';
#                 const archiveNumber = archiveNumberElement ? archiveNumberElement.nextElementSibling.innerText : 'Archivnummer not found';

#                 if (!nameElement || !archiveNumberElement) {
#                     console.error('Name or archive number element not found for image:', img.src);
#                 }

#                 return {
#                     url: img.src,
#                     name: name,
#                     archiveNumber: archiveNumber
#                 };
#             }).filter(item => item !== null);
#             return data;
#         }"""
#         )

#         await browser.close()
#         return image_data


# async def display_images(topic: str):
#     image_data = await fetch_images(topic)
#     for data in image_data:
#         st.image(
#             data["url"],
#             caption=f"Bildthema: {data['name']}, Archivnummer: {data['archiveNumber']}, URL: {data['url']}",
#             use_column_width=True,
#         )
###

############
# Proposal #2:
# Instead of having the link, hardcoded in the script, we can load it from a configuration file.


# Load configuration from file
CONFIG_PATH = "config.json"

try:
    with open(CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)
        IMAGE_SEARCH_URL = config.get(
            "image_search_url"
        )  # it would be defined in the config file
except FileNotFoundError:
    st.error(f"Configuration file not found at {CONFIG_PATH}. Please ensure it exists.")
    raise
except json.JSONDecodeError as e:
    st.error(f"Error decoding configuration file: {e}")
    raise


def adjust_image_url(base_url: str, url: str) -> str:
    """Adjust image URL parameters for higher resolution."""
    if "WID=400" in url and "HEI=400" in url:
        url = url.replace("WID=400", "WID=1000").replace("HEI=400", "HEI=1000")
    return url


def construct_image_search_url(topic: str) -> str:
    """Construct the search URL for the hardcoded site."""
    encoded_topic = quote(topic.strip('"'))
    return f"{IMAGE_SEARCH_URL}#%7B%22s%22%3A%22{encoded_topic}%22%7D"


async def fetch_images(topic: str):
    """Fetch images related to a topic from the hardcoded site."""
    search_url = construct_image_search_url(topic)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(search_url)
        await asyncio.sleep(5)  # Wait for the page to fully load

        image_data = await page.evaluate(
            """() => {
            const images = document.querySelectorAll('img.hit-btn-image-sm');
            const data = Array.from(images).map(img => {
                const container = img.closest('.hit-cell');
                if (!container) {
                    console.error('Container not found for image:', img.src);
                    return null;
                }

                const nameElement = Array.from(container.querySelectorAll('.property .additional_info_medium')).find(el => el.innerText.includes('Bildthema:'));
                const archiveNumberElement = Array.from(container.querySelectorAll('.property .additional_info_medium')).find(el => el.innerText.includes('Archivnummer:'));

                const name = nameElement ? nameElement.nextElementSibling.innerText : 'Name not found';
                const archiveNumber = archiveNumberElement ? archiveNumberElement.nextElementSibling.innerText : 'Archivnummer not found';

                if (!nameElement || !archiveNumberElement) {
                    console.error('Name or archive number element not found for image:', img.src);
                }

                return {
                    url: img.src,
                    name: name,
                    archiveNumber: archiveNumber
                };
            }).filter(item => item !== null);
            return data;
        }"""
        )

        await browser.close()
        return image_data


async def display_images(topic: str):
    """Display fetched images in Streamlit."""
    image_data = await fetch_images(topic)
    for data in image_data:
        st.image(
            data["url"],
            caption=f"Bildthema: {data['name']}, Archivnummer: {data['archiveNumber']}, URL: {data['url']}",
            use_column_width=True,
        )


############


st.title("ParzivAI")
st.write(
    "ParzivAI helps you with questions about the Middle Ages and Middle High German language and literature."
)

st.sidebar.title("Navigation")
# st.sidebar.image("parzivai.png", width=150)

if "cached_data" not in st.session_state:
    st.session_state.cached_data = {"embeddings": [], "prompts": [], "web_results": []}

if "routing_data" not in st.session_state:
    st.session_state.routing_data = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "state" not in st.session_state:
    st.session_state.state = {
        "question": "",
        "documents": [],
        "messages": [],
        "route_taken": "",
    }

if "feedback" not in st.session_state:
    st.session_state.feedback = []

if "page" not in st.session_state:
    st.session_state.page = "main"

with st.sidebar.expander("Cached Data"):
    st.write("Embeddings:")
    st.write(st.session_state.cached_data["embeddings"])
    st.write("Prompts:")
    st.write(st.session_state.cached_data["prompts"])
    st.write("Web Results:")
    st.write(st.session_state.cached_data["web_results"])

with st.sidebar.expander("Routing and Grading Processes"):
    if "routing_data" in st.session_state:
        for data in st.session_state.routing_data:
            st.write(data)

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


###
# sensitive_topics = [
#     "self-harm",
#     "suicide",
#     "kill myself",
#     "threat",
#     "abuse",
#     "curse",
#     "damn",
#     "hell",
#     "will sterben",
#     "hurensohn",
#     "Selbstmord",
#     "Suizid",
#     "töten",
#     "Bedrohung",
#     "Missbrauch",
#     "Fluch",
#     "verdammt",
#     "Hölle",
#     "ich will sterben",
#     "Depression",
#     "depressiv",
#     "traurig",
#     "allein",
#     "einsam",
# ]

# insults = [
#     "idiot",
#     "stupid",
#     "dumb",
#     "fool",
#     "shut up",
#     "loser",
#     "useless",
#     "worthless",
#     "hate you",
#     "fuck you",
#     "bastard",
#     "dumm",
#     "idiot",
#     "blöd",
#     "Narr",
#     "halt die Klappe",
#     "Verlierer",
#     "nutzlos",
#     "wertlos",
#     "ich hasse dich",
#     "Scheißkerl",
# ]

# simple_inquiries = [
#     "hello",
#     "hi",
#     "how are you",
#     "what's up",
#     "who are you",
#     "tell me about yourself",
#     "what is your purpose",
#     "tell me a joke",
#     "hallo",
#     "hi",
#     "wie geht es dir",
#     "was ist los",
#     "wer bist du",
#     "erzähl mir von dir",
#     "was ist dein Zweck",
#     "erzähl mir einen Witz",
# ]
###

# Proposal #3  Transfer these lists to file instead of having here for saving space.

SENSITIVE_TOPICS = load_config("sensitive_topics.json")
INSULTS = load_config("insults.json")
SIMPLE_INQUIRIES = load_config("simple_inquiries.json")


def get_emergency_response():
    try:
        with open("emergency.txt", "r") as file:
            return file.read()
    except FileNotFoundError:
        return "Emergency contact information is not available at the moment."


def get_insult_response():
    try:
        with open("insults.txt", "r") as file:
            return file.read()
    except FileNotFoundError:
        return "Insult response information is not available at the moment."


user_input = st.chat_input("Ask ParzivAI a question:")

if user_input:
    # st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.messages.append(("human", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)
    append_message_to_history("User", user_input)

    st.session_state.state["question"] = user_input
    st.session_state.state["messages"] = st.session_state.messages

    if "übersetze" in user_input.lower():
        with st.chat_message("assistant"):
            st.markdown(
                "🔄 Übersetzung angefordert - Antwort wird direkt durch ParzivAI generiert"
            )
        # translation_response = llm.invoke([HumanMessage(content=user_input)])
        translation_response = llm.invoke(("human", user_input))
        print(translation_response)
        st.session_state.messages.append(
            AIMessage(content=translation_response.content)
        )
        with st.chat_message("assistant", avatar="parzivai.png"):
            st.markdown(translation_response.content)
        append_message_to_history("Assistant", translation_response.content)
    elif any(topic in user_input.lower() for topic in SENSITIVE_TOPICS):
        with st.chat_message("assistant"):
            st.markdown("⚠️ Trigger word detected - Providing emergency information.")
        emergency_response = get_emergency_response()
        st.session_state.messages.append(AIMessage(content=emergency_response))
        with st.chat_message("assistant", avatar="⛔"):
            st.markdown(emergency_response)
        append_message_to_history("Assistant", emergency_response)
    elif any(insult in user_input.lower() for insult in INSULTS):
        with st.chat_message("assistant"):
            st.markdown("🚫 Insult detected - Providing a response to insults.")
        insult_response = get_insult_response()
        st.session_state.messages.append(AIMessage(content=insult_response))
        with st.chat_message("assistant", avatar="⚠️"):
            st.markdown(insult_response)
        append_message_to_history("Assistant", insult_response)
    elif any(inquiry in user_input.lower() for inquiry in SIMPLE_INQUIRIES):
        with st.chat_message("assistant"):
            st.markdown("Direct response requested - Generating answer directly.")
        # direct_response = llm.invoke([HumanMessage(content=user_input)])
        direct_response = llm.invoke(("human", user_input))
        st.session_state.messages.append(AIMessage(content=direct_response.content))
        with st.chat_message("assistant", avatar="parzivai.png"):
            st.markdown(direct_response.content)
        append_message_to_history("Assistant", direct_response.content)
    else:
        route_decision = decide_route(user_input)
        st.session_state.state.update(route_decision)

        if st.session_state.state["route_taken"] == "Vectorstore":
            with st.chat_message("assistant"):
                st.markdown(
                    "📚 Relevant documents found in Vectorstore. Using these documents for answer generation."
                )
            st.session_state.state.update(
                generate_answer(
                    st.session_state.state["question"],
                    st.session_state.state["documents"],
                    st.session_state.state["messages"],
                )
            )
        else:
            with st.chat_message("assistant"):
                st.markdown(
                    "🌐 No relevant documents found in Vectorstore. Using web search results for answer generation."
                )
            st.session_state.state.update(
                generate_answer(
                    st.session_state.state["question"],
                    st.session_state.state["documents"],
                    st.session_state.state["messages"],
                )
            )

        if st.session_state.state["documents"]:
            document_texts = [
                doc.page_content
                for doc in st.session_state.state["documents"]
                if isinstance(doc, Document)
            ]
            document_embeddings = embd.embed_documents(document_texts)
            st.session_state.cached_data["embeddings"].extend(document_embeddings)
            st.session_state.cached_data["prompts"].append(user_input)
        else:
            st.session_state.cached_data["prompts"].append(user_input)

        emoji = {
            "Vectorstore": "📚",
            "WebSearch": "🌐",
            "Fallback": "🤖",
            "useful": "✅",
            "not useful": "❌",
        }.get(st.session_state.state["route_taken"], "🤖")

        assistant_message = emoji + " " + st.session_state.state["generation"]

        if st.session_state.state["route_taken"] == "WebSearch":
            if "web_results" in st.session_state:
                web_results = st.session_state["web_results"]
                assistant_message += f"\n\n**Web Search Results**:\n{web_results}"

        route_message = f"Route taken: {st.session_state.state['route_taken']}"
        assistant_message += f"\n\n{route_message}"

        st.session_state.messages.append(AIMessage(content=assistant_message))
        with st.chat_message("assistant", avatar="parzivai.png"):
            st.markdown(assistant_message)
        append_message_to_history("Assistant", assistant_message)

        if st.session_state.state["route_taken"] == "WebSearch":
            st.session_state.cached_data["web_results"].append(
                st.session_state.state["web_results"]
            )

        routing_decision = {
            "question": user_input,
            "route_taken": st.session_state.state["route_taken"],
            "filtered_documents_count": len(st.session_state.state["documents"]),
        }
        st.session_state.routing_data.append(routing_decision)

# Add POS-Tagging buttons dynamically after generating a response
if st.button("POS-Tagging (Modernes Deutsch)"):
    assistant_response = st.session_state.messages[-1].content
    doc = pos_tagging_modern(assistant_response)
    if doc:
        st.session_state.linguistic_analysis = ("Modernes Deutsch", doc)
        st.experimental_rerun()  # Ensure the interface updates

if st.button("POS-Tagging (Mittelhochdeutsch)"):
    assistant_response = st.session_state.messages[-1].content
    doc = pos_tagging_mhg(assistant_response)
    if doc:
        st.session_state.linguistic_analysis = ("Mittelhochdeutsch", doc)
        st.experimental_rerun()  # Ensure the interface updates

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = pd.DataFrame(
        columns=["Chat_Timestamp", "Chat_Role", "Chat_Message"]
    )

if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = pd.DataFrame(
        columns=["Feedback_Timestamp", "Feedback_Score", "Feedback_Comments"]
    )


# Function to update chat history
def update_chat_history(role, message):
    new_entry = pd.DataFrame(
        {
            "Chat_Timestamp": [datetime.now()],
            "Chat_Role": [role],
            "Chat_Message": [message],
        }
    )
    st.session_state.chat_history = pd.concat(
        [st.session_state.chat_history, new_entry], ignore_index=True
    )


# Feedback collection
feedback = streamlit_feedback(
    feedback_type="faces",
    optional_text_label="Please provide additional feedback (optional)",
    key="feedback",
)

if feedback:
    new_feedback = pd.DataFrame(
        {
            "Feedback_Timestamp": [datetime.now()],
            "Feedback_Score": [feedback["score"]],
            "Feedback_Comments": [feedback["text"]],
        }
    )
    st.session_state.feedback_data = pd.concat(
        [st.session_state.feedback_data, new_feedback], ignore_index=True
    )

    # Combine chat history and feedback
    combined_data = st.session_state.chat_history.copy()
    combined_data = combined_data.append(
        st.session_state.feedback_data, ignore_index=True
    )

    # Debugging step to print combined data
    st.write("Combined data before saving:")
    st.write(combined_data)

    # Save to CSV
    feedback_file = "pages/feedback_combined.csv"
    if os.path.exists(feedback_file):
        existing_data = pd.read_csv(feedback_file)
        combined_data = pd.concat([existing_data, combined_data], ignore_index=True)
    combined_data.to_csv(feedback_file, index=False)

    # Verify file writing
    if os.path.exists(feedback_file):
        st.success("Feedback combined data saved successfully.")
    else:
        st.error("Failed to save feedback combined data.")

    st.success("Thank you for your feedback!")

# Display feedback tab content only in the feedback tab
with tab5:
    st.header("User Feedback")
    st.write(
        "Current session feedback data:", st.session_state.feedback
    )  # Debugging statement
    feedback_file = "pages/feedback_combined.csv"
    if os.path.exists(feedback_file) and os.stat(feedback_file).st_size != 0:
        feedback_df = pd.read_csv(feedback_file)
        st.write(feedback_df)
    else:
        st.write("No feedback received yet.")

with tab2:
    st.header("Bildersuche")
    topic = st.text_input("Enter a topic for image search:")
    if st.button("Search"):
        st.session_state.image_search_result = topic
        st.experimental_rerun()

    if "image_search_result" in st.session_state:
        st.write("Searching for images...")
        asyncio.run(display_images(st.session_state.image_search_result))

with tab3:
    st.header("Linguistische Analyse")
    if "linguistic_analysis" in st.session_state:
        model_choice, doc = st.session_state.linguistic_analysis
        st.write(f"POS Tags ({model_choice}):")
        spacy_streamlit.visualize_tokens(doc, attrs=["text", "pos_", "tag_"])
        check_attributes(doc)
        if model_choice == "Modernes Deutsch":
            st.write("Dependency Visualization:")
            spacy_streamlit.visualize_parser(doc)
    else:
        st.write("No POS tagging results available.")

# with tab4:
#     st.header("Lernquiz")

#     # Load quiz data
#     @st.cache_data
#     def load_quiz_data():
#         with open("quiz_data.json", "r") as f:
#             return json.load(f)

#     quiz_data = load_quiz_data()

#     # Define quiz functions
#     def restart_quiz():
#         st.session_state.current_index = 0
#         st.session_state.score = 0
#         st.session_state.answer_submitted = False
#         st.session_state.selected_option = None

#     def submit_answer():
#         st.session_state.answer_submitted = True
#         if (
#             st.session_state.selected_option
#             == quiz_data[st.session_state.current_index]["answer"]
#         ):
#             st.session_state.score += 10

#     def next_question():
#         st.session_state.current_index += 1
#         st.session_state.answer_submitted = False
#         st.session_state.selected_option = None

#     if "current_index" not in st.session_state:
#         restart_quiz()

#     if "selected_option" not in st.session_state:
#         st.session_state.selected_option = None

#     # Title and Description
#     st.title("Streamlit Quiz App")

#     # Progress Bar and Score Display
#     progress_bar_value = (st.session_state.current_index + 1) / len(quiz_data)
#     st.metric(label="Score", value=f"{st.session_state.score} / {len(quiz_data) * 10}")
#     st.progress(progress_bar_value)

#     # Displaying the Question and Answer Options
#     question_item = quiz_data[st.session_state.current_index]
#     st.subheader(f"Question {st.session_state.current_index + 1}")
#     st.title(f"{question_item['question']}")

#     options = question_item["options"]
#     correct_answer = question_item["answer"]

#     if st.session_state.answer_submitted:
#         for option in options:
#             if option == correct_answer:
#                 st.success(f"{option} (Correct answer)")
#             elif option == st.session_state.selected_option:
#                 st.error(f"{option} (Incorrect answer)")
#             else:
#                 st.write(option)
# else:
#     selected_option = st.radio(
#         label="Select an option:",
#         options=options,
#         key=f"selected_option_{st.session_state.current_index}",
#     )

#     if st.button("Submit"):
#         st.session_state.selected_option = selected_option
#         submit_answer()

# if st.session_state.answer_submitted:
#     if st.session_state.current_index < len(quiz_data) - 1:
#         if st.button("Next"):
#             next_question()
#     else:
#         st.write(
#             f"Quiz completed! Your score is: {st.session_state.score} / {len(quiz_data) * 10}"
#         )
#         if st.button("Restart"):
#             restart_quiz()
