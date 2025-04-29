import os
from dotenv import load_dotenv
import asyncio
from importlib import resources
import torch
import pandas as pd
from datetime import datetime
import streamlit as st
from streamlit_feedback import streamlit_feedback
import spacy_streamlit
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from parzivai.input_output import get_vectorstore, load_embeddings_model
from parzivai.image_search import fetch_images_for_topic
from parzivai.text_tagging import (
    check_attributes,
    POS_DESCRIPTIONS,
    load_modern_model,
    load_mhg_model,
    pos_tagging_mhg,
    pos_tagging_modern,
)
from parzivai.chat_models import (
    instantiate_llm,
    get_emergency_response,
    get_insult_response,
    SENSITIVE_TOPICS,
    INSULTS,
    SIMPLE_INQUIRIES,
)

# Page configuration (must be first Streamlit command)
st.set_page_config(page_title="ParzivAI")

# avoid some torch incompatibility issues with newer Python versions
# see https://github.com/SaiAkhil066/DeepSeek-RAG-Chatbot/issues/4
torch.classes.__path__ = []


# Add cache approach of getting models here, to make it easier for unit-tests
@st.cache_resource
def get_cached_retriever():
    embedding_model = load_embeddings_model()
    return get_vectorstore(embedding_model)


@st.cache_resource
def get_models():
    return load_modern_model(), load_mhg_model()


# Set API keys
load_dotenv()  # TODO create a .env file in the root directory with TAVILY_API_KEY and delete initialization of TAVILY_API_KEY below
if not os.getenv("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = "xxx"
    st.warning(
        "TAVILY_API_KEY not set in .env or environment. Search may not function properly."
    )
# Initialize web search tool
web_search_tool = TavilySearchResults()
# set data file path
PKG = resources.files("parzivai")
FILE_PATH = PKG / "data"
AVATAR_IMAGE = str(FILE_PATH / "parzival.png")
llm = instantiate_llm()
EMOJI_MAP = {
    "Vectorstore": "📚",
    "WebSearch": "🌐",
    "Fallback": "🤖",
    "useful": "✅",
    "not useful": "❌",
}


def save_chat_to_history(role, message):
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


def append_to_rendered_messages(role, content):
    msg = (
        HumanMessage(content=content) if role == "user" else AIMessage(content=content)
    )
    st.session_state.messages.append(msg)


@st.cache_data(ttl=3600)
def retrieve(question, retriever) -> dict:
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def extract_content(doc_or_str):
    """Safely extract content from Document or fallback to string."""
    return getattr(doc_or_str, "page_content", doc_or_str)


# Define grading function manually
def grade_document(question: str, document: str) -> str:
    prompt = f"User question: {question}\n\nRetrieved document: {document}\n\nIs this document relevant to the user's question? Please answer 'yes' or 'no'."
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"LLM response: {response.content}")
    return (
        "yes"
        if "yes" in response.content.lower() or "ja" in response.content.lower()
        else "no"
    )


def generate_answer(question, documents, messages):
    context = "\n\n".join(extract_content(doc) for doc in documents)
    prompt = f"User question: {question}\n\nContext from documents: {context}\n\nPlease provide a detailed and informative answer based on the context provided."
    messages.append(HumanMessage(content=prompt))
    generation = llm.invoke(messages)

    return {
        "documents": documents,
        "question": question,
        "generation": generation.content,
    }


def llm_fallback_answer(question):
    messages = [HumanMessage(content=question), AIMessage(content="")]
    generation = llm.invoke(messages)
    return {"question": question, "generation": generation.content}


def grade_documents(question, documents):
    filtered_docs = [
        d
        for d in documents
        if grade_document(question, extract_content(d)).lower() == "yes"
    ]
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


def decide_route(question, retriever):
    documents = retrieve(question, retriever)["documents"]
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


def grade_generation_v_documents_and_question(question, generation):
    score = grade_document(question, generation)
    return "useful" if score == "yes" else "not useful"


def initialize_session_state():
    st.session_state.page = "main"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.state = {
        "question": "",
        "documents": [],
        "messages": [],
        "route_taken": "",
    }
    st.session_state.cached_data = {"embeddings": [], "prompts": [], "web_results": []}
    st.session_state.routing_data = []
    st.session_state.feedback = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = pd.DataFrame(
            {
                "Chat_Timestamp": pd.Series(dtype="datetime64[ns]"),
                "Chat_Role": pd.Series(dtype="str"),
                "Chat_Message": pd.Series(dtype="str"),
            }
        )
    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = pd.DataFrame(
            {
                "Feedback_Timestamp": pd.Series(dtype="datetime64[ns]"),
                "Feedback_Score": pd.Series(dtype="int"),
                "Feedback_Comments": pd.Series(dtype="str"),
            }
        )


def save_chat_history_and_messages(role: str, message: str):
    append_to_rendered_messages(role, message)
    save_chat_to_history(role, message)


def process_user_input(user_input, retriever):
    save_chat_history_and_messages("User", user_input)
    st.session_state.state["question"] = user_input
    st.session_state.state["messages"] = st.session_state.messages

    if is_translation_request(user_input):
        handle_translation(user_input)
    elif contains_any(user_input, SENSITIVE_TOPICS["sensitive_topics"]):
        handle_emergency()
    elif contains_any(user_input, INSULTS["insults"]):
        handle_insult()
    elif contains_any(user_input, SIMPLE_INQUIRIES["simple_inquiries"]):
        handle_direct_response(user_input)
    else:
        handle_routing_and_answer(user_input, retriever)


def is_translation_request(text: str) -> bool:
    return "übersetze" in text.lower()


def contains_any(text: str, keywords: list) -> bool:
    return any(k.lower() in text.lower() for k in keywords)


def handle_translation(user_input: str):
    save_chat_history_and_messages(
        "Assistant",
        "🔄 Übersetzung angefordert - Antwort wird direkt durch ParzivAI generiert",
    )
    response = llm.invoke([HumanMessage(content=user_input)])
    save_chat_history_and_messages("Assistant", response.content)


def handle_emergency():
    save_chat_history_and_messages(
        "Assistant", "⚠️ Trigger word detected - Providing emergency information."
    )
    emergency_info = get_emergency_response()
    save_chat_history_and_messages("Assistant", emergency_info)


def handle_insult():
    save_chat_history_and_messages(
        "Assistant", "🚫 Insult detected - Providing a response to insults."
    )
    response = get_insult_response()
    save_chat_history_and_messages("Assistant", response)


def handle_direct_response(user_input: str):
    save_chat_history_and_messages(
        "Assistant", "💬 Direct response requested - Generating answer directly."
    )
    response = llm.invoke([HumanMessage(content=user_input)])
    save_chat_history_and_messages("Assistant", response.content)


def handle_routing_and_answer(user_input: str, retriever):
    routing_info = decide_route(user_input, retriever)
    st.session_state.state.update(routing_info)

    if routing_info["route_taken"] == "Vectorstore":
        save_chat_history_and_messages(
            "Assistant",
            "📚 Relevant documents found. Using them for answer generation.",
        )
    else:
        save_chat_history_and_messages(
            "Assistant", "🌐 No relevant documents found. Using web search results."
        )

    result = generate_answer(
        st.session_state.state["question"],
        st.session_state.state["documents"],
        st.session_state.state["messages"],
    )
    st.session_state.state["generation"] = result

    update_embedding_cache(user_input)

    assistant_message = build_final_response_message(
        routing_info["route_taken"], result
    )
    save_chat_history_and_messages("Assistant", assistant_message)

    if "web_results" in routing_info:
        st.session_state.cached_data["web_results"].append(routing_info["web_results"])

    st.session_state.routing_data.append(
        {
            "question": user_input,
            "route_taken": routing_info["route_taken"],
            "filtered_documents_count": len(st.session_state.state["documents"]),
        }
    )


def update_embedding_cache(user_input: str):
    docs = st.session_state.state.get("documents", [])
    texts = [extract_content(doc) for doc in docs]
    if texts:
        embeddings = load_embeddings_model().embed_documents(texts)
        st.session_state.cached_data["embeddings"].extend(embeddings)
    st.session_state.cached_data["prompts"].append(user_input)


def build_final_response_message(route: str, result: dict) -> str:
    emoji = EMOJI_MAP.get(route, "🤖")
    message = f"{emoji} {result['generation']}"
    if "web_results" in result:
        message += f"\n\n**Web Search Results**:\n{result['web_results']}"
    message += f"\n\nRoute taken: {route}"
    return message


def show_pos_tagging_options(latest_response: str, nlp_modern, nlp_mhg):
    st.markdown("### POS-Tagging Options")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("POS-Tagging (Modernes Deutsch)"):
            doc = pos_tagging_modern(nlp_modern, latest_response)
            if doc:
                st.session_state.linguistic_analysis = ("Modernes Deutsch", doc)
                st.rerun()
    with col2:
        if st.button("POS-Tagging (Mittelhochdeutsch)"):
            doc = pos_tagging_mhg(nlp_mhg, latest_response)
            if doc:
                st.session_state.linguistic_analysis = ("Mittelhochdeutsch", doc)
                st.rerun()


def main():
    # Main function to run the Streamlit app
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "ParzivAI Chatbot",
            "Bildersuche",
            "Linguistische Analyse",
            "User Feedback",
        ]
    )
    # Apply custom CSS
    st.markdown(
        """
    <style>
    div.stButton > button:first-child {
        display: block;
        margin: 0 auto;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.title("ParzivAI")
    st.write(
        "ParzivAI helps you with questions about the Middle Ages and Middle High German language and literature."
    )
    st.sidebar.title("Navigation")
    st.sidebar.image(AVATAR_IMAGE, width=150)
    # function to initialize all session state variables
    initialize_session_state()
    retriever = get_cached_retriever()
    nlp_modern, nlp_mhg = get_models()

    user_input = st.chat_input("Ask ParzivAI a question:")
    if user_input:
        process_user_input(user_input, retriever)

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

    with st.sidebar.expander("POS Tag Descriptions"):
        for pos, desc in POS_DESCRIPTIONS.items():
            st.write(f"{pos}: {desc}")

    for message in st.session_state["messages"]:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        content = message.content
        avatar_icon = (
            AVATAR_IMAGE if role == "assistant" else None
        )  # Use a different variable name
        with st.chat_message(role, avatar=avatar_icon):
            st.markdown(content)

    # Add POS-Tagging buttons dynamically after generating a response
    assistant_response = next(
        (
            msg.content
            for msg in reversed(st.session_state.messages)
            if isinstance(msg, AIMessage)
        ),
        None,
    )
    if assistant_response:
        show_pos_tagging_options(assistant_response, nlp_modern, nlp_mhg)

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
        combined_data = pd.concat(
            [st.session_state.chat_history, st.session_state.feedback_data],
            ignore_index=True,
        )

        # Debugging step to print combined data
        st.write("Combined data before saving:")
        st.write(combined_data)

        # Save to CSV

        feedback_file = FILE_PATH / "pages/feedback_combined.csv"
        os.makedirs(feedback_file.parent, exist_ok=True)

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

    with tab1:
        st.header("ParzivAI Chatbot")

    with tab2:
        st.header("Bildersuche")
        with st.form("image_search_form"):
            topic = st.text_input("Enter a topic for image search:")
            search_submitted = st.form_submit_button("Search")

        if search_submitted:
            if topic.strip():
                st.session_state.image_search_result = topic.strip()
            else:
                st.warning("Please enter a topic to search for images.")

        if "image_search_result" in st.session_state:
            st.write("Searching for images...")
            image_data = asyncio.run(
                fetch_images_for_topic(st.session_state.image_search_result)
            )

            for data in image_data:
                st.image(
                    data["url"],
                    caption=f"Bildthema: {data['name']}, Archivnummer: {data['archiveNumber']}, URL: {data['url']}",
                    use_container_width=True,
                )

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

    # Display feedback tab content only in the feedback tab
    with tab4:
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


if __name__ == "__main__":
    # Run the main function
    main()
