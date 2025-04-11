import os
import streamlit as st
import pandas as pd
import spacy_streamlit
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from streamlit_feedback import streamlit_feedback
import asyncio
from datetime import datetime
from input_output import get_vectorstore, load_embeddings_model
from image_search import display_images
from text_tagging import check_attributes
import torch
from text_tagging import POS_DESCRIPTIONS
from langchain_core.messages import HumanMessage, AIMessage
from chat_models import instantiate_llm, get_emergency_response, get_insult_response
from chat_models import SENSITIVE_TOPICS, INSULTS, SIMPLE_INQUIRIES
from text_tagging import pos_tagging_mhg, pos_tagging_modern

# avoid some torch incompatibility issues with newer Python versions
# see https://github.com/SaiAkhil066/DeepSeek-RAG-Chatbot/issues/4
torch.classes.__path__ = []
# Page configuration (must be first Streamlit command)

# Set API keys
# os.environ['TAVILY_API_KEY'] = 'xxx'
# Initialize web search tool
web_search_tool = TavilySearchResults()
st.set_page_config(page_title="ParzivAI")
avatar = "parzival.png"
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "ParzivAI Chatbot",
        "Bildersuche",
        "Linguistische Analyse",
        "User Feedback",
    ]
)


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


@st.cache_data(ttl=3600)
def retrieve(question):
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


# Define grading function manually
def grade_document(question: str, document: str) -> str:
    prompt = f"User question: {question}\n\nRetrieved document: {document}\n\nIs this document relevant to the user's question? Please answer 'yes' or 'no'."
    print(f"Grading document with prompt: {prompt}")
    # response = llm.invoke([HumanMessage(content=prompt)])
    response = llm.invoke(("human", prompt))
    print(f"LLM response: {response.content}")
    return (
        "yes"
        if "yes" in response.content.lower() or "ja" in response.content.lower()
        else "no"
    )


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
    messages = [HumanMessage(content=question), AIMessage(content="")]
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


def process_user_input(user_input):
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
        with st.chat_message("assistant", avatar=avatar):
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
        with st.chat_message("assistant", avatar=avatar):
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
            embd = load_embeddings_model()
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
        with st.chat_message("assistant", avatar=avatar):
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


if __name__ == "__main__":
    # Main function to run the Streamlit app
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
    st.sidebar.image("parzival.png", width=150)
    retriever = get_vectorstore()
    llm = instantiate_llm()
    user_input = st.chat_input("Ask ParzivAI a question:")
    if user_input:
        process_user_input(user_input)

    if "cached_data" not in st.session_state:
        st.session_state.cached_data = {
            "embeddings": [],
            "prompts": [],
            "web_results": [],
        }

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

    with st.sidebar.expander("POS Tag Descriptions"):
        for pos, desc in POS_DESCRIPTIONS.items():
            st.write(f"{pos}: {desc}")

    for message in st.session_state["messages"]:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        # content = message.content
        content = message[1]
        avatar = avatar if role == "assistant" else None
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
        append_message_to_history(role, content)

    # Add POS-Tagging buttons dynamically after generating a response
    if st.button("POS-Tagging (Modernes Deutsch)"):
        assistant_response = st.session_state.messages[-1].content
        doc = pos_tagging_modern(assistant_response)
        if doc:
            st.session_state.linguistic_analysis = f("Modernes Deutsch", doc)
            st.experimental_update()  # Ensure the interface updates

    if st.button("POS-Tagging (Mittelhochdeutsch)"):
        assistant_response = st.session_state.messages[-1].content
        doc = pos_tagging_mhg(assistant_response)
        if doc:
            st.session_state.linguistic_analysis = ("Mittelhochdeutsch", doc)
            st.experimental_update()  # Ensure the interface updates

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = pd.DataFrame(
            columns=["Chat_Timestamp", "Chat_Role", "Chat_Message"]
        )

    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = pd.DataFrame(
            columns=["Feedback_Timestamp", "Feedback_Score", "Feedback_Comments"]
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

    with tab1:
        st.header("ParzivAI Chatbot")

    with tab2:
        st.header("Bildersuche")
        topic = st.text_input("Enter a topic for image search:")
        if st.button("Search"):
            st.session_state.image_search_result = topic
            st.experimental_update()

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
