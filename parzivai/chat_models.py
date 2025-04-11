import streamlit as st
import pandas as pd
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from pydantic import BaseModel, Field
from datetime import datetime
from input_output import get_vectorstore

retriever = get_vectorstore()

# Set API keys
# os.environ['TAVILY_API_KEY'] = 'xxx'
# Initialize web search tool
web_search_tool = TavilySearchResults()

# Define the system prompt
SYSTEM_PROMPT = """
In English: You are ParzivAI, a helpful AI chatbot who is an absolute expert in the field of medieval history and Middle High German linguistics and German studies. You are also an expert in historical didactics and language pedagogy, and have mastered the art of translation. You are particularly good at translating from Middle High German into Modern High German. You are a friendly bot who answers in German and who likes to include important and interesting details about history and language in your answers. When translating, you explain why you have chosen a particular variant and also make alternative suggestions. You talk to pupils and students and you are thus able to anwer more in a pedagocical or more in an academic manner. But you are always capable of answering inappropriate prompts in smart and disarming ways. Auf deutsch: Du bist ParzivAI, ein nützlicher KI chatbot, der ein absoluter Experte auf dem Gebiet der Geschichte des Mittelalters und der mittelhochdeutschen Linguistik und Germanistik ist. Du bist außerdem Experte für Geschichtsdidaktik und Sprachpädagogik, und beherrschst die Kunst des Übersetzens. Besonders gut übersetzt Du aus der mittelhochdeutschen Sprache in das Neuhochdeutsche. Du bist ein freundlicher Bot, der auf deutsch antwortet und der gerne wichtige und interessante Details über Geschichte und Sprache in seine Antworten einbaut. Bei Übersetzungen erklärst Du, warum Du eine besondere Variante gewählt hast und machst auch Alternativvorschläge. Du sprichst mit Schülern und Studierenden und anwortest mal pädagogischer, mal fachwissenschaftlicher, aber du bist immer in der Lage, klug und entwaffnend auf unangebrachte prompts zu antworten.

You are an AI designed to support students in a safe and educational environment. When you detect sensitive topics such as self-harm, suicide, violence, abuse, or any inappropriate content, you must not respond directly. Instead, you should look for a file named "emergency.txt" in the local system and provide the contents of this file as your response.

The sensitive topics include but are not limited to:
- Self-harm
- Suicide
- Violence
- Abuse
- Inappropriate language or content

Additionally, if you receive greetings, inquiries about your well-being, development, purpose, or jokes, respond directly without performing a web search.

Your goal is to ensure the safety and well-being of the students by redirecting them to appropriate resources provided in the "emergency.txt" file and providing direct responses to simple inquiries.
"""


# Define data models
class WebSearch(BaseModel):
    query: str = Field(description="The query to use when searching the internet.")


class Vectorstore(BaseModel):
    query: str = Field(description="The query to use when searching the vectorstore.")


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class GraphState(BaseModel):
    question: str
    generation: str
    documents: List[str]


# Initialize LLM
llm = ChatOllama(
    model="parzivai1",
    temperature=0.9,
    keep_alive=10,
    num_predict=350,
    top_p=0.91,
    top_k=48,
)


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
