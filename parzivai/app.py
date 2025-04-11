import os
import streamlit as st
import pandas as pd
import spacy
import spacy_streamlit
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFDirectoryLoader,
    CSVLoader,
)
from langchain_community.vectorstores import FAISS
# deprecated
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from urllib.parse import quote, urljoin
from pydantic import BaseModel, Field
from streamlit_feedback import streamlit_feedback
import json
import playwright
from playwright.async_api import async_playwright
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
import requests
from bs4 import BeautifulSoup
import asyncio
import torch
from datetime import datetime
from input_output import get_vectorstore
# avoid some torch incompatibility issues with newer Python versions
# see https://github.com/SaiAkhil066/DeepSeek-RAG-Chatbot/issues/4
torch.classes.__path__ = []
# Set API keys
# os.environ['TAVILY_API_KEY'] = 'xxx'

retriever = get_vectorstore()

# Page configuration (must be first Streamlit command)
st.set_page_config(page_title="ParzivAI")

# tab1, tab2, tab3, tab4, tab5 = st.tabs(
tab1, tab2, tab3, tab5 = st.tabs(
    [
        "ParzivAI Chatbot",
        "Bildersuche",
        "Linguistische Analyse",
        # "Lernquiz",
        "User Feedback",
    ]
)

with tab1:
    st.header("ParzivAI Chatbot")

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
