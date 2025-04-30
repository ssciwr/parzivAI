from streamlit.testing.v1 import AppTest
from unittest import mock
import pytest
from parzivai.app import *

app_script = """
import streamlit as st
from parzivai.app import main
main()
"""


@pytest.fixture
def get_app():
    return AppTest.from_string(app_script)


@mock.patch("parzivai.app.load_mhg_model")
def test_run_app(mock_get_models, get_app):
    mock_get_models.return_value = (mock.Mock(), mock.Mock())
    get_app.run(timeout=800)
    assert not get_app.exception


def test_initialize_session_state():
    st.session_state.clear()

    initialize_session_state()
    assert st.session_state.page == "main"
    assert isinstance(st.session_state.messages, list)
    assert isinstance(st.session_state.state, dict)
    assert isinstance(st.session_state.cached_data, dict)
    assert isinstance(st.session_state.routing_data, list)
    assert isinstance(st.session_state.feedback, list)
    assert isinstance(st.session_state.chat_history, pd.DataFrame)
    assert isinstance(st.session_state.feedback_data, pd.DataFrame)

    assert list(st.session_state.chat_history.columns) == [
        "Chat_Timestamp",
        "Chat_Role",
        "Chat_Message",
    ]
    assert list(st.session_state.feedback_data.columns) == [
        "Feedback_Timestamp",
        "Feedback_Score",
        "Feedback_Comments",
    ]


@mock.patch("parzivai.app.load_embeddings_model")
@mock.patch("parzivai.app.get_vectorstore")
def test_get_cached_retriever(mock_get_vectorstore, mock_load_embeddings_model):
    st.cache_resource.clear()
    mock_embedding = mock.Mock()
    mock_retriever = mock.Mock()
    mock_load_embeddings_model.return_value = mock_embedding
    mock_get_vectorstore.return_value = mock_retriever

    retriever = get_cached_retriever()

    mock_load_embeddings_model.assert_called_once()
    mock_get_vectorstore.assert_called_once_with(mock_embedding)
    assert retriever == mock_retriever


@mock.patch("parzivai.app.load_modern_model")
@mock.patch("parzivai.app.load_mhg_model")
def test_get_models(mock_load_mhg_model, mock_load_modern_model):

    mock_modern = mock.Mock()
    mock_mhg = mock.Mock()
    mock_load_modern_model.return_value = mock_modern
    mock_load_mhg_model.return_value = mock_mhg

    modern, mhg = get_models()

    mock_load_modern_model.assert_called_once()
    mock_load_mhg_model.assert_called_once()
    assert modern == mock_modern
    assert mhg == mock_mhg


@mock.patch("parzivai.app.st")
@mock.patch("parzivai.app.initialize_session_state")
@mock.patch("parzivai.app.get_cached_retriever")
@mock.patch("parzivai.app.get_models")
def test_main_initialization(
    mock_get_models, mock_get_cached_retriever, mock_initialize_session_state, mock_st
):
    from parzivai.app import main

    mock_st.chat_input.return_value = None
    mock_st.tabs.return_value = (mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock())
    mock_get_models.return_value = (mock.Mock(), mock.Mock())
    mock_tab = mock.MagicMock()
    mock_tab.__enter__.return_value = mock_tab
    mock_tab.__exit__.return_value = None
    mock_st.tabs.return_value = [mock_tab, mock_tab, mock_tab, mock_tab]
    main()

    mock_st.tabs.assert_called_once()
    mock_st.title.assert_called_once_with("ParzivAI")
    mock_st.write.assert_called()
    mock_st.sidebar.title.assert_called_once_with("Navigation")
    mock_st.sidebar.image.assert_called_once()

    mock_initialize_session_state.assert_called_once()
    mock_get_cached_retriever.assert_called_once()
    mock_get_models.assert_called_once()


def test_save_chat_to_history_adds_entry():
    mock_session = mock.MagicMock()
    mock_session.chat_history = pd.DataFrame(
        columns=["Chat_Timestamp", "Chat_Role", "Chat_Message"]
    )
    with mock.patch("parzivai.app.st.session_state", mock_session):
        save_chat_to_history("user", "hello")
        assert len(mock_session.chat_history) == 1
        assert mock_session.chat_history.iloc[0]["Chat_Message"] == "hello"


def test_append_to_rendered_messages_user():
    mock_session = mock.MagicMock()
    mock_session.messages = []
    with mock.patch("parzivai.app.st.session_state", mock_session):
        append_to_rendered_messages("user", "hi there")
        assert len(mock_session.messages) == 1
        assert mock_session.messages[0].content == "hi there"


def test_is_translation_request_detects():
    assert is_translation_request("Bitte übersetze das.") is True
    assert is_translation_request("Translate this.") is False


def test_contains_any():
    keywords = ["suicide", "violence"]
    assert contains_any("There is violence.", keywords) is True
    assert contains_any("Hello world!", keywords) is False


def test_build_final_response_message():
    result = {"generation": "This is the answer.", "web_results": "Some search results"}
    msg = build_final_response_message("WebSearch", result)
    assert "🌐" in msg
    assert "Some search results" in msg


def test_retrieve_success():
    mock_retriever = mock.Mock()
    mock_retriever.invoke.return_value = ["doc1", "doc2"]
    result = retrieve("Who is Parzival?", mock_retriever)
    assert result["documents"] == ["doc1", "doc2"]
    assert result["question"] == "Who is Parzival?"


@mock.patch("parzivai.app.append_to_rendered_messages")
@mock.patch("parzivai.app.save_chat_to_history")
def test_save_chat_history_and_messages_combined(mock_save, mock_append):
    save_chat_history_and_messages("assistant", "hello back")
    mock_append.assert_called_once()
    mock_save.assert_called_once()


@mock.patch("parzivai.app.handle_translation")
@mock.patch("parzivai.app.handle_emergency")
@mock.patch("parzivai.app.handle_insult")
@mock.patch("parzivai.app.handle_direct_response")
@mock.patch("parzivai.app.handle_routing_and_answer")
def test_process_user_input_flow(
    mock_routing, mock_direct, mock_insult, mock_emergency, mock_translation
):
    mock_session = mock.MagicMock()

    mock_session.chat_history = pd.DataFrame(
        columns=["Chat_Timestamp", "Chat_Role", "Chat_Message"]
    )
    mock_session.messages = []
    mock_session.state = {}

    with mock.patch("parzivai.app.st.session_state", mock_session):

        process_user_input("Bitte übersetze dies.", retriever=mock.Mock())
        mock_translation.assert_called()

        process_user_input("threat", retriever=mock.Mock())
        mock_emergency.assert_called()

        process_user_input("you idiot", retriever=mock.Mock())
        mock_insult.assert_called()

        process_user_input("who are you", retriever=mock.Mock())
        mock_direct.assert_called()

        process_user_input("Tell me about Parzival.", retriever=mock.Mock())
        mock_routing.assert_called()


def test_extract_content_with_string():
    from parzivai.app import extract_content

    assert extract_content("simple text") == "simple text"


def test_extract_content_with_document():
    from types import SimpleNamespace
    from parzivai.app import extract_content

    doc = SimpleNamespace(page_content="doc text")
    assert extract_content(doc) == "doc text"


@mock.patch("parzivai.app.llm")
def test_grade_document_positive(mock_llm):
    from parzivai.app import grade_document

    mock_llm.invoke.return_value.content = "Yes, definitely."
    assert grade_document("Is this relevant?", "some text") == "yes"


@mock.patch("parzivai.app.llm")
def test_grade_document_negative(mock_llm):
    from parzivai.app import grade_document

    mock_llm.invoke.return_value.content = "No, not really."
    assert grade_document("Is this relevant?", "some text") == "no"


@mock.patch("parzivai.app.llm")
def test_generate_answer(mock_llm):
    from parzivai.app import generate_answer

    mock_llm.invoke.return_value.content = "Generated answer"
    messages = []
    result = generate_answer("question?", ["doc1", "doc2"], messages)
    assert result["question"] == "question?"
    assert "Generated answer" in result["generation"]
    assert len(messages) == 1


@mock.patch("parzivai.app.llm")
def test_llm_fallback_answer(mock_llm):
    from parzivai.app import llm_fallback_answer

    mock_llm.invoke.return_value.content = "Fallback generation"
    result = llm_fallback_answer("fallback question?")
    assert result["question"] == "fallback question?"
    assert "Fallback generation" in result["generation"]


@mock.patch("parzivai.app.retrieve")
@mock.patch("parzivai.app.grade_documents")
def test_decide_route_vectorstore(mock_grade_documents, mock_retrieve):
    from parzivai.app import decide_route

    mock_retrieve.return_value = {"documents": ["doc1"]}
    mock_grade_documents.return_value = {"documents": ["doc1"]}

    result = decide_route("test question", retriever=mock.Mock())
    assert result["route_taken"] == "Vectorstore"
    assert "documents" in result


@mock.patch("parzivai.app.retrieve")
@mock.patch("parzivai.app.grade_documents")
@mock.patch("parzivai.app.web_search")
def test_decide_route_web_search(mock_web_search, mock_grade_documents, mock_retrieve):
    from parzivai.app import decide_route

    mock_retrieve.return_value = {"documents": ["doc1"]}
    mock_grade_documents.return_value = {"documents": []}
    mock_web_search.return_value = {
        "documents": ["web_doc"],
        "web_results": "some results",
    }

    result = decide_route("test question", retriever=mock.Mock())
    assert result["route_taken"] == "WebSearch"
    assert "documents" in result
