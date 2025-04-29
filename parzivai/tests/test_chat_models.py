import pytest
from unittest import mock
import parzivai.chat_models as chat_models
from pathlib import Path

MODEL = "llama3.2:1b"


def test_instantiate_llm():
    llm = chat_models.instantiate_llm(model=MODEL)
    assert llm
    assert llm.model == "llama3.2:1b"
    assert not llm.seed
    # more tests should be added here, for example
    # passing other models, testing exceptions for invalid parameters, etc.


def test_get_emergency_response(monkeypatch):
    emergency_response = chat_models.get_emergency_response()
    assert emergency_response
    assert isinstance(emergency_response, str)
    assert "emergency" in emergency_response.lower()

    monkeypatch.setattr(chat_models, "FILE_PATH", Path("invalid_path"))
    warning = chat_models.get_emergency_response()
    assert "not available" in warning.lower()


def test_get_insult_response(monkeypatch):
    insult_response = chat_models.get_insult_response()
    assert insult_response
    assert isinstance(insult_response, str)
    assert "insult" in insult_response.lower()

    monkeypatch.setattr(chat_models, "FILE_PATH", Path("invalid_path"))
    warning = chat_models.get_insult_response()
    assert "not available" in warning.lower()


@mock.patch("builtins.open", new_callable=mock.mock_open, read_data="Emergency info")
def test_get_emergency_response_success(mock_open):
    result = chat_models.get_emergency_response()
    assert result == "Emergency info"
    mock_open.assert_called_once()


@mock.patch("builtins.open", side_effect=FileNotFoundError)
def test_get_emergency_response_file_not_found(mock_open):
    result = chat_models.get_emergency_response()
    assert result == "Emergency contact information is not available at the moment."
    mock_open.assert_called_once()


@mock.patch("builtins.open", new_callable=mock.mock_open, read_data="Insult info")
def test_get_insult_response_success(mock_open):
    result = chat_models.get_insult_response()
    assert result == "Insult info"
    mock_open.assert_called_once()


@mock.patch("builtins.open", side_effect=FileNotFoundError)
def test_get_insult_response_file_not_found(mock_open):
    result = chat_models.get_insult_response()
    assert result == "Insult response information is not available at the moment."
    mock_open.assert_called_once()


def test_pydantic_models():
    websearch = chat_models.WebSearch(query="mittelalter")
    assert websearch.query == "mittelalter"

    vectorstore = chat_models.Vectorstore(query="ritter")
    assert vectorstore.query == "ritter"
