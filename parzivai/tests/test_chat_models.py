import pytest
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
    # modify file path using monkeypatch
    monkeypatch.setattr(chat_models, "FILE_PATH", Path("invalid_path"))
    warning = chat_models.get_emergency_response()
    assert "not available" in warning.lower()


def test_get_insult_response(monkeypatch):
    insult_response = chat_models.get_insult_response()
    assert insult_response
    assert isinstance(insult_response, str)
    assert "insult" in insult_response.lower()
    # modify file path using monkeypatch
    monkeypatch.setattr(chat_models, "FILE_PATH", Path("invalid_path"))
    warning = chat_models.get_insult_response()
    assert "not available" in warning.lower()
