import os
import pytest
import tempfile
import json
from unittest import mock
import langchain_huggingface as lc_hf

from parzivai.input_output import load_config, load_embeddings_model, get_vectorstore


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file."""
    config_path = tmp_path / "urls.json"
    data = {"urls": ["https://de.wikipedia.org/wiki/Mittelhochdeutsche_Sprache"]}
    config_path.write_text(json.dumps(data))
    return config_path


@pytest.fixture
def fake_embedding_model():
    class FakeEmbedder:
        def embed_documents(self, docs):
            return [[0.0] * 768 for _ in docs]

    return FakeEmbedder()


def test_load_config_success(temp_config_file):
    config = load_config(temp_config_file.name)
    assert "urls" in config
    assert isinstance(config["urls"], list)


def test_load_config_file_not_found():
    with pytest.raises(RuntimeError, match="Configuration file not found"):
        load_config("nonexistent.json")


def test_load_embeddings_model_success():
    model = load_embeddings_model()
    assert isinstance(model, lc_hf.HuggingFaceEmbeddings)


@mock.patch("parzivai.input_output.FAISS.load_local")
@mock.patch("parzivai.input_output.os.path.exists", return_value=True)
def test_get_vectorstore_load_existing(
    mock_exists, mock_load_local, fake_embedding_model
):
    """Test getting vectorstore when existing."""
    mock_vs = mock.Mock()
    mock_load_local.return_value = mock_vs
    mock_vs.as_retriever.return_value = "fake_retriever"

    retriever = get_vectorstore(fake_embedding_model)
    assert retriever == "fake_retriever"


@mock.patch("parzivai.input_output.load_documents_and_create_vectorstore")
@mock.patch("parzivai.input_output.os.path.exists", return_value=False)
def test_get_vectorstore_create_new(
    mock_exists, mock_create_vectorstore, fake_embedding_model
):
    """Test getting vectorstore when no index exists."""
    mock_vs = mock.Mock()
    mock_create_vectorstore.return_value = mock_vs
    mock_vs.as_retriever.return_value = "new_retriever"

    retriever = get_vectorstore(fake_embedding_model)
    assert retriever == "new_retriever"
