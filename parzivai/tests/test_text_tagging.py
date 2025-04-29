import pytest
from parzivai.text_tagging import (
    load_modern_model,
    load_mhg_model,
    pos_tagging_modern,
    pos_tagging_mhg,
    check_attributes,
    TAG_TO_POS,
)
from unittest.mock import MagicMock, patch


@patch("parzivai.text_tagging.spacy.load")
def test_load_modern_model_success(mock_spacy_load):
    mock_spacy_load.return_value = "mocked_nlp"
    assert load_modern_model() == "mocked_nlp"


@patch("parzivai.text_tagging.spacy.load", side_effect=Exception("load failed"))
def test_load_modern_model_failure(mock_spacy_load):
    with pytest.raises(
        RuntimeError, match="Could not load or download modern German model"
    ):
        load_modern_model()


@patch("parzivai.text_tagging.spacy.load")
def test_load_mhg_model_success(mock_spacy_load):
    mock_model = MagicMock()
    mock_spacy_load.return_value = mock_model
    result = load_mhg_model()
    mock_model.add_pipe.assert_called_with("sentencizer")
    assert result == mock_model


@patch("parzivai.text_tagging.spacy.load", side_effect=Exception("load failed"))
def test_load_mhg_model_failure(mock_spacy_load):
    with pytest.raises(RuntimeError, match="Could not load Middle High German model"):
        load_mhg_model()


def test_pos_tagging_modern():
    fake_nlp = MagicMock()
    fake_doc = "fake_doc"
    fake_nlp.return_value = fake_doc
    result = pos_tagging_modern(fake_nlp, "some text")
    assert result == fake_doc


def test_pos_tagging_mhg_success():
    fake_nlp = MagicMock()
    fake_token = MagicMock(tag_="VVFIN")
    fake_doc = [fake_token]
    fake_nlp.return_value = fake_doc

    doc = pos_tagging_mhg(fake_nlp, "some text")
    assert any(token.pos_ == TAG_TO_POS.get(token.tag_, "X") for token in doc)


def test_pos_tagging_mhg_no_model():
    nlp_none = None
    with pytest.raises(RuntimeError, match="Middle High German model is not available"):
        pos_tagging_mhg(nlp_none, "text")


def test_check_attributes_prints(capfd):
    fake_token = MagicMock(text="word", pos_="NOUN", tag_="NA")
    fake_doc = [fake_token]
    check_attributes(fake_doc)
    captured = capfd.readouterr()
    assert "Text: word, POS: NOUN, TAG: NA" in captured.out
