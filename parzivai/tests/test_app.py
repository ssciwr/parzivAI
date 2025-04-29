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
