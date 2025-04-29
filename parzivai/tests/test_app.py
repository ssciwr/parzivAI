from streamlit.testing.v1 import AppTest
import pytest


app_script = """
import streamlit as st
from parzivai.app import main
main()
"""


@pytest.fixture
def get_app():
    return AppTest.from_string(app_script)


def test_run_app(get_app):
    get_app.run(timeout=100)
    assert not get_app.exception
