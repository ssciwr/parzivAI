from streamlit.testing.v1 import AppTest
import pytest
from parzivai.app import main


@pytest.fixture
def get_app():
    return AppTest.from_function(main)


@pytest.fixture
def set_tavily_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test_key")


def test_run_app(get_app, set_tavily_key):
    get_app.run(timeout=10)
    assert get_app.title == "ParzivAI"
    assert not get_app.exception
