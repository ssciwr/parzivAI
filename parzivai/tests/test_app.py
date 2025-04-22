from streamlit.testing.v1 import AppTest
import pytest

@pytest.fixture
def get_app():
    return AppTest.from_file("app.py")


def test_run_app(get_app):
    get_app.run(timeout=10)
    assert get_app.title == "ParzivAI"
    assert not get_app.exception