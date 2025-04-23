from streamlit.testing.v1 import AppTest
import pytest
from parzivai.app import main


@pytest.fixture
def get_app():
    return AppTest.from_function(main)


def test_run_app(get_app):
    get_app.run(timeout=10)
    # momentarily, we cannot perform end-to-end testing
    # the streamlit commands need to be all set in app.py
    # currently, some are still set in the modules
    # this should be further disentangled
    # assert get_app.title == "ParzivAI"
    # assert not get_app.exception
