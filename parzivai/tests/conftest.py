import pytest
import os


@pytest.fixture(scope="session", autouse=True)
def set_environment_variables():
    """Set environment variables for all tests."""
    os.environ["TAVILY_API_KEY"] = "test_key"
