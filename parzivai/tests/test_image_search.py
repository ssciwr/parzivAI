import pytest
from parzivai.image_search import (
    adjust_image_url,
    construct_image_search_url,
)

IMAGE_SEARCH_URL = "https://de.wikipedia.org/wiki/Mittelhochdeutsche_Sprache#/media/Datei:Inschrift_Grab_Ulrich_III_Liechtenstein.jpg"


@pytest.fixture(autouse=True)
def patch_image_search_url(monkeypatch):
    monkeypatch.setattr("parzivai.image_search.IMAGE_SEARCH_URL", IMAGE_SEARCH_URL)


def test_adjust_image_url():
    url = "https://de.wikipedia.org/wiki/Mittelhochdeutsche_Sprache#/media/Datei:Inschrift_Grab_Ulrich_III_Liechtenstein.jpg?WID=400&HEI=400"
    expected = "https://de.wikipedia.org/wiki/Mittelhochdeutsche_Sprache#/media/Datei:Inschrift_Grab_Ulrich_III_Liechtenstein.jpg?WID=1000&HEI=1000"
    assert adjust_image_url(url) == expected

    url = "https://de.wikipedia.org/wiki/Mittelhochdeutsche_Sprache#/media/Datei:Inschrift_Grab_Ulrich_III_Liechtenstein.jpg"
    assert adjust_image_url(url) == url


def test_construct_image_search_url():
    topic = "mittelalter"
    expected_url = f"{IMAGE_SEARCH_URL}#%7B%22s%22%3A%22mittelalter%22%7D"
    assert construct_image_search_url(topic) == expected_url

    topic_with_quotes = '"ritter"'
    expected_url_with_quotes = f"{IMAGE_SEARCH_URL}#%7B%22s%22%3A%22ritter%22%7D"
    assert construct_image_search_url(topic_with_quotes) == expected_url_with_quotes
