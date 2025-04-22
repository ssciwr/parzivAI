import streamlit as st
from urllib.parse import quote
import json
from playwright.async_api import async_playwright
import asyncio
from importlib import resources
# Load configuration from file
PKG = resources.files("parzivai")
CONFIG_PATH = PKG / "data" / "config.json"

try:
    with open(CONFIG_PATH, "r") as config_file:
        config = json.load(config_file)
        IMAGE_SEARCH_URL = config.get(
            "image_search_url"
        )  # it would be defined in the config file
except FileNotFoundError:
    st.error(f"Configuration file not found at {CONFIG_PATH}. Please ensure it exists.")
    raise
except json.JSONDecodeError as e:
    st.error(f"Error decoding configuration file: {e}")
    raise


def adjust_image_url(url: str) -> str:
    """Adjust image URL parameters for higher resolution."""
    if "WID=400" in url and "HEI=400" in url:
        url = url.replace("WID=400", "WID=1000").replace("HEI=400", "HEI=1000")
    return url


def construct_image_search_url(topic: str) -> str:
    """Construct the search URL for the hardcoded site."""
    encoded_topic = quote(topic.strip('"'))
    return f"{IMAGE_SEARCH_URL}#%7B%22s%22%3A%22{encoded_topic}%22%7D"


async def fetch_images(topic: str):
    """Fetch images related to a topic from the hardcoded site."""
    search_url = construct_image_search_url(topic)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(search_url)
        await asyncio.sleep(5)  # Wait for the page to fully load

        image_data = await page.evaluate(
            """() => {
            const images = document.querySelectorAll('img.hit-btn-image-sm');
            const data = Array.from(images).map(img => {
                const container = img.closest('.hit-cell');
                if (!container) {
                    console.error('Container not found for image:', img.src);
                    return null;
                }

                const nameElement = Array.from(container.querySelectorAll('.property .additional_info_medium')).find(el => el.innerText.includes('Bildthema:'));
                const archiveNumberElement = Array.from(container.querySelectorAll('.property .additional_info_medium')).find(el => el.innerText.includes('Archivnummer:'));

                const name = nameElement ? nameElement.nextElementSibling.innerText : 'Name not found';
                const archiveNumber = archiveNumberElement ? archiveNumberElement.nextElementSibling.innerText : 'Archivnummer not found';

                if (!nameElement || !archiveNumberElement) {
                    console.error('Name or archive number element not found for image:', img.src);
                }

                return {
                    url: img.src,
                    name: name,
                    archiveNumber: archiveNumber
                };
            }).filter(item => item !== null);
            return data;
        }"""
        )

        await browser.close()
        return image_data


async def display_images(topic: str):
    """Display fetched images in Streamlit."""
    image_data = await fetch_images(topic)
    for data in image_data:
        st.image(
            data["url"],
            caption=f"Bildthema: {data['name']}, Archivnummer: {data['archiveNumber']}, URL: {data['url']}",
            use_column_width=True,
        )
