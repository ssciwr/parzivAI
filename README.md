# ParzivAI: AI-based assistent for medieval history and middle high German
![License: MIT](https://img.shields.io/github/license/ssciwr/parzivAI)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/parzivAI/ci.yml?branch=main)
![codecov](https://img.shields.io/codecov/c/github/ssciwr/parzivAI)
![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_parzivAI&metric=alert_status)
![Language](https://img.shields.io/github/languages/top/ssciwr/parzivAI)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssciwr/parzivAI/blob/main/example_notebooks/demo.ipynb)

**_This project is currently under development!_**

ParzivAI is an interactive chatbot designed to answer questions related to medieval history and the Middle High German language and literature. In addition to the chat functionality, ParzivAI offers linguistic analysis tools, an image search feature, and an interactive quiz on historical topics.

- **Chatbot:** Interact with ParzivAI to receive detailed answers to historical and linguistic questions.
- **Linguistic Analysis:** Automatically perform part-of-speech tagging for both modern German and Middle High German texts.
- **Image Search:** Search for historically themed images.
- **Quiz:** Test your knowledge of the Middle Ages with an interactive quiz.
- **Feedback:** Provide feedback on the quality of the responses directly in the app.

## Installation instructions

Ensure that Python 3.8 or higher is installed. It is recommended to use conda environments. In your conda (or other Python) environment, install parzivAI from the GitHub repository:
```bash
pip install pip install git+https://github.com/ssciwr/parzivAI
```
parzivAI makes use from [spaCy](https://spacy.io/) under the hood. Download the required spaCy models using 
```bash
python -m spacy download de_core_news_sm
```
(*TODO: Download models on the fly if not found through the spacy cli*)

For Middle High German, a specially trained model must be loaded, and its path needs to be integrated into the code.
(*TODO: Make sure this is platform-agnostic and can also be done on-the-fly*)

### Playwright Setup
(*TODO: Do we still want to include playwright?*)

If you use the Playwright library for image search, execute the following command:

```bash
playwright install
```

## Usage 
Start the application with the following command:

```bash
streamlit run app.py
```
Then open your browser and navigate to `http://localhost:8501`.
(*TODO: Check port number and exceptions*)


## API Key 
Ensure that your Tavily API key is set as an environment variable:
```bash
export TAVILY_API_KEY='your_api_key_here'
```

## Project structure
### English
```
ParzivAI/
│
├── parzivai/main.py           # Main Streamlit application
├── pyproject.toml             # Installation file and dependencies
├── requirements-dev.txt       # Developer dependencies
├── README.md                  # Overview
├── example_notebooks/         # Usage examples
├── docs/                      # Documentation
```

## License 
This project is licensed under the MIT License. See the `LICENSE` file for further details.

(*TODO: Further components of the documentation*)
