# ParzivAI: AI-based assistent for medieval history and middle high German
![License: MIT](https://img.shields.io/github/license/ssciwr/parzivAI)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/parzivAI/ci.yml?branch=main)
![codecov](https://img.shields.io/codecov/c/github/ssciwr/parzivAI)
![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_parzivAI&metric=alert_status)
![Language](https://img.shields.io/github/languages/top/ssciwr/parzivAI)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssciwr/parzivAI/blob/main/example_notebooks/demo.ipynb)

**_This project is currently under development!_**
*For the German overview, please see [here](README_German.md)*

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

For Middle High German, a specially trained model must be loaded, and its path needs to be integrated into the code. The model can be found [here](https://github.com/Middle-High-German-Conceptual-Database/Spacy-Model-for-Middle-High-German). Git clone the repository and place it in the same folder as the parzivAI repo:
```
you-folder/
│
├── parzivai                                 # parzivai 
├── Spacy-Model-for-Middle-High-German       # spaCy model
```
(*TODO: Make sure this is platform-agnostic and can also be done on-the-fly*)

You will also need the Ollama chatbot, that you can install using
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
on linux OS, for other OS please refer to the [Ollama installation guide](https://ollama.com/download). The default model used by parzivAI is the llama3.2:1b model that you need to download using
```
ollama pull llama3.2:1b
```

### Playwright Setup
(*TODO: Do we still want to include playwright?*)

If you use the Playwright library for image search, execute the following command:

```bash
playwright install
```

## API Key 
Ensure that your Tavily API key is set as an environment variable:
```bash
export TAVILY_API_KEY='your_api_key_here'
```

## Usage 
Cd into the `parzivai` package folder and start the application with the following command:

```bash
streamlit run app.py
```
Then open your browser and navigate to `http://localhost:8501`.
(*TODO: Check port number and exceptions*)


## Project structure
### English
```
ParzivAI/
│
├── parzivai/app.py           # Main Streamlit application
├── parzivai/chat_models.py   # parzivai chat models module
├── parzivai/image_search.py  # parzivai image search module
├── parzivai/input_output.py  # parzivai input output handling module
├── parzivai/text_tagging.py  # parzivai syntactic annotations module
├── parzivai/data/*.json      # parzivai data and configuration files
├── parzivai/test/*.py        # parzivai unit and integration/system/end-to-end tests
├── pyproject.toml            # Installation file and dependencies
├── requirements-dev.txt      # Developer dependencies
├── README.md                 # Overview
├── example_notebooks/        # Usage examples
├── docs/                     # Documentation
```

## Create models
Download models using ollama (`ollama pull my-model`) or from Hugging Face. You can use `ollama create my-model` in the model directory to add the model to the local ollama model library (on Linux, stored in `/usr/share/ollama/.ollama/models`, on Windows in `~/.ollama/models`). You can see the installed models using `ollama list`.

Test the ollama model using the `ollama run my-model` command. Ollama can use various model formats for input, see [here](https://github.com/ollama/ollama/blob/main/docs/import.md).


## License 
This project is licensed under the MIT License. See the `LICENSE` file for further details.

(*TODO: Further components of the documentation*)

