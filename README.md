
# Title

Description


## Installation instructions

## Usage


# ParzivAI: AI-Assisted Chatbot for Medieval History and Middle High German
# ParzivAI: KI-basierter Assistent für Mittelalterliche Geschichte und Mittelhochdeutsche Sprache

## Overview / Übersicht

### English
ParzivAI is an interactive chatbot designed to answer questions related to medieval history and the Middle High German language and literature. In addition to the chat functionality, ParzivAI offers linguistic analysis tools, an image search feature, and an interactive quiz on historical topics.

### Deutsch
ParzivAI ist ein interaktiver Chatbot, der entwickelt wurde, um Fragen zur mittelalterlichen Geschichte und zur mittelhochdeutschen Sprache und Literatur zu beantworten. Neben der Chat-Funktion bietet ParzivAI linguistische Analysetools, eine Bildersuchfunktion und ein interaktives Quiz zu historischen Themen.

## Functions / Funktionen

### English

- **Chatbot:** Interact with ParzivAI to receive detailed answers to historical and linguistic questions.
- **Linguistic Analysis:** Automatically perform part-of-speech tagging for both modern German and Middle High German texts.
- **Image Search:** Search for historically themed images.
- **Quiz:** Test your knowledge of the Middle Ages with an interactive quiz.
- **Feedback:** Provide feedback on the quality of the responses directly in the app.

### Deutsch

- **Chatbot:** Interagiere mit ParzivAI, um detaillierte Antworten zu historischen und sprachlichen Fragen zu erhalten.
- **Linguistische Analyse:** Automatische Part-of-Speech-Tagging für moderne deutsche und mittelhochdeutsche Texte.
- **Bildersuche:** Suche nach themenspezifischen historischen Bildern.
- **Quiz:** Teste Dein Wissen über das Mittelalter in einem interaktiven Lernquiz.
- **Feedback:** Teile Feedback zur Qualität der Antworten direkt in der App.

## Installation

### Requirements / Voraussetzungen

#### English
Ensure that Python 3.8 or higher is installed. Install all dependencies with:

```bash
pip install -r requirements.txt
```

#### Deutsch
Stelle sicher, dass Python 3.8 oder höher installiert ist. Installiere alle Abhängigkeiten mit:

```bash
pip install -r requirements.txt
```

### Spacy Models / Spacy-Modelle

#### English
For modern German texts, you need the Spacy model de_core_news_sm. Install it with:

```bash
python -m spacy download de_core_news_sm
```

For Middle High German, a specially trained model must be loaded, and its path integrated into the code.

#### Deutsch

Für moderne deutsche Texte benötigst Du das Spacy-Modell `de_core_news_sm`. Installiere es mit:

```bash
python -m spacy download de_core_news_sm
```

Für Mittelhochdeutsch muss ein speziell trainiertes Modell geladen und der Pfad in den Code integriert werden.

### Playwright Setup

#### English
If you use the Playwright library for image search, execute the following command:

```bash
playwright install
```

#### Deutsch
Wenn Du die Playwright-Bibliothek für die Bildersuche verwendest, führe diesen Befehl aus:

```bash
playwright install
```

## Usage / Verwendung

### Starting the Local Server / Lokalen Server starten

#### English
Start the application with the following command:

```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`.

#### Deutsch
Starte die Anwendung mit dem folgenden Befehl:

```bash
streamlit run app.py
```

Öffne Deinen Browser und navigiere zu `http://localhost:8501`.

### Features / Funktionalitäten

#### English
1. **Chatbot:** Ask questions about medieval history, linguistics, or literature.
2. **Linguistic Analysis:** Apply part-of-speech tagging to both modern German and Middle High German texts.
3. **Image Search:** Search for images related to historical themes.
4. **Quiz:** Test your knowledge of the Middle Ages with an interactive quiz.

#### Deutsch
1. **Chatbot:** Stelle Fragen zu mittelalterlicher Geschichte, Linguistik oder Literatur.
2. **Linguistische Analyse:** Wende POS-Tagging (Part-of-Speech) auf moderne und mittelhochdeutsche Texte an.
3. **Bildersuche:** Suche nach Bildern, die mit historischen Themen in Verbindung stehen.
4. **Quiz:** Teste Dein Wissen über das Mittelalter in einem Lernquiz.

## API Key / API-Schlüssel
### English
Ensure that your Tavily API key is set as an environment variable:

```bash
export TAVILY_API_KEY='your_api_key_here'
```

### Deutsch
Stelle sicher, dass Dein Tavily API-Schlüssel als Umgebungsvariable gesetzt ist:

```bash
export TAVILY_API_KEY='your_api_key_here'
```

## Project Structure / Projektstruktur
### English
```
ParzivAI/
│
├── parzivai/main.py           # Main Streamlit application
├── requirements-dev.txt       # Dependencies
├── README.md                  # Documentation
├── example_notebooks/         # Usage examples
├── docs/                      # Documentation
```


### Deutsch
```
ParzivAI/
│
├── parzivai/main.py           # Hauptanwendung für Streamlit
├── requirements-dev.txt       # Abhängigkeiten
├── README.md                  # Dokumentation
├── example_notebooks/         # Verwendungsbeispiele
├── docs/                      # Dokumentation
```

## License / Lizenz
### English
This project is licensed under the MIT License. See the `LICENSE` file for further details.

### Deutsch
Dieses Projekt steht unter der MIT-Lizenz. Siehe die `LICENSE`-Datei für weitere Details.
