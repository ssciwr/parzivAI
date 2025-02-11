# ParzivAI: KI-basierter Assistent für Mittelalterliche Geschichte und Mittelhochdeutsche Sprache
![License: MIT](https://img.shields.io/github/license/ssciwr/parzivAI)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/parzivAI/ci.yml?branch=main)
![codecov](https://img.shields.io/codecov/c/github/ssciwr/parzivAI)
![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_parzivAI&metric=alert_status)
![Language](https://img.shields.io/github/languages/top/ssciwr/parzivAI)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssciwr/parzivAI/blob/main/example_notebooks/demo.ipynb)

**_Dieses Projekt befindet sich aktuell unter aktiver Entwicklung!_**

ParzivAI ist ein interaktiver Chatbot, der entwickelt wurde, um Fragen zur mittelalterlichen Geschichte und zur mittelhochdeutschen Sprache und Literatur zu beantworten. Neben der Chat-Funktion bietet ParzivAI linguistische Analysetools, eine Bildersuchfunktion und ein interaktives Quiz zu historischen Themen.

- **Chatbot:** Interagiere mit ParzivAI, um detaillierte Antworten zu historischen und sprachlichen Fragen zu erhalten.
- **Linguistische Analyse:** Automatische Part-of-Speech-Tagging für moderne deutsche und mittelhochdeutsche Texte.
- **Bildersuche:** Suche nach themenspezifischen historischen Bildern.
- **Quiz:** Teste Dein Wissen über das Mittelalter in einem interaktiven Lernquiz.
- **Feedback:** Teile Feedback zur Qualität der Antworten direkt in der App.

## Installationsanweisungen

Stelle sicher, dass Python 3.8 oder höher installiert ist. Es ist empfohlen, conda Umgebungen zu nutzen. In deiner conda (oder anderer Python) Umgebung, installiere parzivAI vom GitHub Repository:
```bash
pip install pip install git+https://github.com/ssciwr/parzivAI
```
parzivAI nutzt [spaCy](https://spacy.io/) für die Textverarbeitung. Lade die benötigten spaCy Modelle herunter mit
```bash
python -m spacy download de_core_news_sm
```

Für Mittelhochdeutsch muss ein speziell trainiertes Modell geladen und der Pfad in den Code integriert werden.

### Playwright Setup

Wenn Du die Playwright-Bibliothek für die Bildersuche verwendest, führe diesen Befehl aus:

```bash
playwright install
```

## Verwendung
Starte die Anwendung mit dem folgenden Befehl:

```bash
streamlit run app.py
```

Öffne Deinen Browser und navigiere zu `http://localhost:8501`.

## API-Schlüssel
Stelle sicher, dass Dein Tavily API-Schlüssel als Umgebungsvariable gesetzt ist:

```bash
export TAVILY_API_KEY='your_api_key_here'
```

## Projektstruktur
```
ParzivAI/
│
├── parzivai/main.py           # Hauptanwendung für Streamlit
├── pyproject.toml             # Installationsdatei und Abhängigkeiten
├── requirements-dev.txt       # Entwickler-Abhängigkeiten
├── README.md                  # Übersicht
├── example_notebooks/         # Verwendungsbeispiele
├── docs/                      # Dokumentation
```

## Lizenz
Dieses Projekt steht unter der MIT-Lizenz. Siehe die `LICENSE`-Datei für weitere Details.
