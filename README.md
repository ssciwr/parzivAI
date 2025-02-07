
# Title

Description


## Installation instructions

## Usage



# ParzivAI: KI-basierter Assistent für Mittelalterliche Geschichte und Mittelhochdeutsche Sprache

## Übersicht

ParzivAI ist ein interaktiver Chatbot, der entwickelt wurde, um Fragen zur mittelalterlichen Geschichte und zur mittelhochdeutschen Sprache und Literatur zu beantworten. Neben der Chat-Funktion bietet ParzivAI linguistische Analysetools, eine Bildersuchfunktion und ein interaktives Quiz zu historischen Themen.

## Funktionen

- **Chatbot:** Interagiere mit ParzivAI, um detaillierte Antworten zu historischen und sprachlichen Fragen zu erhalten.
- **Linguistische Analyse:** Automatische Part-of-Speech-Tagging für moderne deutsche und mittelhochdeutsche Texte.
- **Bildersuche:** Suche nach themenspezifischen historischen Bildern.
- **Quiz:** Teste Dein Wissen über das Mittelalter in einem interaktiven Lernquiz.
- **Feedback:** Teile Feedback zur Qualität der Antworten direkt in der App.

## Installation

### Voraussetzungen

Stelle sicher, dass Python 3.8 oder höher installiert ist. Installiere alle Abhängigkeiten mit:

```bash
pip install -r requirements.txt
```

### Spacy-Modelle

Für moderne deutsche Texte benötigst Du das Spacy-Modell `de_core_news_sm`. Installiere es mit:

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

### Lokalen Server starten

Starte die Anwendung mit dem folgenden Befehl:

```bash
streamlit run app.py
```

Öffne Deinen Browser und navigiere zu `http://localhost:8501`.

### Funktionalitäten

1. **Chatbot:** Stelle Fragen zu mittelalterlicher Geschichte, Linguistik oder Literatur.
2. **Linguistische Analyse:** Wende POS-Tagging (Part-of-Speech) auf moderne und mittelhochdeutsche Texte an.
3. **Bildersuche:** Suche nach Bildern, die mit historischen Themen in Verbindung stehen.
4. **Quiz:** Teste Dein Wissen über das Mittelalter in einem Lernquiz.

## API-Schlüssel

Stelle sicher, dass Dein Tavily API-Schlüssel als Umgebungsvariable gesetzt ist:

```bash
export TAVILY_API_KEY='your_api_key_here'
```

## Projektstruktur

```
ParzivAI/
│
├── app.py                    # Hauptanwendung für Streamlit
├── requirements.txt           # Abhängigkeiten
├── README.md                  # Dokumentation
├── faiss_index/               # FAISS-Index
├── pdfs/                      # PDF-Dokumente
├── texte.csv                  # CSV-Datei mit Texten
├── models/                    # Sprachmodelle
├── quiz_data.json             # Quiz-Daten
```

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe die `LICENSE`-Datei für weitere Details.
