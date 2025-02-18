# Translate Mate

**Translate Mate** is a web-based application built using **Flask** (Python). It takes input in **English** and provides responses either in **English and Tamil** or **English and Hindi**, based on the user's preference. The app uses predefined **intents** and **patterns** to match the user's input and return appropriate responses.

## Features

- **Pattern Matching**: Matches user input with predefined patterns and responds accordingly.
- **Multilingual Responses**: Supports English responses along with Tamil or Hindi based on user preference.
- **Intent-Based Responses**: Provides responses based on predefined intents such as greetings, asking about capabilities, or asking for the time.

## Files in the Project

- **app.py**: Main Flask application file for routing and backend logic. Handles interactions and matches user input to predefined intents.
- **models**: Contains any logic related to intent recognition or classification (e.g., machine learning models or rule-based systems).
- **words_classes**: Files related to processing words and classifying them into intents.
- **.json files**: Store user queries, patterns, and predefined responses for matching user input.
- **.html files**: Frontend templates for rendering web pages and interacting with the user.

## Requirements

- Python 3.x
- Flask
- JSON for storing patterns and responses
- Gunicorn (for production)
- NGINX (if deploying to cloud platforms like AWS)

### Install Dependencies

To get started with the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/translate-mate.git
   cd translate-mate
