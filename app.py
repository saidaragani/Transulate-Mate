from flask import Flask, request, jsonify, render_template
import spacy
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import json
import random

app = Flask(__name__)

# Use the English spaCy model since the input is in English
nlp = spacy.load("en_core_web_sm")

# Initialize language-specific data structures
models = {'tamil': None, 'hindi': None}
word_classes = {
    'tamil': {'words': None, 'classes': None},
    'hindi': {'words': None, 'classes': None}
}
intents = {'tamil': None, 'hindi': None}
responses = {'tamil': {}, 'hindi': {}}

# Track the last used language; default is Tamil.
last_selected_language = 'tamil'

def load_language_data(language):
    """Force reload model, words, classes, and responses for the selected language."""
    global models, word_classes, intents, responses
    try:
        # Determine file suffix: for Hindi, append "-H"; otherwise, use no suffix.
        model_suffix = "-H" if language == "hindi" else ""
        print(f"Loading model file: chatbot_model{model_suffix}.h5")
        models[language] = load_model(f'chatbot_model{model_suffix}.h5')
        
        # Load pickle files for words and classes
        with open(f'words{model_suffix}.pkl', 'rb') as f:
            word_classes[language]['words'] = pickle.load(f)
        with open(f'classes{model_suffix}.pkl', 'rb') as f:
            word_classes[language]['classes'] = pickle.load(f)
        
        # Load intents JSON file
        intent_file = "hindi.json" if language == "hindi" else "tamil.json"
        print(f"Loading intents file: {intent_file}")
        with open(intent_file, 'r', encoding='utf-8') as file:
            intents[language] = json.load(file)
        
        # Build responses dictionary from the intents
        responses[language] = {intent['tag']: intent['responses'] for intent in intents[language].get('intents', [])}
        
        print(f"Successfully loaded {language} data.")
    except Exception as e:
        print(f"Error loading {language} data: {e}")
        raise  # Re-raise the exception to handle it in the calling function

# Define words to ignore in the bag-of-words model
ignore_words = {'?', '!', '.', ','}

def preprocess_input(text, language):
    """Tokenize the input text and create a bag-of-words vector."""
    doc = nlp(text)
    text_words = {token.lemma_.lower() for token in doc if token.text not in ignore_words}
    words = word_classes[language].get('words', [])
    bag = [1 if word in text_words else 0 for word in words]
    return np.array([bag])

def predict_class(text, language):
    """Predict the intent class using the appropriate deep learning model."""
    input_data = preprocess_input(text, language)
    model = models[language]
    if model is None:
        raise ValueError(f"Model for {language} is not loaded.")
    classes = word_classes[language].get('classes', [])
    prediction = model.predict(input_data)[0]
    predicted_index = np.argmax(prediction)
    predicted_tag = classes[predicted_index] if classes else None
    print(f"Predicted intent: {predicted_tag}")  # Debugging line
    return predicted_tag

def get_response(tag, language):
    """Select a random response from the loaded responses for the given intent tag."""
    return random.choice(responses[language].get(tag, ["I'm not sure how to respond to that."]))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global last_selected_language
    data = request.get_json()
    if not data:
        return jsonify({'response': "Invalid request. Please send JSON data.", 'language': last_selected_language}), 400
    
    user_message = data.get('message', '')
    new_language = data.get('language', last_selected_language).lower()
    
    # Switch language if necessary
    if new_language != last_selected_language:
        last_selected_language = new_language
        load_language_data(new_language)
    
    if not user_message:
        return jsonify({'response': "Please send a message.", 'language': last_selected_language}), 400
    
    try:
        predicted_class = predict_class(user_message, last_selected_language)
        response = get_response(predicted_class, last_selected_language)
        return jsonify({'response': response, 'language': last_selected_language})
    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({'response': "An error occurred while processing your message.", 'language': last_selected_language}), 500

if __name__ == '__main__':
    # Initial load for the default language
    load_language_data(last_selected_language)
    app.run(debug=True)