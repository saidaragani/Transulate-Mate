from flask import Flask, request, jsonify, render_template
import spacy
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import json
import random

app = Flask(__name__)

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the trained model
model = load_model('chatbot_model.h5')

# Load the words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load intents file
with open('projectt_updated.json') as file:
    intents = json.load(file)

# Create a dictionary for responses
responses = {}
for intent in intents['intents']:
    responses[intent['tag']] = intent['responses']

ignore_words = ['?', '!', '.', ',']

def preprocess_input(text):
    """
    Tokenize, lemmatize, and create a bag of words array for the input text.
    """
    # Tokenize and lemmatize input text
    doc = nlp(text)
    text_words = [token.lemma_.lower() for token in doc if token.text not in ignore_words]

    # Create a bag of words array
    bag = [1 if word in text_words else 0 for word in words]

    return np.array([bag])

def predict_class(text):
    """
    Predict the class of the input text using the trained model.
    """
    # Preprocess the input text
    input_data = preprocess_input(text)

    # Predict the class
    prediction = model.predict(input_data)[0]

    # Get the index of the highest probability
    predicted_class_index = np.argmax(prediction)

    # Map the index to the class name
    return classes[predicted_class_index]

def get_response(tag):
    """
    Get a response for a given class tag.
    """
    return random.choice(responses.get(tag, ["I'm not sure how to respond to that."]))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    if user_message:
        predicted_class = predict_class(user_message)
        response = get_response(predicted_class)
        return jsonify({'response': response})
    return jsonify({'response': "Sorry, I didn't understand that."})

if __name__ == '__main__':
    app.run(debug=True)
