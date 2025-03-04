import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import pickle
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents JSON file
with open('hindi.json', encoding="utf-8") as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add to documents in our corpus
        documents.append((word_list, intent['tag']))
        # Add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))

# Sort classes
classes = sorted(list(set(classes)))

# Save words and classes for later use
pickle.dump(words, open('words-H.pkl', 'wb'))
pickle.dump(classes, open('classes-H.pkl', 'wb'))

# Create our training data
training = []
output_empty = [0] * len(classes)

# Create a bag-of-words for each pattern
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Split the training data into features (train_x) and labels (train_y)
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

# Build a neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model using stochastic gradient descent
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=1)

# Save the trained model
model.save('chatbot_model-H.h5')

print("Training complete. Model saved as chatbot_model-H.h5")