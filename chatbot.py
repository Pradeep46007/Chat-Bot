import json
import random
import os
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Ensure necessary NLTK data packages are downloaded
def download_nltk_requirements():
    for package in ['punkt', 'wordnet', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'corpora/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

# 1. Create a default intents.json if it doesn't already exist
def create_default_intents(filepath='intents.json'):
    if not os.path.exists(filepath):
        intents_data = {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["Hi", "Hello", "Hey", "Good day", "Howdy", "Is anyone there?"],
                    "responses": ["Hello!", "Hi there, how can I help you?", "Greetings! What can I do for you today?"]
                },
                {
                    "tag": "goodbye",
                    "patterns": ["Bye", "See you later", "Goodbye", "Have a good day", "I'm leaving"],
                    "responses": ["Goodbye!", "Have a great day!", "Talk to you later!"]
                },
                {
                    "tag": "thanks",
                    "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping"],
                    "responses": ["Happy to help!", "Anytime!", "My pleasure!"]
                },
                {
                    "tag": "order_status",
                    "patterns": ["Where is my order?", "Track my package", "Has my order shipped?", "Order status"],
                    "responses": ["Please provide your 8-digit Order ID to track your shipment.", "You can track your order status in your account dashboard under 'My Orders'."]
                },
                {
                    "tag": "refund",
                    "patterns": ["I want a refund", "How do I return an item?", "Can I get my money back?", "Return policy"],
                    "responses": ["Items can be returned within 30 days of delivery. Visit our returns portal to generate a shipping label."]
                }
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(intents_data, f, indent=4)
        print(f"Created sample '{filepath}' file.")

# 2. Data Preprocessing & Bag-of-Words Vectorization
def prepare_data(intents_filepath='intents.json'):
    with open(intents_filepath) as file:
        data = json.load(file)

    words = []
    labels = []
    documents = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            # Tokenization
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, intent['tag']))
            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    # Lemmatization and lowercasing
    words = sorted(list(set([lemmatizer.lemmatize(w.lower()) for w in words])))
    labels = sorted(list(set(labels)))

    training = []
    output = []
    output_empty = [0] * len(labels)

    # Build Bag-of-Words representations
    for doc in documents: 
        bag = []
        pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
        
        for w in words:
            bag.append(1 if w in pattern_words else 0)

        output_row = list(output_empty)
        output_row[labels.index(doc[1])] = 1

        training.append(bag)
        output.append(output_row)

    X_train = np.array(training)
    y_train = np.array(output)

    # Save processed vocabulary and labels for runtime prediction
    with open('words.pkl', 'wb') as f:
        pickle.dump(words, f)
    with open('labels.pkl', 'wb') as f:
        pickle.dump(labels, f)

    return X_train, y_train, words, labels, data

# 3. Model Architecture & Training
def build_and_train_model(X_train, y_train, model_filename='chatbot_model.h5'):
    model = Sequential([
        Dense(128, input_shape=(len(X_train[0]),), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(y_train[0]), activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print("Training model...")
    model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=0)
    model.save(model_filename)
    print(f"Model saved to '{model_filename}'.")
    return model

# 4. Inference Helper Functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_intent(sentence, model, words, labels, ERROR_THRESHOLD=0.25):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]), verbose=0)[0]
    
    # Filter out predictions below the probability threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({"intent": labels[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I didn't quite understand that. Could you rephrase?"
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I don't have an answer for that."

# 5. Interactive Chat Loop
def run_chat_loop():
    download_nltk_requirements()
    create_default_intents('intents.json')

    if not (os.path.exists('chatbot_model.h5') and os.path.exists('words.pkl') and os.path.exists('labels.pkl')):
        X_train, y_train, words, labels, intents_json = prepare_data('intents.json')
        model = build_and_train_model(X_train, y_train, 'chatbot_model.h5')
    else:
        with open('words.pkl', 'rb') as f:
            words = pickle.load(f)
        with open('labels.pkl', 'rb') as f:
            labels = pickle.load(f)
        model = load_model('chatbot_model.h5')
        with open('intents.json') as f:
            intents_json = json.load(f)

    print("\n==================================================")
    print("Chatbot is online! Type 'quit' or 'exit' to stop.")
    print("==================================================\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower().strip() in ['quit', 'exit']:
                print("Bot: Goodbye!")
                break
            if not user_input.strip():
                continue

            intents = predict_intent(user_input, model, words, labels)
            response = get_response(intents, intents_json)
            print(f"Bot: {response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye!")
            break

if __name__ == "__main__":
    run_chat_loop()