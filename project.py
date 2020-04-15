import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

# Load in intents.json data
def load_json_file():
    with open('intents.json') as file:
        data = json.load(file)
    return data

# Return a list including all the different words from our patterns inside of Intents.json
def tokenize_words(words):
    tokenized_words = nltk.word_tokenize(words)
    return tokenized_words

# Stemming takes each word in our pattern and reduces it to the root word. Removes question marks to improve accuracy
def stem_words(words):
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    return words

# Append words and classes to their corresponding lists. Fills our documents with all our pattern responses and classes
def append_words_and_classes(intents, words, classes, document_patterns, document_classes):
     for intent in intents:
            patterns = intent['patterns']
            for pattern in patterns:
                tokenized_words = tokenize_words(pattern)
                words.extend(tokenized_words)
                document_patterns.append(tokenized_words)
                document_classes.append(intent['class'])
                
                if intent['class'] not in classes:
                    classes.append(intent['class'])

# Returns a list of zeros. The length of the list is the length of size.                 
def fill_list_zeros(size):
    return [0 for _ in range(len(size))]

# Create the bag of words used for training our model
def training_bag_of_words(input_words, pattern_words, bag_of_words):
     for word in input_words:
            if word in pattern_words:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
     return bag_of_words


# Create the Neural Network
def create_neural_network(training,output):
        neural_network = tflearn.input_data(shape=[None, len(training[0])]) # Input layer -> Takes our training bag of words
        neural_network = tflearn.fully_connected(neural_network, 8) # 8 neurons for first hidden layer
        neural_network = tflearn.fully_connected(neural_network, 8) # 8 neurons for second hidden layer
        neural_network = tflearn.fully_connected(neural_network, len(output[0]), activation='softmax') # Softmax activation on each output neuron to provide us with a probability
        neural_network = tflearn.regression(neural_network) # Apply regression to the provided input
        return neural_network

# Converts our user input string into a list of zeros and ones. We use this list as input for our model.
def convert_input_into_bag_of_words(user_input, words):
    bag = fill_list_zeros(words)
    user_input_words = tokenize_words(user_input)
    # Stem the words in user input
    user_input_words = [stemmer.stem(word.lower()) for word in user_input_words]

    # If the current word we're evaluating in our user input string is inside our patterns word list -> change the current index in our bag of words from 0 to 1. 
    for user_word in user_input_words:
        for i, word in enumerate(words):
            if word == user_word:
                bag[i] = 1

        return np.array(bag)

# Returns a list of all the likely responses 
def fetch_responses(model, bag_of_words, user_input, words):
    responses = model.predict([bag_of_words])[0]
    return responses
     
# Returns the index of the response with the highest probability (softmax activation)
def fetch_most_likely_response(responses):
    most_likely_response = np.argmax(responses)
    return most_likely_response

def main():
    start_program = True
    data = load_json_file() # Load in our training data

    while start_program:
        # Initalize Variables
        words = [] 
        classes = []
        document_patterns = []
        document_classes = []
        training = []
        output = []
        intents = data['intents']

        # Fill our lists with the corresponding data from our Intents.json
        append_words_and_classes(intents, words, classes, document_patterns, document_classes)

        # Stem our words and remove all duplicates. Removing duplicates will increase accuracy
        words = stem_words(words)
        words = sorted(list(set(words)))

        # Sort our class list
        classes = sorted(classes)
   
        # Create an output list with the length of our total number of classes. Fill each index with zero
        output_zeros = fill_list_zeros(classes)

        # Loop through our patterns inside intent.json
        for x, pattern in enumerate(document_patterns):
            bag = []

            # Stems each word in patterns
            pattern_words = [stemmer.stem(w) for w in pattern]

            # converts our user input into a bag of words
            bag = training_bag_of_words(words,pattern_words,bag)


            # Copy the output zero list
            output_position = output_zeros[:]

            # Look through the classes list. Find that class index and set it to one
            output_position[classes.index(document_classes[x])] = 1

            # Append our training list with the bag of words
            training.append(bag)

            # Append our output list with zeros or ones. 0 represents the word isn't our pattern. 1 represents the word exists in our pattern.
            output.append(output_position)
        
            # Dump our data in a file
            with open('data.pickle', 'wb') as f:
                pickle.dump((words, classes, training, output), f)
        
        
        # Transform our training and output lists into numpy arrays
        training = np.array(training)
        output = np.array(output)

        # Clears the default graph stack and resets the global default graph
        tf.reset_default_graph()
        
        # Create our Neural Network
        neural_network = create_neural_network(training, output)

        # Train our Deep Neural Network model for 1000 iterations and save it. 
        model = tflearn.DNN(neural_network) 
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True) 
        model.save('model.tflearn')

        # Chat function to initalize chatting with the bot and fetch the highest predicited response from each input
        def start_chatting():
            print('Start talking with McMaster Student Services (type quit to stop)')
            while True:
                user_input = input('You: ')
                if user_input.lower() == 'quit': # Escape option for the user
                    break

                # Convert our user input into a bag of words
                bag_of_words = convert_input_into_bag_of_words(user_input, words)
                
                # Fetch all the likely responses for the user
                responses =  fetch_responses(model, bag_of_words, user_input, words)

                # Select all the responses from the highest predicted neuron (class)
                most_likely_response = fetch_most_likely_response(responses) 
                predicted_class = classes[most_likely_response]

                # Constant threshold value.
                RESPONSE_THRESHOLD = 0.2

                # If the response is greater than the threshold -> Fetch a random response from the highest predicted class, otherwise tell the user it doesn't understand the question
                if responses[most_likely_response] > RESPONSE_THRESHOLD:
                    for intent in data['intents']:
                        if intent['class'] == predicted_class:
                            bot_responses = intent['responses']
                    print('McMaster Student Services: ', random.choice(bot_responses))
                else:
                    print("I don't understand. I'm a robot so I get things wrong from time to time. Please try again :)")

        break # stop the program

    start_chatting()

main()

print('stop')