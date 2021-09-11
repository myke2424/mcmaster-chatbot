from chatbot.bot import ChatBot
from chatbot.data_set import DataSet
from chatbot.neural_network import NeuralNetwork

CHAT_BOT_DATA_FILENAME = "data.json"
NUMBER_OF_NEURONS = 8
NUMBER_OF_ITERATIONS = 1000

data_set = DataSet(filename=CHAT_BOT_DATA_FILENAME)
neural_network = NeuralNetwork(training_data=data_set.training_data, neurons=NUMBER_OF_NEURONS, output=data_set.output)
model = neural_network.build_model(number_of_iterations=NUMBER_OF_ITERATIONS)
mcmaster_chat_bot = ChatBot(model=model, data_set=data_set)

__all__ = ['mcmaster_chat_bot']
