import tflearn


class NeuralNetwork:
    ACTIVATION = "softmax"

    def __init__(self, training_data: dict, neurons: int, output: list):
        self.training_data = training_data
        self.neurons = neurons
        self.output = output

    def _create_hidden_layers(self, neural_network, number_of_layers: int):
        """
        Create the hidden layers with the number of layers / neurons provided
        :param neural_network: Neural network obj
        :param number_of_layers: Number of hidden layers we want to create for our DNN
        :return: neural network
        """
        for i in range(number_of_layers):
            net = tflearn.fully_connected(neural_network, self.neurons)
        return net

    def _create_net(self):
        """ Creates the neural network """
        # Create the input layer from our training bag of words
        neural_network = tflearn.input_data(shape=[None, len(self.training_data[0])])

        # 2 hidden layers
        neural_network = self._create_hidden_layers(neural_network=neural_network, number_of_layers=2)

        # Softmax activation on each output neuron to provide us with a probability
        neural_network = tflearn.fully_connected(neural_network, len(self.output[0]), activation=self.ACTIVATION)

        # Apply regression to the provided input
        neural_network = tflearn.regression(neural_network)
        return neural_network

    def build_model(self, number_of_iterations: int):
        """ Train our Deep Neural Network - Retrieval Based Model for the number of iterations provided and save it. """
        net = self._create_net()
        model = tflearn.DNN(net)
        model.fit(self.training_data, self.output, n_epoch=number_of_iterations, batch_size=8, show_metric=True)
        model.save("model.tflearn")
        return model
