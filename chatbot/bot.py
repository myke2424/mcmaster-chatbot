import random

import numpy as np

from chatbot.data_set import DataSet


class ChatBot:
    RESPONSE_THRESHOLD = 0.2

    def __init__(self, model, data_set: "DataSet"):
        self.model = model
        self.data_set = data_set

    @property
    def _responses(self) -> list:
        """  Returns a list of all the likely responses """
        responses = self.model.predict([self.data_set.bag_of_words])[0]
        return responses

    def _fetch_most_likely_response(self) -> int:
        """ Returns the index of the response with the highest probability (softmax activation) """
        most_likely_response = np.argmax(self._responses)
        return most_likely_response

    def _fetch_responses_and_class_from_neuron(self) -> tuple:
        """ Select all the responses from the highest predicted neuron (class) """
        most_likely_response = self._fetch_most_likely_response()
        predicted_class = self.data_set.classes[most_likely_response]

        return most_likely_response, predicted_class

    def _generate_response(self) -> None:
        """
        Generate a response for the end-user. If the response is greater than the threshold,
        fetch a random response from the highest predicted class.
        """
        most_likely_response, predicted_class = self._fetch_responses_and_class_from_neuron()

        if self._responses[most_likely_response] > self.RESPONSE_THRESHOLD:
            for intent in self.data_set.raw_data:
                if intent["class"] == predicted_class:
                    bot_responses = intent["responses"]
            print("McMaster Student Services: ", random.choice(bot_responses))
        else:
            print("I don't understand. I'm a robot so I get things wrong from time to time. Please try again :)")

    def start(self) -> None:
        """ Initialize chat-bot... Start CHATTING! """
        print("Start talking with McMaster Student Services (type quit to stop)")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "quit":  # Escape option for the user
                break

            self._generate_response()
