import json

from chatbot.data_cleaner import DataCleaner
from chatbot.exceptions import InvalidDataSetError
from chatbot.utils import dump_to_file, fill_list_zeros, load_json


class DataSet:
    def __init__(self, filename: str):
        self.raw_data = load_json(filename)["data"]
        self.training_data = []
        self.output = []

        self.words = []
        self.classes = []
        self.document_patterns = []
        self.document_classes = []
        self.bag_of_words = []

        self._create_training_data()

    @staticmethod
    def _validate_intent(intent: dict) -> None:
        if intent.get("patterns") is None or intent.get("class") is None:
            raise InvalidDataSetError(
                f"Invalid data set - Required keys ('patterns', 'class', responses'): {json.dumps(intent, indent=2)}")

    def _fill_data(self, intent: dict) -> None:
        """ Fill pattern/class/word lists """
        self._validate_intent(intent)

        patterns = intent["patterns"]
        class_ = intent["class"]

        for pattern_words in patterns:
            tokenized_words = DataCleaner.tokenize_words(pattern_words)
            self.words.extend(tokenized_words)
            self.document_patterns.append(tokenized_words)
            self.document_classes.append(class_)

            if intent["class"] not in self.classes:
                self.classes.append(class_)

        self.classed = sorted(self.classes)

    def _fill(self) -> None:
        """ Fill our lists with the corresponding data from our raw data intent  """
        for intent in self.raw_data:
            self._fill_data(intent)

        self.classes = sorted(self.classes)

    def _dump(self):
        """Dump our data in a file """
        args = (self.words, self.classes, self.training_data, self.output)
        dump_to_file(data=args, output_filename="../data.pickle")

    def _create_training_data(self):
        self._fill()
        _DataBuilder.build_training_data_set(self)
        self._dump()


class _DataBuilder:
    @staticmethod
    def build_training_data_set(data_set: "DataSet"):
        """ """
        for x, pattern in enumerate(data_set.document_patterns):
            # Stems each word in patterns
            pattern_words = [DataCleaner.stem_words(w) for w in pattern]

            # converts our user input into a bag of words
            data_set.bag_of_words = _DataBuilder.build_bag_of_words(data_set.words, pattern_words)

            # Create an output list with the length of our total number of classes. Fill each index with zero
            output_zeros = fill_list_zeros(data_set.classes)
            output_position = output_zeros[:]

            # Look through the classes list. Find that class index and set it to one
            output_position[data_set.classes.index(data_set.document_classes[x])] = 1

            # Append our training list with the bag of words
            data_set.training_data.append(data_set.bag_of_words)

            # Append our output list with zeros or ones. 0 represents the word isn't our pattern. 1 represents the word exists in our pattern.
            data_set.output.append(output_position)

    @staticmethod
    def build_bag_of_words(input_words: list, pattern_words: list):
        """  Create the bag of words used for training our model  """
        bag_of_words = []
        for word in input_words:
            if word in pattern_words:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
        return bag_of_words
