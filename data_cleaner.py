import nltk
from nltk.stem.lancaster import LancasterStemmer


class DataCleaner:
    STEMMER = LancasterStemmer()

    @staticmethod
    def tokenize_words(words: list) -> list:
        """
        Word tokenization is the process of splitting a large sample of text into words.
        :param words: List of words (Includes single words and/or sentences)
        :return: List of individual words
        """
        tokenized_words = nltk.word_tokenize(words)
        return tokenized_words

    @classmethod
    def stem_words(cls, words: list) -> list:
        """
        Stem our words and remove all duplicates. Removing duplicates will increase accuracy
        :param words: List of words (Includes single words and/or sentences)
        :return: List of stemmed words
        """
        words = [cls.STEMMER.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))
        return words
