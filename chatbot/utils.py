import json
import pickle


def load_json(filename: str) -> dict:
    """ Load JSON file into dict """
    try:
        with open(filename) as file:
            data = json.load(file)
            return data

    except FileNotFoundError as e:
        print(e)


def fill_list_zeros(n: int) -> list:
    """ Create a zero filled list of size n """
    return [0 for _ in range(len(n))]


def dump_to_file(data: tuple, output_filename: str) -> None:
    """ Serialize the data and dump to a file """
    with open(output_filename, "wb") as f:
        pickle.dump(data, f)
