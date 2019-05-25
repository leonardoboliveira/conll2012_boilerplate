from features import FeatureMapper as m
from . import mentions


def file_to_struct(path):
    with open(path, 'r') as f:
        train_file = f.readlines()
        pairs = mentions.get_mention_pairs(train_file)
        return m.make_vectors(pairs)


def train_file_to_list(file):
    """

    :param file: file name to be read
    :return: list of all lines
    """
    with open(file, "r") as f:
        return f.readlines()
