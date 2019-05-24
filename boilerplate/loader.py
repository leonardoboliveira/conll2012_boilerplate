from . import mentions


def file_to_struct(path):
    with open(path, 'r') as f:
        train_file = f.readlines()
        pairs = mentions.get_mention_pairs(train_file)
        # doc_dict = document_dictionary(train_file)
        # input_vector = make_input_vector(pairs, doc_dict)
        # output_vector = make_output_vector(pairs)
        # return input_vector, output_vector


def train_file_to_list(file):
    """

    :param file: file name to be read
    :return: list of all lines
    """
    with open(file, "r") as f:
        return f.readlines()
