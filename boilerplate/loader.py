import numpy as np
from . import mentions


def file_to_struct(path):
    with open(path, 'r') as f:
        train_file = f.readlines()
        pairs = mentions.get_mention_pairs(train_file)
        doc_dict = document_dictionary(train_file)
        input_vector = make_input_vector(pairs, doc_dict)
        output_vector = make_output_vector(pairs)
    return input_vector, output_vector


def document_dictionary(documents):
    """
    Transforms the list of lists into a dictionary with a increasing id and the full text of the document
    :param documents: list of list of sentences (outer list is the document, inner list are the sentences)
    :return: dicionary {id: text}
    """
    full_doc = ''
    doc_no = 0
    output = {}
    for doc in documents:
        for sentence in doc:
            full_doc += sentence
        output[doc_no] = full_doc
        full_doc = ''
        doc_no += 1
    return output


def get_documents(train_list):
    """
    :param train_list: list of all lines in the conll file (raw info)
    :return: list of list of sentences. Each outer list represents a document, each inner list is a sentence in the
            document. The file may contain more than one document.

    """
    document = []
    part = []
    sentence = ''
    for i in range(len(train_list)):
        if train_list[i] == '\n':  # On break lines,
            part.append(sentence)  # add the sentence to a paragraph
            sentence = ''
            continue
        cols = train_list[i].split()
        if cols[0] == '#begin' or cols[0] == '#end':  # Extremes of the document
            if len(part) > 0:
                document.append(part)
                part = []
            continue
        else:
            if cols[3] == '\'s' or cols[3] == '.' or cols[3] == ',' or cols[3] == '?':
                sentence = sentence.strip() + cols[3] + ' '  # Adding punctuation to the previous sentence
            else:
                sentence += cols[3] + ' '
    return document


def train_file_to_list(file):
    """

    :param file: file name to be read
    :return: list of all lines
    """
    with open(file, "r") as f:
        return f.readlines()
