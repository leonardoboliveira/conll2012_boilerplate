def file_to_struct(path):
    pass


def get_documents(train_file):
    """
    For more information about conn files, see http://conll.cemantix.org/2012/data.html

    :param train_file: conll file name
    :return: list of all sentences in the document inside the file

    """
    train_list = train_file_to_list(train_file)  # transforming file to lines
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
