import difflib
import re

CONLL_DOC_ID_COLUMN = 0
CONLL_PART_NUM_COLUMN = 1
CONLL_WORD_SEQ = 2
CONLL_WORD_COLUMN = 3
CONLL_POS_COLUMN = 4
CONLL_PARSE_BIT_COLUMN = 5
CONLL_LEMMA_COLUMN = 6
CONLL_FRAMESET_ID_COLUMN = 7
CONLL_SENSE_COLUMN = 8
CONLL_SPEAKER_COLUMN = 9
CONLL_NAMED_COLUMN = 10


def check_usable_pairs(mention_info, i, j):
    """
    Default implementation for checking if two mentions should be considered as a pair. These pairs are not necessarily
    correferences
    :param mention_info: list of mentions
    :param i: first index
    :param j: second index
    :return: True if mention i and mention j should be paired
    """
    if mention_info[i]['id'].split('_')[0] != mention_info[j]['id'].split('_')[0]:
        return False
    if mention_info[i]['id'] == mention_info[j]['id']:
        return True
    # Skipping some indexes so not all pairs will be generated
    if j % 2 == 0 or j % 3 == 0 or j % 5 == 0 or j % 7 == 0 or j % 11 == 0:
        return False
    return True


def get_mention_pairs(train_list, increment_mention_info=None, increment_mention_pair=None,
                      use_pair=check_usable_pairs):
    """
    Builds a list of pair of mentions for the file. Each pair may or may not have a coreference. Each position
    simulates an object with the first two positions being mentions and the following are dictionaries with extra
    features

    :param train_list: list of lines in the file
    :param increment_mention_info: function to add more information to the mention
    :param increment_mention_pair: function to add more information to the mention pair
    :param use_pair: function to define if two mentions should be paired or not
    :return: list of objects
    """
    mention_info = train_dictionary(train_list, increment_mention_info)
    mention_pair_list = []
    for i in range(1, len(mention_info)):
        for j in range(0, i):
            if use_pair(mention_info, i, j):
                pair = [mention_info[i], mention_info[j],
                        {'coref': (mention_info[i]['id'] == mention_info[j]['id']) + 0}]
                mention_pair_list.append(pair)

    mention_pair_list = get_sentence_dist(mention_pair_list, train_list, increment_mention_pair)

    return mention_pair_list


def train_dictionary(train_list, fill_information=None):
    """
    Build a list of dictionaries with the information about each mention. The mentions are not yet grouped

    :param train_list: list of lines in the document
    :param fill_information: function to add more information into the cluster
    :return:
    """
    mention_info = []
    cluster_start, start_pos, cluster_end, end_pos = get_mention(train_list)
    mention_cluster = create_mention_cluster_list(cluster_start, start_pos, cluster_end, end_pos)
    for m in mention_cluster:
        mention_dict = {}
        mention_words = get_mention_words(train_list, m[1], m[2])
        mention_dict['id'] = m[0]
        mention_dict['mention_start'] = m[1]
        mention_dict['mention_end'] = m[2]
        mention_dict['mention'] = " ".join(mention_words)
        mention_dict['first_word'] = mention_words[0]
        mention_dict['last_word'] = mention_words[-1]
        mention_dict['pre_words'] = get_preceding_words(train_list, m[1])
        mention_dict['next_words'] = get_next_words(train_list, m[2])
        mention_dict['mention_sentence'] = mention_sentence(train_list, m[1])
        mention_dict['speaker'] = train_list[m[1] - 1].split()[CONLL_SPEAKER_COLUMN]

        if fill_information:
            fill_information(mention_dict, mention_words)

        mention_info.append(mention_dict)

    mention_info = sorted(mention_info, key=lambda k: k['mention_start'])
    mention_info = check_mention_contain(mention_info)
    mention_info = get_index(mention_info)
    return mention_info


def get_sentence_dist(mention_pair_list, train_list, increment_mention_pair=None):
    """
    Adds distance information about the mention pairs.
    'overlap' : True (1) if the second element overlaps the first
    'speaker' : True (1) if both mentions have the same speaker
    'mention_exact_match' : True(1) if both mentions are from the same sentence
    'mention_partial_match' : True(1) if the sentences are similar

    :param mention_pair_list: list of pair of mentions
    :param train_list: list of lines in the document
    :param increment_mention_pair: function to add more information to each pair
    :return:
    """
    for m in mention_pair_list:
        m.append({'overlap': (m[1]['overlap'] == m[0]['id']) + 0})
        m.append({'speaker': (m[1]['speaker'] == m[0]['speaker']) + 0})
        m.append({'mention_exact_match': (m[1]['mention'] == m[0]['mention']) + 0})

        seq = difflib.SequenceMatcher(None, m[0]['mention'], m[1]['mention'])
        score = seq.ratio()
        m.append({'mention_partial_match': (score > 0.6) + 0})

        if increment_mention_pair:
            increment_mention_pair(m, train_list)

    return mention_pair_list


def get_mention(train_list):
    """
    Creates four auxiliary lists with IDs and positions. These lists are not completely related to each other. It will
    simply gather the open/close parenthesis in the order it appears in the document.
    Check the formatting of a CoNLL document in the site

        * Cluster start: ID of the opening parenthesis cluster
        * Start pos: line of starting position
        * Cluster end: ID of the closing parenthesis cluster (
        * End pos: line

    :param train_list: List of lines of the file
    :return: cluster_start, start_pos, cluster_end, end_pos
    """
    cluster_start = []
    start_pos = []
    cluster_end = []
    end_pos = []
    i = 1
    for line in train_list:
        if line == '\n' or line == '-':  # Ignore empty lines
            i += 1
            continue
        part_number = line.split()[CONLL_PART_NUM_COLUMN]  # Part number column
        coref_col = line.split()[-1]  # Coref column
        for j in range(len(coref_col)):  # There can be more than one parenthesis. Tracking all of them
            if coref_col[j] == '(':
                cluster_start.append((str(part_number) + '_' + re.findall(r'\d+', coref_col[j + 1:])[0]))
                start_pos.append(i)
            if coref_col[j] == ')':
                cluster_end.append((str(part_number) + '_' + re.findall(r'\d+', coref_col[:j])[-1]))
                end_pos.append(i)
        i += 1
    return cluster_start, start_pos, cluster_end, end_pos


def create_mention_cluster_list(cluster_start, start_pos, cluster_end, end_pos):
    """
    Builds a list of mentions. One mention per item. The clusters are not yet grouped
     * First position is the id = [document_cluster]
     * Second position is the starting line
     * Third position is the ending line (counting from
    header). Use function get_mention to build the lists properly.


    :param cluster_start: List of IDs of starting cluster IDs (opening parenthesis)
    :param start_pos: List of positions for the starting cluster ID
    :param cluster_end: List of IDs for ending cluster (closing parenthesis)
    :param end_pos: List of positions for the ending cluster ID
    :return: List with three items
    """
    cluster_start_end_list = []
    for start, pos in zip(cluster_start, start_pos):  # Join ID and position
        cluster = [start, pos]
        for i in range(len(cluster_end)):  # Search where is the closing parenthesis
            if cluster_end[i] == start:
                cluster.append(end_pos[i])  # Found it. Add to the pairing
                break
        del cluster_end[i]  # Remove from original list. Will not be used anymore
        del end_pos[i]
        cluster_start_end_list.append(cluster)
    return cluster_start_end_list


def get_mention_words(train_list, pos1, pos2):
    """
    Gets the list of words between these lines
    :param train_list: lines of the document
    :param pos1: initial line
    :param pos2: final line
    :return: List of all words
    """
    mention = []
    for line_no in range(pos1 - 1, pos2):
        word = train_list[line_no].split()[CONLL_WORD_COLUMN]
        mention.append(word)
    return mention


def get_preceding_words(train_list, pos, max_words=5):
    """
    Get the previous max_words in the document (if they exists)
    :param train_list: list of lines
    :param pos: word position (numer of the line)
    :param max_words: max words to look ahead
    :return:
    """
    word_part = train_list[pos - 1].split()[CONLL_PART_NUM_COLUMN]
    num_words = 0
    word = []
    for i in range(2, pos):
        if train_list[pos - i] != '\n':
            doc_id = train_list[pos - i].split()[CONLL_DOC_ID_COLUMN]  # Checking if begin/end of document was reached
            if doc_id == '#begin' or doc_id == '#end':
                break
            part_no = train_list[pos - i].split()[CONLL_PART_NUM_COLUMN]
            if part_no == word_part:  # Will add only words for the same part
                word.append(train_list[pos - i].split()[CONLL_WORD_COLUMN])
                num_words += 1
            if num_words == max_words:
                break
    return word


def get_next_words(train_list, pos, max_words=5):
    """
    Get the next max_words in the document (if they exists)
    :param train_list: list of lines
    :param pos: word position (numer of the line)
    :param max_words: max words to look ahead
    :return:
    """
    pos = pos - 1
    word_part = train_list[pos].split()[CONLL_PART_NUM_COLUMN]
    num_words = 0
    word = []
    for i in range(1, len(train_list) - pos):
        if train_list[pos + i] != '\n':
            doc_id = train_list[pos + i].split()[CONLL_DOC_ID_COLUMN]  # Checking if begin/end of document was reached
            if doc_id == '#begin' or doc_id == '#end':
                break
            part_no = train_list[pos + i].split()[CONLL_PART_NUM_COLUMN]
            if part_no == word_part:
                word.append(train_list[pos + i].split()[CONLL_WORD_COLUMN])
                num_words += 1
            if num_words == max_words:
                break
        i += 1
    return word


def mention_sentence(train_list, pos):
    """
    Gets the sentence that contains the mention in this line
    :param train_list: list of lines
    :param pos: line of reference
    :return:
    """
    pos = pos - 1
    i = 1
    start = 0
    end = 0
    while True:
        if train_list[pos - i] == '\n':
            start = pos - i
            break
        if train_list[pos - i].split()[CONLL_DOC_ID_COLUMN] == '#begin':
            start = pos - i
            break
        i += 1
    start += 2
    i = 1
    while True:
        if train_list[pos + i] == '\n':
            end = pos + i
            break
        if train_list[pos + i].split()[CONLL_DOC_ID_COLUMN] == '#end':
            start = pos + i
            break
        i += 1
    sentence = get_mention_words(train_list, start, end)
    return " ".join(sentence)


def check_mention_contain(newlist):
    """
    Adds information about mentions containing/overlapping other mentions. The difference between contain and overlap
    will be defined by the end position. In this context, overlapping is not reflexive.
    The keys 'contained'/'overlap' will have the id of the container/overlapped mention or False if it is not contained

    :param newlist: list of dictionaries, one item per mention. Each item must have the keys:
        mention_start. mention_end and id

    :return: the modified list
    """
    for i in range(0, len(newlist)):
        start = newlist[i]['mention_start']
        end = newlist[i]['mention_end']
        for j in range(0, len(newlist)):
            c_start = newlist[j]['mention_start']
            c_end = newlist[j]['mention_end']
            if c_start == start and c_end == end:
                continue
            if c_start >= start:  # If starts after the ref
                if c_end <= end:  # and end before, it is contained by the ref
                    newlist[j]['contained'] = newlist[i]['id']
                if c_start <= end:  # if starts before the end of the ref, only overlaps
                    newlist[j]['overlap'] = newlist[i]['id']

    for k in range(0, len(newlist)):  # Filling missing info
        if 'contained' in newlist[k]:
            continue
        else:
            newlist[k]['contained'] = False
        if 'overlap' in newlist[k]:
            continue
        else:
            newlist[k]['overlap'] = False
    return newlist


def get_index(mention_info):
    """
    Adds the 'index' and 'mention_position'.
    'index' is a counter for each mention in each document
    'mention_position' is the index in a [0,1] interval (0 is the first, 1 is the last).
    :param mention_info: the mention information dictionary. Must have the 'id' key and ordered by document
    :return: the same mention_info object
    """
    doc_count = '0'
    count = 0
    i = 0
    mentions_in_each_doc = []
    for m in mention_info:
        if m['id'].split('_')[0] == doc_count:  # Keep counting while it's the same doc
            count += 1
        else:
            mentions_in_each_doc.append(count)  # save the counter
            doc_count = m['id'].split('_')[0]
            count = 1
        m['index'] = count
    mentions_in_each_doc.append(count)  # save last counter
    doc_count = '0'
    for m in mention_info:
        if m['id'].split('_')[0] == doc_count:
            m['mention_position'] = m['index'] / mentions_in_each_doc[i]
        else:
            doc_count = m['id'].split('_')[0]
            i += 1
            m['mention_position'] = m['index'] / mentions_in_each_doc[i]

    return mention_info
