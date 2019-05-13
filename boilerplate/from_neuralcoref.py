import re

import numpy as np


def train_network_data(path):
    train_file = open(path, 'r')
    doc_dict = document_dictionary(train_file)
    train_file = open(path, 'r')
    pairs = get_mention_pairs(train_file)
    input_vector = make_input_vector(pairs, doc_dict)
    output_vector = make_output_vector(pairs)
    return input_vector, output_vector


def document_dictionary(train_file):
    documents = get_documents(train_file)
    doc_sent = ''
    doc_no = 0
    doc_dict = {}
    for document in documents:
        for part in document:
            doc_sent += part
        doc_dict[doc_no] = doc_sent
        doc_sent = ''
        doc_no += 1
    return doc_dict


def get_mention_pairs(train_file):
    mention_info = train_dictionary(train_file)
    mention_pair_list = []
    for i in range(1, len(mention_info)):
        for j in range(0, i):
            pair = []
            if mention_info[i]['id'].split('_')[0] == mention_info[j]['id'].split('_')[0]:
                pair.append(mention_info[i])
                pair.append(mention_info[j])
                if mention_info[i]['id'] == mention_info[j]['id']:
                    pair.append({'coref': 1})
                else:
                    if j % 2 == 0 or j % 3 == 0 or j % 5 == 0 or j % 7 == 0 or j % 11 == 0:
                        continue
                    else:
                        pair.append({'coref': 0})
                mention_pair_list.append(pair)

    mention_pair_list = get_sentence_dist(mention_pair_list, train_file)

    return mention_pair_list


def make_input_vector(pairs, doc_dict):
    feature_input = make_feature_input(pairs, doc_dict)
    len_f_input = len(feature_input)
    input_ = []
    for f_input in feature_input:
        con = np.concatenate((f_input[0], f_input[1], f_input[2], f_input[3], f_input[4]))
        input_.append(con)
        del con
    return input_


def make_output_vector(pairs):
    output = []
    len_mentions = len(pairs)
    for m in pairs:
        output.append(m[2]['coref'])
    output = np.array(output).reshape((len_mentions, 1))
    return output


def get_documents(train_file):
    train_list = train_file_to_list(train_file)
    document = []
    part = []
    sentence = ''
    for i in range(len(train_list)):
        if train_list[i] == '\n':
            part.append(sentence)
            sentence = ''
            continue
        cols = train_list[i].split()
        if cols[0] == '#begin' or cols[0] == '#end':
            if len(part) > 0:
                document.append(part)
                part = []
            continue
        else:
            if cols[3] == '\'s' or cols[3] == '.' or cols[3] == ',' or cols[3] == '?':
                sentence = sentence.strip() + cols[3] + ' '
            else:
                sentence += cols[3] + ' '
    return document


def train_dictionary(train_file):
    mention_info = []
    train_list = train_file_to_list(train_file)
    cluster_start, start_pos, cluster_end, end_pos = get_mention(train_list)
    mention_cluster = create_mention_cluster_list(cluster_start, start_pos, cluster_end, end_pos)
    for m in mention_cluster:
        mention_dict = {}
        mention_words = get_mention_words(train_list, m[1], m[2])
        doc = nlp(mention_words)
        mention_dict['id'] = m[0]
        mention_dict['mention_start'] = m[1]
        mention_dict['mention_end'] = m[2]
        mention_dict['mention'] = mention_words
        mention_dict['first_word'] = mention_words.split()[0]
        mention_dict['last_word'] = mention_words.split()[-1]
        if mention_words.isdigit() or mention_words == 'its' or mention_words.lower() == 'that' or mention_words.lower() == 'this':
            mention_dict['head_word'] = ''
        else:
            if len(list(doc.noun_chunks)) > 0:
                mention_dict['head_word'] = list(doc.noun_chunks)[0].root.head.text
            else:
                mention_dict['head_word'] = ''
        mention_dict['pre_words'] = get_preceding_words(train_list, m[1])
        mention_dict['next_words'] = get_next_words(train_list, m[2])
        mention_dict['mention_sentence'] = mention_sentence(train_list, m[1])
        mention_dict['mention_type'] = mention_type(doc, mention_words).tolist()
        mention_dict['mention_length'] = get_mention_length(mention_words)
        mention_dict['speaker'] = train_list[m[1] - 1].split()[9]
        mention_info.append(mention_dict)

    mention_info = sorted(mention_info, key=lambda k: k['mention_start'])
    mention_info = check_mention_contain(mention_info)
    mention_info = get_index(mention_info)
    return mention_info


def get_sentence_dist(mention_pair_list, train_file):
    train_list = train_file_to_list(train_file)
    for m in mention_pair_list:
        count = 0
        m1 = m[0]['mention_start']
        m2 = m[1]['mention_start']
        if m1 < m2:
            for t in range(m1, m2 + 1):
                if train_list[t] == '\n':
                    count += 1
        seq = difflib.SequenceMatcher(None, m[0]['mention'], m[1]['mention'])
        score = seq.ratio()
        m.append({'sentence_dist_count': distance(count)})
        m.append({'mention_dist_count': distance(m[0]['index'] - m[1]['index'])})
        if m[1]['overlap'] == m[0]['id']:
            m.append({'overlap': 1})
        else:
            m.append({'overlap': 0})
        if m[1]['speaker'] == m[0]['speaker']:
            m.append({'speaker': 1})
        else:
            m.append({'speaker': 0})
        if m[1]['head_word'] == m[0]['head_word']:
            m.append({'head_match': 1})
        else:
            m.append({'head_match': 0})
        if m[1]['mention'] == m[0]['mention']:
            m.append({'mention_exact_match': 1})
        else:
            m.append({'mention_exact_match': 0})
        if score > 0.6:
            m.append({'mention_partial_match': 1})
        else:
            m.append({'mention_partial_match': 0})
    return mention_pair_list


def make_feature_input(pairs, doc_dict):
    docs_avg = calculate_docs_average(doc_dict)
    input_feature_list = []
    i = 0
    for m in pairs:
        i += 1
        input_feature_vector = []
        mention_avg = get_average_vector(m[0]['mention'].split())
        antecedent_avg = get_average_vector(m[1]['mention'].split())
        mention_features = get_mention_features(m[0], docs_avg)
        antecedent_features = get_mention_features(m[1], docs_avg)
        pair_features = get_pair_features(m)

        input_feature_vector.append(antecedent_avg)
        input_feature_vector.append(antecedent_features)
        input_feature_vector.append(mention_avg)
        input_feature_vector.append(mention_features)
        input_feature_vector.append(pair_features)
        input_feature_list.append(input_feature_vector)

    return input_feature_list


def train_file_to_list(file):
    train_list = []
    for line in file:
        train_list.append(line)
    return train_list


def get_mention(train_list):
    cluster_start = []
    start_pos = []
    cluster_end = []
    end_pos = []
    i = 1
    for line in train_list:
        if line == '\n' or line == '-':
            i += 1
            continue
        part_number = line.split()[1]
        coref_col = line.split()[-1]
        for j in range(len(coref_col)):
            if coref_col[j] == '(':
                cluster_start.append((str(part_number) + '_' + re.findall(r'\d+', coref_col[j + 1:])[0]))
                start_pos.append(i)
            if coref_col[j] == ')':
                cluster_end.append((str(part_number) + '_' + re.findall(r'\d+', coref_col[:j])[-1]))
                end_pos.append(i)
        i += 1
    return cluster_start, start_pos, cluster_end, end_pos


def create_mention_cluster_list(cluster_start, start_pos, cluster_end, end_pos):
    cluster_start_end_list = []
    for start, pos in zip(cluster_start, start_pos):
        cluster = [start, pos]
        for i in range(len(cluster_end)):
            if cluster_end[i] == start:
                cluster.append(end_pos[i])
                break
        del cluster_end[i]
        del end_pos[i]
        cluster_start_end_list.append(cluster)
    return cluster_start_end_list


def get_mention_words(train_file_as_list, pos1, pos2):
    mention = ''
    for line_no in range(pos1 - 1, pos2):
        word = train_file_as_list[line_no].split()[3]
        # if word == '\'s' or word == ',' or word == '.':
        #    mention = mention.strip() + word + ' '
        # else:
        mention += word + ' '
    return mention.strip()


def get_preceding_words(list, pos):
    word_part = list[pos - 1].split()[1]
    i = 2
    num_words = 0
    word = []
    while (True):
        if list[pos - i] != '\n':
            if list[pos - i].split()[0] == '#begin' or list[pos - i].split()[0] == '#end':
                break
            part_no = list[pos - i].split()[1]
            if part_no == word_part:
                word.append(list[pos - i].split()[3])
                num_words += 1
            if num_words == 5:
                break
        i += 1
    return word


def get_next_words(list, pos):
    pos = pos - 1
    word_part = list[pos].split()[1]
    i = 1
    num_words = 0
    word = []
    while (True):
        if list[pos + i] != '\n':
            if list[pos + i].split()[0] == '#begin' or list[pos + i].split()[0] == '#end':
                break
            part_no = list[pos + i].split()[1]
            if part_no == word_part:
                word.append(list[pos + i].split()[3])
                num_words += 1
            if num_words == 5:
                break
        i += 1
    return word


def mention_sentence(train_list, pos):
    pos = pos - 1
    i = 1
    start = 0
    end = 0
    while (True):
        if train_list[pos - i] == '\n':
            start = pos - i
            break
        if train_list[pos - i].split()[0] == '#begin':
            start = pos - i
            break
        i += 1
    start += 2
    i = 1
    while (True):
        if train_list[pos + i] == '\n':
            end = pos + i
            break
        i += 1
    sentence = get_mention_words(train_list, start, end)
    return sentence


# pronoun: [1, 0, 0, 0]
# proper:  [0, 1, 0, 0]
# nominal(common noun): [0, 0, 1, 0]
# list:    [0, 0, 0, 1]
def mention_type(doc, mention):
    # pos 0: pronoun, pos 1: proper noun, pos 2: common noun
    token_type = [0, 0, 0]
    for token in doc:
        if token.pos_ == 'PRON':
            token_type[0] += 1
        elif token.pos_ == 'PROPN':
            token_type[1] += 1
        elif token.pos_ == 'NOUN':
            token_type[2] += 1
    m = max(token_type)
    a = [i for i, j in enumerate(token_type) if j == m]
    is_dominant = m >= len(mention.split()) / 2
    if is_dominant:
        if a[0] == 0:
            return np.array([1, 0, 0, 0])
        if a[0] == 1:
            return np.array([0, 1, 0, 0])
        if a[0] == 2:
            return np.array([0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 1])


def get_mention_length(mention):
    mention_words = mention.split()
    mention_len = len(mention_words)
    len_in_words = num2words(mention_len)
    return len_in_words


def check_mention_contain(newlist):
    for i in range(0, len(newlist)):
        start = newlist[i]['mention_start']
        end = newlist[i]['mention_end']
        for j in range(0, len(newlist)):
            c_start = newlist[j]['mention_start']
            c_end = newlist[j]['mention_end']
            if c_start == start and c_end == end:
                continue
            if c_start >= start and c_end <= end:
                newlist[j]['contained'] = newlist[i]['id']
            if c_start >= start and c_start <= end:
                newlist[j]['overlap'] = newlist[i]['id']

    for k in range(0, len(newlist)):
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
    doc_count = '0'
    count = 0
    i = 0
    mentions_in_each_doc = []
    for m in mention_info:
        if m['id'].split('_')[0] == doc_count:
            count += 1
        else:
            mentions_in_each_doc.append(count)
            doc_count = m['id'].split('_')[0]
            count = 1
        m['index'] = count
    mentions_in_each_doc.append(count)
    doc_count = '0'
    for m in mention_info:
        if m['id'].split('_')[0] == doc_count:
            m['mention_position'] = m['index'] / mentions_in_each_doc[i]
        else:
            doc_count = m['id'].split('_')[0]
            i += 1
            m['mention_position'] = m['index'] / mentions_in_each_doc[i]

    return mention_info


def distance(a):
    d = np.zeros((10))
    d[a == 0, 0] = 1
    d[a == 1, 1] = 1
    d[a == 2, 2] = 1
    d[a == 3, 3] = 1
    d[a == 4, 4] = 1
    d[(5 <= a) & (a < 8), 5] = 1
    d[(8 <= a) & (a < 16), 6] = 1
    d[(16 <= a) & (a < 32), 7] = 1
    d[(a >= 32) & (a < 64), 8] = 1
    d[a >= 64, 9] = 1
    return d.tolist()


def calculate_docs_average(doc_dict):
    doc_avg = []
    for d in doc_dict:
        doc_avg.append(get_average_vector(doc_dict[d].split()))
    return doc_avg


def get_average_vector(word_list):
    sum = np.zeros((50, 1))
    for i in range(0, len(word_list)):
        sum += get_vector(word_list[i])
    average_vector = sum / (i + 1)
    return average_vector


# p: previous, n: next, w: words, a: average, s: sentence
def get_mention_features(mention, doc_average):
    features = []
    # head_w = get_vector(mention['head_word'])
    first_w = get_vector(mention['first_word'])
    last_w = get_vector(mention['last_word'])
    mention_length = get_vector(mention['mention_length'])
    mention_type = np.array(mention['mention_type']).reshape((4, 1))
    mention_position = np.array(mention['mention_position']).reshape((1, 1))
    if mention['contained'] == False:
        mention_contain = np.zeros((1, 1))
    else:
        mention_contain = np.ones((1, 1))
    if len(mention['pre_words']) > 0:
        mention_p_w1 = get_vector(mention['pre_words'][0])
    else:
        mention_p_w1 = np.zeros((50, 1))
    if len(mention['pre_words']) > 1:
        mention_p_w2 = get_vector(mention['pre_words'][1])
    else:
        mention_p_w2 = np.zeros((50, 1))
    if len(mention['next_words']) > 0:
        mention_n_w1 = get_vector(mention['next_words'][0])
    else:
        mention_n_w1 = np.zeros((50, 1))
    if len(mention['next_words']) > 1:
        mention_n_w2 = get_vector(mention['next_words'][1])
    else:
        mention_n_w2 = np.zeros((50, 1))
    if len(mention['pre_words']) > 0:
        mention_p_w_a = get_average_vector(mention['pre_words'])
    else:
        mention_p_w_a = np.zeros((50, 1))
    if len(mention['next_words']) > 0:
        mention_n_w_a = get_average_vector(mention['next_words'])
    else:
        mention_n_w_a = np.zeros((50, 1))
    mention_s_a = get_average_vector(mention['mention_sentence'].split())
    doc_id = mention['id'].split('_')[0]
    doc_avg = doc_average[int(doc_id)]

    features = np.concatenate((first_w, last_w, mention_p_w1, mention_p_w2, mention_p_w_a, mention_n_w1, mention_n_w2,
                               mention_n_w_a, mention_s_a, mention_length, mention_type, mention_position,
                               mention_contain, doc_avg))
    return features


def get_vector(word):
    table = str.maketrans({key: None for key in string.punctuation})
    word = word.lower()
    if len(word) > 1:
        word = word.translate(table)
    try:
        vec = model[word]
    except:
        vec = np.zeros((50, 1))
    return vec.reshape((50, 1))


def get_pair_features(feature_list):
    # distance features
    mention_dist = np.array(feature_list[4]['mention_dist_count']).reshape((10, 1))
    s_dist = np.array(feature_list[3]['sentence_dist_count']).reshape((10, 1))
    overlap = np.array(feature_list[5]['overlap']).reshape((1, 1))

    # speaker feature
    speaker = np.array(feature_list[6]['speaker']).reshape((1, 1))

    # string matching features
    head_match = np.array(feature_list[7]['head_match']).reshape((1, 1))
    mention_exact_match = np.array(feature_list[8]['mention_exact_match']).reshape((1, 1))
    mention_partial_match = np.array(feature_list[9]['mention_partial_match']).reshape((1, 1))

    pair_features = np.concatenate(
        (mention_dist, s_dist, overlap, speaker, head_match, mention_exact_match, mention_partial_match))

    return pair_features
