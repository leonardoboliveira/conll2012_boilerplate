import string

import numpy as np

# model = gensim.models.KeyedVectors.load_word2vec_format(join(GLOVE_DIR, 'glove.6B.50d.w2vformat.txt'), binary=False)
model = None


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


def make_output_vector(pairs):
    output = []
    len_mentions = len(pairs)
    for m in pairs:
        output.append(m[2]['coref'])
    output = np.array(output).reshape((len_mentions, 1))
    return output


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
