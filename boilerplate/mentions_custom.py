import numpy as np
from num2words import num2words

# nlp = spacy.load('en')
nlp = None


def increment_mention_pair(m, train_list):
    count = 0
    m1 = m[0]['mention_start']
    m2 = m[1]['mention_start']
    if m1 < m2:
        for t in range(m1, m2 + 1):
            if train_list[t] == '\n':
                count += 1
    m.append({'sentence_dist_count': distance(count)})

    m.append({'mention_dist_count': distance(m[0]['index'] - m[1]['index'])})
    m.append({'head_match': (m[1]['head_word'] == m[0]['head_word']) + 0})


def increment_mention(mention_dict, mention_words):
    """
    Adding information about head word and mention type
    :param mention_dict:
    :param mention_words:
    :return: None
    """
    doc = nlp(mention_words)
    if mention_words.isdigit() or mention_words == 'its' or mention_words.lower() == 'that' or mention_words.lower() == 'this':
        mention_dict['head_word'] = ''
    else:
        if len(list(doc.noun_chunks)) > 0:
            mention_dict['head_word'] = list(doc.noun_chunks)[0].root.head.text
        else:
            mention_dict['head_word'] = ''
    mention_dict['mention_type'] = mention_type(doc, mention_words).tolist()
    mention_dict['mention_length'] = get_mention_length(mention_words)


def get_mention_length(mention_words):
    """
    Length of the mention in words (i.e. if the len(mention_words) is 2, returns "two"
    :param mention_words:
    :return:
    """
    mention_len = len(mention_words)
    len_in_words = num2words(mention_len)
    return len_in_words


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
