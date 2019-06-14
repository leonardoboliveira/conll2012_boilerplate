import string

import numpy as np
from tqdm import tqdm

__GLOVE_FILE_NAME__ = 'extra_files/glove.6B.50d.w2vformat.txt'


class Features:
    """
    This class represents a list of features. In custom implementations it can be completely replaced
    """

    def __init__(self, mention_avg, antecedent_avg, mention_features, antecedent_features, pair_features):
        self.mention_avg = mention_avg
        self.antecedent_avg = antecedent_avg
        self.mention_features = mention_features
        self.antecedent_features = antecedent_features
        self.pair_features = pair_features

    def to_vector(self):
        """
        Transforms the current object into a vector. This must be consistent in all sets
        :return:
        """
        return np.concatenate([self.mention_avg, self.antecedent_avg, self.mention_features, self.antecedent_features,
                               self.pair_features])


class FeatureMapper:
    """
        This class is used to map the words into vectors. Also has some extra methods for features used in this
        implementation
    """
    _default_model = None

    def __init__(self, word2vec, train_list):
        """
        Creates a feature mapping using a word2vec dictionary. Originally designed to be used with GloVe. See
        https://nlp.stanford.edu/projects/glove/ for more info.

        :param word2vec: dictionary with <word,[vector of length 50]>
        """
        self.model = word2vec
        # Used to remove punctuation of the words
        self.table = str.maketrans({key: None for key in string.punctuation})
        self.doc_dict = document_dictionary(train_list)

    @classmethod
    def get_default_model(cls):
        """
        Creates the default model implementation based on the GloVe word2vector
        :return:
        """
        if FeatureMapper._default_model is None:
            import gensim
            FeatureMapper._default_model = gensim.models.KeyedVectors.load_word2vec_format(__GLOVE_FILE_NAME__,
                                                                                           binary=False)
        return FeatureMapper._default_model

    def get_vector(self, word):
        """
        Transforms the word into a vector of 50 positions. It will use the model of the class. If the word is not found,
        a 0 vector will be returned. Punctuations are removed
        :param word:
        :return: np.array(50,1)
        """
        word = word.lower()
        if len(word) > 1:
            word = word.translate(self.table)  # This will remove punctuation

        try:  # "easier to ask for forgiveness than permission
            vec = self.model[word]
        except KeyError:
            vec = np.zeros((50, 1))

        return vec.reshape((50, 1))

    def get_average_vector(self, word_list):
        """
        Averages all the words in the list. Each word will be translated into a vector first
        :param word_list:
        :return: np.array(50,1)
        """
        final_sum = np.zeros((50, 1))
        i = 0
        for i in range(0, len(word_list)):
            final_sum += self.get_vector(word_list[i])
        average_vector = final_sum / (i + 1)
        return average_vector

    def calculate_docs_average(self):
        """
        Using the document dictionary of the class and the model, averages all words in each document
        :return: list of all averages (same order as the document dictionary)
        """
        doc_avg = []
        for d in self.doc_dict:
            doc_avg.append(self.get_average_vector(self.doc_dict[d].split()))
        return doc_avg

    def make_input_vector(self, pairs):
        """
        Builds the input feature vector from the mention pairs
        :param pairs: mention pais
        :return: np.array of all features (one line per pair)
        """
        docs_avg = self.calculate_docs_average()
        input_feature_list = []
        for p in tqdm(pairs, desc="features"):
            # Build a Features object
            input_feature_vector = self.make_pair_feature(docs_avg, p)
            # Saves into a list the vector representation
            input_feature_list.append(input_feature_vector.to_vector())

        return input_feature_list

    def make_pair_feature(self, docs_avg, pair):
        """
        Builds the features for one pair
        :param docs_avg: word average vector for all documents
        :param pair: mention pair
        :return: list of list of features. There are 5 groups (each one in a position of the vector):
            - antecedent avg
            - antecedent features
            - mention avg
            - mention features
            - pair features
        """
        mention_avg = self.get_average_vector(pair.mention1.words)
        antecedent_avg = self.get_average_vector(pair.mention2.words)
        mention_features = self.get_mention_features(pair.mention1, docs_avg)
        antecedent_features = self.get_mention_features(pair.mention2, docs_avg)
        pair_features = get_pair_features(pair)

        return Features(mention_avg, antecedent_avg, mention_features, antecedent_features, pair_features)

    def get_mention_features(self, mention, doc_average):
        """
        Buils a vector with all the features of a single mention
        :param mention:
        :param doc_average: list of document average word
        :return: np.array with all features for that mention
        """
        mention_length = self.get_vector(mention.mention_length)
        mention_type = np.array(mention.mention_type).reshape((4, 1))
        mention_position = np.array(mention.mention_position).reshape((1, 1))

        # p: previous, n: next, w: words, a: average, s: sentence
        first_w = self.get_vector(mention.first_word)
        last_w = self.get_vector(mention.last_word)

        # Contained
        mention_contain = np.ones((1, 1)) if mention.contained else np.zeros((1, 1))
        # Previous words
        mention_p_w1, mention_p_w2 = self.get_vector_if_defined(mention.pre_words, [0, 1])
        # Next words
        mention_n_w1, mention_n_w2 = self.get_vector_if_defined(mention.next_words, [0, 1])
        # Previous words Average
        mention_p_w_a = self.get_average_vector(mention.pre_words)
        # Next words Average
        mention_n_w_a = self.get_average_vector(mention.next_words)
        # Mention Sentence Average
        mention_s_a = self.get_average_vector(mention.words)

        # Extra info
        doc_id = mention.doc_id
        doc_avg = doc_average[int(doc_id)]

        features = np.concatenate((first_w, last_w, mention_p_w1, mention_p_w2, mention_p_w_a, mention_n_w1,
                                   mention_n_w2, mention_n_w_a, mention_s_a, mention_length, mention_type,
                                   mention_position, mention_contain, doc_avg))
        return features

    def get_vector_if_defined(self, word_list, indexes):
        """
        Auxiliary function that returns the vector representation of the words in the corresponding indexes. If the
        index is greater than the list length, returns a 0-vector
        :param word_list: list of all words
        :param indexes: list of indexes of the list to be used
        :return: list of vectors that represents the words. Has the same length as the indexes list
        """
        output = []
        for idx in indexes:
            output.append(self.get_vector(word_list[idx]) if idx < len(word_list) else np.zeros((50, 1)))
        return output


def make_vectors(pairs, mapper=None, train_list=None):
    """
    This is the main method. Other implementations should replace this.
    From a mention pair list, it returns the input and output vectors that will be fed into the net
    :param pairs:
    :param mapper: Custom vector mapper, if not informed, will use default FeatureMapper
    :param train_list: list of all files. necessary if no mapper is informed
    :return: input_vector,output_vector
    """
    if mapper is None:
        mapper = FeatureMapper(FeatureMapper.get_default_model(), train_list)
    return mapper.make_input_vector(pairs), make_output_vector(pairs)


def make_output_vector(pairs):
    """
    Builds the output vector from the mentions pairs
    :param pairs:
    :return: np.array(len(pairs),1)
    """
    output = [p.coref + 0 for p in pairs]
    output = np.array(output).reshape((len(output), 1))
    return output


def merge_document_text(documents):
    """
    Merges all the text of a document into one key
    :param documents: list of list of lines. Outer list are the documents, inner lists are the lines for the document
    :return: dictionary<doc_id, text>
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


def get_pair_features(pair):
    """
    Builds a vector of features for this mention pair. Currently, it has the following features:
    * mention distance
    * sentence distance
    * overlap
    * speaker
    * head match
    * mention exact match
    * mention partial match

    :param pair:
    :return: vector of doubles
    """
    # distance features
    mention_dist = np.array(pair.mention_dist_count).reshape((10, 1))
    s_dist = np.array(pair.sentence_dist_count).reshape((10, 1))
    overlap = np.array(pair.overlap).reshape((1, 1))

    # speaker feature
    speaker = np.array(pair.speaker).reshape((1, 1))

    # string matching features
    head_match = np.array(pair.head_match).reshape((1, 1))
    mention_exact_match = np.array(pair.mention_exact_match).reshape((1, 1))
    mention_partial_match = np.array(pair.mention_partial_match).reshape((1, 1))

    pair_features = np.concatenate(
        (mention_dist, s_dist, overlap, speaker, head_match, mention_exact_match, mention_partial_match))

    return pair_features


def document_dictionary(train_file):
    """
    Transforms the list lines of the file into a dictionary with document and text
    :param train_file: list of lines in the document
    :return: dicionary {id: text}
    """
    documents = get_documents(train_file)
    return merge_document_text(documents)
