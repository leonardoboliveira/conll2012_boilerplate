import unittest

from boilerplate import mentions as m


class MentionsTestCase(unittest.TestCase):
    def test_get_mention_pairs(self):
        with open("cnn_0341.gold_conll") as f:
            lines = f.readlines()
            pairs = m.get_mention_pairs(lines)
            self.assertEqual(48, len(pairs))

    def test_mention_sentence(self):
        with open("cnn_0341.gold_conll") as f:
            lines = f.readlines()
            self.assertEqual("Reporter :", m.mention_sentence(lines, 52))

    def test_train_dictionary(self):
        with open("cnn_0341.gold_conll") as f:
            lines = f.readlines()
            info = m.train_dictionary(lines)
            self.assertEqual(48, len(info))

    def test_get_mention(self):
        with open("cnn_0341.gold_conll") as f:
            cluster_start, start_pos, cluster_end, end_pos = m.get_mention(f.readlines())
            mention_cluster = m.create_mention_cluster_list(cluster_start, start_pos, cluster_end, end_pos)
        self.assertEqual(48, len(mention_cluster))

    def test_get_mention_words(self):
        with open("cnn_0341.gold_conll") as f:
            lines = f.readlines()
            self.assertListEqual("A former FBI informant accused of being a double agent".split(),
                                 m.get_mention_words(lines, 2, 11))
            self.assertListEqual("the grand jury indictment".split(), m.get_mention_words(lines, 196, 199))

    def test_get_preceding_words(self):
        with open("cnn_0341.gold_conll") as f:
            lines = f.readlines()
            expected = ["the", "are", "Leung", "Katrina", "against"]
            self.assertListEqual(expected, m.get_preceding_words(lines, 24))

            expected = ["charges", "The", ".", "indicted", "been"]
            self.assertListEqual(expected, m.get_preceding_words(lines, 19))

            expected = ["former", "A"]
            self.assertListEqual(expected, m.get_preceding_words(lines, 4))

    def test_get_next_words(self):
        with open("cnn_0341.gold_conll") as f:
            lines = f.readlines()
            expected = ["in", "an", "alleged", "case", "of"]
            self.assertListEqual(expected, m.get_next_words(lines, 24))

            expected = ["been", "indicted", ".", "The", "charges"]
            self.assertListEqual(expected, m.get_next_words(lines, 12))

            expected = ["embarrassment", "."]
            self.assertListEqual(expected, m.get_next_words(lines, 352))


if __name__ == '__main__':
    unittest.main()
