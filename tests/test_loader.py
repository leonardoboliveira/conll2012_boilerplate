import unittest

from boilerplate import loader as l

TEST_FILE = "cnn_0341.gold_conll"


class LoaderTestCase(unittest.TestCase):

    def test_document_dictionary(self):
        docs = [["This is doc one. ", "Line two doc one. "], ["This is other doc. ", "Second Line "]]

        expected = {0: "This is doc one. Line two doc one. ", 1: "This is other doc. Second Line "}

        self.assertDictEqual(expected, l.document_dictionary(docs))

    def test_get_documents(self):
        docs = l.get_documents(l.train_file_to_list(TEST_FILE))
        self.assertEqual(1, len(docs))
        self.assertEqual("A former FBI informant accused of being a double agent has been indicted. ", docs[0][0])

    def test_train_file_to_list(self):
        lines = l.train_file_to_list(TEST_FILE)
        self.assertEqual(356, len(lines))


if __name__ == '__main__':
    unittest.main()
