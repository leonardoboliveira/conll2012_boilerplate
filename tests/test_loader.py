import unittest

from boilerplate import loader as l

TEST_FILE = "cnn_0341.gold_conll"


class LoaderTestCase(unittest.TestCase):

    def test_get_documents(self):
        docs = l.get_documents(TEST_FILE)
        self.assertEqual(1, len(docs))

    def test_train_file_to_list(self):
        lines = l.train_file_to_list(TEST_FILE)
        self.assertEqual(356, len(lines))


if __name__ == '__main__':
    unittest.main()
