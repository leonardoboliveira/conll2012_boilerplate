import unittest

from boilerplate import mock_trainer

ROOT = "."


class MyTestCase(unittest.TestCase):
    def test_create_mock(self):
        file_name, mock = mock_trainer.create_mock("{}/cnn_0341.gold_conll_in".format(ROOT),
                                                   "{}/cnn_0341.gold_conll_out".format(ROOT))
        self.assertEqual("bn/cnn/03/cnn_0341", file_name)
        self.assertEqual(35, len(mock))
        expected = (20, 21)
        expected2 = (2, 11)
        self.assertTrue(expected in mock)
        self.assertTrue(expected2 in mock[expected])

    def test_create_document(self):
        mapping = {(1, 2): [[(3, 4), (5, 6)]], (7, 8): [[(9, 10), (11, 12)]]}
        doc = mock_trainer.create_document("X", mapping)

        self.assertEqual("X", doc.name)
        self.assertEqual(2, len(doc.clusters))


if __name__ == '__main__':
    unittest.main()
