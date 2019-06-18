import unittest

from boilerplate import saver as s


class MyTestCase(unittest.TestCase):
    def test_lineinfo(self):
        info = s.LineInfo()
        info.add_operation(1, 1)
        info.add_operation(2, 1)
        info.add_operation(1, -1)
        info.add_operation(3, -1)

        self.assertEqual("(1)|(2|3)", str(info))

    def test_document(self):
        doc = s.Document("doc_name")
        doc.add_cluster({1: 2, 4: 4, 5: 10})
        doc.add_cluster({4: 7})

        expected = ["(1", "1)", "-", "(1)|(2", "(1", "-", "2)", "-", "-", "1)"]
        received = [doc.get_line_operations(i + 1) for i in range(len(expected))]

        self.assertListEqual(expected, received)

    if __name__ == '__main__':
        unittest.main()
