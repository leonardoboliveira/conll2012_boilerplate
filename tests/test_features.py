import unittest

from boilerplate import features as f


class FeaturesTestCase(unittest.TestCase):
    def test_get_average_vector(self):
        v = f.get_average_vector("This is doc one. Line two doc one. ".split())
        print("V:{}".format(v))

    def test_get_vector(self):
        v = f.get_vector("Teste.")
        print("V:{}".format(v))


if __name__ == '__main__':
    unittest.main()
