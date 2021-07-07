import unittest

from ccpalign.entity.topfreq import *


class TestFilterDocs(unittest.TestCase):

    def test_filtering_successful(self):
        docs = [
            Counter(['a', 'a', 'b', 'c']),
            Counter(['a', 'c', 'e', 'f']),
        ]
        entities = ['a', 'b', 'f']
        docs = filter_docs(docs, entities)
        expected = [
            {'a': 2, 'b': 1},
            {'a': 1, 'f': 1},
        ]
        self.assertEqual(expected, docs)


class TestMakeEntitySet(unittest.TestCase):

    def test_output_as_expected(self):
        label_to_docs = {
            'subject': [
                Counter(['a', 'b', 'b']),
                Counter(['e', 'e', 'd']),
            ],
            'target': [
                Counter(['a', 'a', 'b']),
                Counter(['a', 'b', 'e']),
            ],
            'control': [
                Counter(['b', 'b', 'b']),
                Counter(['c', 'c', 'a']),
            ],
        }
        entities = make_entity_set(label_to_docs, 2)
        expected = ['a', 'b', 'c', 'e']
        self.assertEqual(expected, list(sorted(entities)))


class TestTakeTopKFrequent(unittest.TestCase):

    def test_entities_are_sorted_correctly(self):
        counts = Counter(['a', 'b', 'b', 'b', 'd', 'e', 'e'])
        entities = take_top_k_frequent(counts, 2)
        expected = {'b', 'e'}
        self.assertSetEqual(expected, entities)
