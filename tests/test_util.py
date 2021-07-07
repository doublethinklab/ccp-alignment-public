import unittest

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from ccpalign.entity.topfreq import make_entity_set
from ccpalign.util import *
from tests.test_entity.dummy import entity_coverage_dummy_data


class TestIxDict(unittest.TestCase):

    def test_entities_are_sorted(self):
        entities = ['b', 'a', 'd', 'c']
        ix_dict = IxDict(entities)
        for expected_ix, entity in enumerate(['a', 'b', 'c', 'd']):
            self.assertEqual(expected_ix, ix_dict[entity])

    def test_len(self):
        entities = ['b', 'a', 'd', 'c']
        ix_dict = IxDict(entities)
        self.assertEqual(4, len(ix_dict))

    def test_items(self):
        entities = ['b', 'a', 'd', 'c']
        ix_dict = IxDict(entities)
        expected = ['a', 'b', 'c', 'd']
        self.assertEqual(expected, ix_dict.items)


class TestAggregateCounts(unittest.TestCase):

    def test_sums_are_correct(self):
        counts = [
            Counter(['a', 'a', 'b']),
            Counter(['a', 'b', 'c']),
        ]
        expected = Counter(['a', 'a', 'a', 'b', 'b', 'c'])
        counts = aggregate_counts(counts)
        self.assertEqual(expected, counts)


class TestGatherControlDocs(unittest.TestCase):

    def setUp(self):
        self.label_to_docs = {
            'target': [
                {'a': 2, 'b': 1},
                {'a': 1, 'b': 1, 'e': 1},
            ],
            'domain1': [
                {'a': 1, 'b': 2},
                {'d': 1, 'e': 2},
            ],
            'domain2': [
                {'b': 3},
                {'a': 1, 'c': 2},
            ],
            'domain3': [
                {'a': 1, 'b': 1, 'c': 1},
                {'a': 1, 'b': 1, 'd': 1},
            ],
        }
        self.label_to_n = {
            'target': 10,
            'domain1': 15,
            'domain2': 20,
            'domain3': 25,
        }
        self.control_labels = ['domain1', 'domain2', 'domain3']

    def test_target_docs_included(self):
        control_docs, _ = gather_control_docs(
            label_to_docs=self.label_to_docs,
            label_to_n=self.label_to_n,
            control_labels=self.control_labels,
            subject='domain1')
        for doc in self.label_to_docs['target']:
            self.assertNotIn(doc, control_docs)

    def test_subject_docs_included(self):
        control_docs, _ = gather_control_docs(
            label_to_docs=self.label_to_docs,
            label_to_n=self.label_to_n,
            control_labels=self.control_labels,
            subject='domain1')
        for doc in self.label_to_docs['domain1']:
            self.assertNotIn(doc, control_docs)

    def test_n_summed_correctly(self):
        _, control_n = gather_control_docs(
            label_to_docs=self.label_to_docs,
            label_to_n=self.label_to_n,
            control_labels=self.control_labels,
            subject='domain1')
        self.assertEqual(45, control_n)


class TestGenerateNullDistribution(unittest.TestCase):

    def test_dummy_data(self):
        label_to_docs, label_to_n, control_labels = entity_coverage_dummy_data()
        entities = make_entity_set(label_to_docs, 3)
        entity_dict = IxDict(entities)
        dist = generate_null_dist(
            label_to_docs=label_to_docs,
            label_to_n=label_to_n,
            control_labels=control_labels,
            ix_dict=entity_dict,
            num_samples=100)
        plt.figure(figsize=(12, 8))
        sns.distplot(dist)
        plt.savefig('temp/null_dist.png')


class TestScoreCossim(unittest.TestCase):

    def test_float_returned(self):
        vec_subject = np.array([[1, 0, 3]])
        vec_target = np.array([[1, 2, 3]])
        score = score_cossim(vec_subject, vec_target)
        self.assertIsInstance(score, float)


class TestScoreCossimDiff(unittest.TestCase):

    def setUp(self):
        label_to_docs, label_to_n, control_labels = entity_coverage_dummy_data()
        entities = make_entity_set(label_to_docs, 3)
        entity_dict = IxDict(entities)
        self.label_to_score = score_cossim_diff(
            label_to_docs=label_to_docs,
            label_to_n=label_to_n,
            control_labels=control_labels,
            ix_dict=entity_dict)

    def test_target_not_included(self):
        self.assertNotIn('target', self.label_to_score)

    def test_scores_reasonable(self):
        # subject should be positive
        self.assertTrue(self.label_to_score['subject'] > 0.)
        # controls 1 and 2 should be negative
        self.assertTrue(self.label_to_score['control1'] < 0.)
        self.assertTrue(self.label_to_score['control2'] < 0.)
        # control 3 should be slightly positive
        self.assertTrue(self.label_to_score['control3'] > 0.)
        # but less than subject
        self.assertTrue(self.label_to_score['control3']
                        < self.label_to_score['subject'])


class TestShuffle(unittest.TestCase):

    def test_shuffle_randomizes_docs_and_retains_proportions(self):
        label_to_docs = {
            'subject': [1, 2],
            'control': [3, 4, 5],
        }
        sampler = Shuffle(label_to_docs)
        label_to_sample = sampler.sample(42)
        self.assertEqual(2, len(label_to_sample['subject']))
        self.assertEqual(3, len(label_to_sample['control']))
        self.assertFalse({1, 2} == set(label_to_sample['subject']))
        self.assertFalse({3, 4, 5} == set(label_to_sample['control']))


class TestVectorize(unittest.TestCase):

    def test_frequencies_correct(self):
        docs = [
            {'a': 1, 'b': 2},
            {'a': 1, 'c': 1},
        ]
        entity_dict = IxDict(['a', 'b', 'c'])
        expected = np.array([
            [0.4, 0.4, 0.2]
        ])
        vec = vectorize(docs, 5, entity_dict)
        self.assertTrue(np.array_equal(expected, vec))
