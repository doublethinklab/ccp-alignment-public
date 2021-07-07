import unittest

from ccpalign.entity.coverage import evaluate
from tests.test_entity.dummy import entity_coverage_dummy_data


class TestEvaluate(unittest.TestCase):

    def test_dummy1(self):
        label_to_docs, label_to_n, control_labels = entity_coverage_dummy_data()
        null_dist, df = evaluate(
            label_to_docs=label_to_docs,
            label_to_n=label_to_n,
            control_labels=control_labels,
            k=3,
            num_samples=100,
            random_seed=42)
        print(df)
        # TODO : an actual test case
