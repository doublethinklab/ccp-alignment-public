import unittest

from ccpalign.entity.framing import evaluate
from tests.test_entity.dummy import entity_framing_dummy_data


class TestEvaluate(unittest.TestCase):

    def test_dummy1(self):
        entity_to_label_to_docs, entity_to_label_to_n, \
        control_labels, cat_dict = \
            entity_framing_dummy_data()
        null_dist, df = evaluate(
            entity_to_label_to_docs=entity_to_label_to_docs,
            entity_to_label_to_n=entity_to_label_to_n,
            control_labels=control_labels,
            cat_dict=cat_dict,
            num_samples=100,
            random_seed=42)
        print(df)
        # TODO : an actual test case
