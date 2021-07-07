"""Dummy data for entity coverage bias testing."""
from collections import Counter
import random
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import binom
from sklearn.metrics.pairwise import cosine_similarity

from ccpalign.util import IxDict


def entity_coverage_dummy_data() \
        -> Tuple[Dict[str, List[Counter]], Dict[str, int], List[str]]:
    random.seed(42)
    np.random.seed(42)
    params = {
        'subject': [0.03, 0.02, 0.01],
        'target': [0.04, 0.01, 0.01],
        'control1': [0.01, 0.05, 0.05],
        'control2': [0.02, 0.06, 0.04],
        'control3': [0.03, 0.01, 0.05],
    }
    control_labels = ['control1', 'control2', 'control3']
    label_to_docs = {}
    label_to_n = {}
    for label in params:
        docs = []
        label_to_n[label] = 0
        # choose a number of docs uniformly in [200, 1000]
        n_docs = random.choice(range(200, 1001))
        for _ in range(n_docs):
            # pick two of the entities uniformly
            doc = Counter([])
            for ix in random.sample(range(3), k=2):
                # pick a document length uniformly in [100, 300]
                n_tokens = random.choice(range(100, 301))
                label_to_n[label] += n_tokens
                # generate a binom count for each
                c = binom.rvs((n_tokens,), params[label][ix])
                doc.update([str(ix)] * c)
            docs.append(doc)
        label_to_docs[label] = docs
    return label_to_docs, label_to_n, control_labels


def entity_framing_dummy_data():
    random.seed(42)
    np.random.seed(42)
    entities = ['1', '2']
    cats = ['a', 'b']
    cat_dict = IxDict(cats)
    control_labels = ['control1', 'control2', 'control3']
    params = {
        'target': {
            '1': {'a': 0.02, 'b': 0.05},
            '2': {'a': 0.07, 'b': 0.02},
        },
        'subject': {
            '1': {'a': 0.03, 'b': 0.06},
            '2': {'a': 0.06, 'b': 0.04},
        },
        'control1': {
            '1': {'a': 0.05, 'b': 0.05},
            '2': {'a': 0.04, 'b': 0.03},
        },
        'control2': {
            '1': {'a': 0.06, 'b': 0.05},
            '2': {'a': 0.07, 'b': 0.06},
        },
        'control3': {
            '1': {'a': 0.02, 'b': 0.04},
            '2': {'a': 0.03, 'b': 0.05},
        },
    }
    entity_to_label_to_docs = {}
    entity_to_label_to_n = {}
    for entity in entities:
        entity_to_label_to_docs[entity] = {}
        entity_to_label_to_n[entity] = {}
        for label in params:
            entity_to_label_to_docs[entity][label] = []
            entity_to_label_to_n[entity][label] = 0
            # choose a number of docs uniformly in [200, 1000]
            n_docs = random.choice(range(200, 1001))
            for _ in range(n_docs):
                doc = Counter([])
                for cat in cats:
                    # pick a document length uniformly in [100, 300]
                    n_tokens = random.choice(range(100, 301))
                    entity_to_label_to_n[entity][label] += n_tokens
                    # generate a binom count for the cat
                    c = binom.rvs((n_tokens,), params[label][entity][cat])
                    doc.update([cat] * c)
                entity_to_label_to_docs[entity][label].append(doc)
    return entity_to_label_to_docs, entity_to_label_to_n, control_labels, \
           cat_dict
