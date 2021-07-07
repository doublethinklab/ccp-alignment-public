from collections import Counter
import random
from typing import Dict, List, Optional, Tuple, Union

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


class IxDict:

    def __init__(self, items: List[str]):
        items = list(sorted(items))
        self.item_to_ix = dict(zip(items, range(len(items))))
        self.ix_to_item = {ix: item for item, ix in self.item_to_ix.items()}

    def __getitem__(self, item: Union[int, str]):
        if isinstance(item, int):
            return self.ix_to_item[item]
        elif isinstance(item, str):
            return self.item_to_ix[item]
        else:
            raise ValueError(type(item))

    def __len__(self):
        return len(self.item_to_ix)

    @property
    def items(self):
        return list(self.item_to_ix.keys())


class Shuffle:

    def __init__(self, label_to_docs: Dict[str, List]):
        self.label_to_docs = label_to_docs
        self.label_to_n = {label: len(docs) for label, docs
                           in label_to_docs.items()}
        docs = []
        for label, label_docs in label_to_docs.items():
            for doc in label_docs:
                docs.append({
                    'label': label,
                    'doc': doc,
                })
        self.df = pd.DataFrame(docs)

    def sample(self, random_seed: Optional[int] = None):
        if random_seed:
            random.seed(random_seed)
        label_to_sample = {}
        self.df['label'] = np.random.permutation(self.df.label.values)
        for label in self.label_to_docs:
            docs = list(self.df[self.df.label == label].doc.values)
            label_to_sample[label] = docs
        return label_to_sample


def aggregate_counts(counts: List[Counter]) -> Counter:
    aggregate = counts[0]
    for c in counts[1:]:
        aggregate += c
    return aggregate


def calc_p(null_dist: List[float], score: float) -> float:
    # one-sided
    n = len(null_dist)
    mu = np.mean(null_dist)
    if score < mu:
        return sum(1 for x in null_dist if x < score) / n
    else:  # score > mu
        return sum(1 for x in null_dist if x > score) / n


def gather_control_docs(label_to_docs: Dict[str, List[Dict]],
                        label_to_n: Dict[str, int],
                        control_labels: List[str],
                        subject: str) -> Tuple[List[Dict], int]:
    control_docs = []
    control_n = 0
    for control_label in control_labels:
        if control_label == subject:
            continue
        control_label_docs = label_to_docs[control_label]
        control_docs += control_label_docs
        control_n += label_to_n[control_label]
    return control_docs, control_n


def generate_null_dist(label_to_docs: Dict[str, List[Counter]],
                       label_to_n: Dict[str, int],
                       control_labels: List[str],
                       ix_dict: IxDict,
                       num_samples: int,
                       random_seed: Optional[int] = None) -> List[float]:
    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)

    sampler = Shuffle(label_to_docs)

    # sample, vectorize, score
    scores = []
    print(f'Sampling {num_samples} times to generate null distribution...')
    with tqdm(total=num_samples) as pbar:
        for _ in range(num_samples):
            label_to_samples = sampler.sample()
            label_to_score = score_cossim_diff(
                label_to_docs=label_to_samples,
                label_to_n=label_to_n,
                control_labels=control_labels,
                ix_dict=ix_dict)
            scores += list(label_to_score.values())
            pbar.update()

    return scores


def score_cossim(subject, target):
    cossim = cosine_similarity(subject, target)
    cossim = cossim[0]
    return float(cossim)


def score_cossim_diff(label_to_docs: Dict[str, List[Dict]],
                      label_to_n: Dict[str, int],
                      control_labels: List[str],
                      ix_dict: IxDict) -> Dict[str, float]:
    label_to_score = {}
    target = vectorize(
        docs=label_to_docs['target'],
        n=label_to_n['target'],
        ix_dict=ix_dict)
    for label, docs in label_to_docs.items():
        if label == 'target':
            continue
        subject = vectorize(
            docs=docs,
            n=label_to_n[label],
            ix_dict=ix_dict)
        control_docs, control_n = gather_control_docs(
            label_to_docs, label_to_n, control_labels, label)
        control = vectorize(control_docs, control_n, ix_dict)
        score = score_cossim(subject, target) - score_cossim(control, target)
        label_to_score[label] = score
    return label_to_score


def vectorize(docs: List[Dict], n: int, ix_dict: IxDict) -> np.array:
    vec = np.zeros((1, len(ix_dict)))
    if n > 0:
        for doc in docs:
            for entity, count in doc.items():
                entity_ix = ix_dict[entity]
                vec[0, entity_ix] += count
        vec /= n
    return vec
