"""Entity coverage bias."""
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ccpalign.entity.topfreq import filter_docs, make_entity_set
from ccpalign.util import calc_p, generate_null_dist, IxDict, score_cossim_diff


def evaluate(label_to_docs: Dict[str, List[Counter]],
             label_to_n: Dict[str, int],
             control_labels: List[str],
             k: int,
             num_samples: int,
             random_seed: Optional[int] = None) \
        -> Tuple[List[float], pd.DataFrame, List[str]]:
    # generate entity set and dict
    print('Generating entity set...')
    entities = make_entity_set(label_to_docs, k)
    entity_dict = IxDict(entities)

    # filter docs to drop unnecessary entities
    print('Filtering dropped entities...')
    label_to_docs = {label: filter_docs(docs, entities)
                     for label, docs in label_to_docs.items()}

    # generate the null distribution
    print('Generating null distribution...')
    null_dist = generate_null_dist(
        label_to_docs=label_to_docs,
        label_to_n=label_to_n,
        control_labels=control_labels,
        ix_dict=entity_dict,
        num_samples=num_samples,
        random_seed=random_seed)

    # score
    print('Scoring...')
    label_to_score = score_cossim_diff(
        label_to_docs=label_to_docs,
        label_to_n=label_to_n,
        control_labels=control_labels,
        ix_dict=entity_dict)

    # p-values
    label_to_p = {}
    for label, score in label_to_score.items():
        p = calc_p(null_dist, score)
        label_to_p[label] = p

    # make a data frame
    df = []
    for label in label_to_score:
        df.append({
            'label': label,
            'score': label_to_score[label],
            'p': label_to_p[label],
        })
    df = pd.DataFrame(df)

    return null_dist, df, entities
