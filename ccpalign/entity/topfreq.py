"""Top frequent entities."""
from collections import Counter
from typing import Dict, List, Set

from ccpalign.util import aggregate_counts


def filter_docs(docs: List[Counter], entities: List[str]) \
        -> List[Dict]:
    filtered_docs = []
    for counts in docs:
        filtered_counts = {}
        for entity, count in counts.items():
            if entity in entities:
                filtered_counts[entity] = count
        filtered_docs.append(filtered_counts)
    return filtered_docs


def make_entity_set(label_to_docs: Dict[str, List[Counter]], k: int) \
        -> List[str]:
    entities = set()
    for label, docs in label_to_docs.items():
        counts = aggregate_counts(docs)
        label_entities = take_top_k_frequent(counts, k)
        entities = entities.union(label_entities)
    # return a list, as that is what we want to work with downstream
    return list(entities)


def take_top_k_frequent(counts: Counter, k: int) -> Set[str]:
    counts = [{'entity': e, 'count': c} for e, c in counts.items()]
    counts = list(reversed(sorted(counts, key=lambda x: x['count'])))
    counts = counts[0:k]
    entities = [x['entity'] for x in counts]
    return set(entities)
