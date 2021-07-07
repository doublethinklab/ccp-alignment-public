""""Lexical category bias.

https://www.aclweb.org/anthology/2020.nlp4if-1.2.pdf
"""
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from recombinator.optimal_block_length import optimal_block_length
from recombinator.tapered_block_bootstrap import tapered_block_bootstrap
from tqdm import tqdm


# these are the LIWC cats used in the paper
nlp4if_cats = [
    'friend',
    'posemo',
    'negemo',
    'anx',
    'anger',
    'achieve',
    'leisure',
    'money',
    'relig',
    'death',
    'we',
    'number',
    'health',
    'sexual',
    'time'
]


def get_indicators(subject_samples: np.array,
                   target_samples: np.array,
                   control_samples: np.array) -> np.array:
    """Calculate indicator score variables from samples.

    This, like the sampling methods below, calculates over all observations
    for a single category.

    Args:
      subject_samples: numpy.array, shape (n_observations, n_samples), for the
        target source.
      target_samples: numpy.array, shape (n_observations, n_samples),
        for the authoritarian source(s).
      control_samples: numpy.array, shape (n_observations, n_samples), for the
        control source(s).

    Returns:
      numpy.array, shape (n_samples,).
    """
    # take the mean frequency over the observations
    subject_means = subject_samples.mean(axis=1)
    target_means = target_samples.mean(axis=1)
    control_means = control_samples.mean(axis=1)

    # determine the indicators of bias in the direction of the authoritarian
    # source(s)
    subject_diff = control_means - subject_means
    target_diff = control_means - target_means
    indicators = subject_diff * target_diff > 0

    # (n_samples,)
    return indicators


def mean_ci(scores: np.array, alpha: float = 0.05) \
        -> Tuple[float, float, float]:
    left = np.percentile(scores, alpha/2*100)
    right = np.percentile(scores, 100-alpha/2*100)
    return float(scores.mean()), left, right


# NOTE: this is the function for consumers to use, the rest break this down
def score(subject: pd.DataFrame,
          target: pd.DataFrame,
          control: pd.DataFrame,
          cats: List[str] = nlp4if_cats,
          n_samples: int = 1000,
          alpha: float = 0.05) -> Tuple[np.array, float, float, float]:
    """Calculate lexical category bias score.

    Args:
      subject: pandas.DataFrame, must have a `date` column, and columns for each
        category in `cats`. This data is for the subject - i.e. the data
        generator you wish to study.
      target: pandas.DataFrame, must have a `date` column, and columns
        for each category in `cats`. This data is for the authoritarian target
        of the bias.
      control: pandas.DataFrame, must have a `date` column, and columns for each
        category in `cats`. This data is for the control group.
      cats: List of strings, the names of each category to be analyzed.
      n_samples: Int, the number of bootstrap samples to take.
      alpha: Float, significance level for CI.

    Returns:
      scores: numpy.array, the bootstrapped score distribution.
      mean: Float.
      low: Float, lower bound of CI.
      high: Float, upper bound of CI.
    """
    # make sure the dfs are sorted by date
    subject = subject.sort_values(by='date', ascending=True)
    target = target.sort_values(by='date', ascending=True)
    control = control.sort_values(by='date', ascending=True)

    # sample and score
    scores = sample_scores(subject, target, control, cats, n_samples)

    # determine CI
    mean, low, high = mean_ci(scores, alpha)

    return scores, mean, low, high


def sample_scores(subject: pd.DataFrame,
                  target: pd.DataFrame,
                  control: pd.DataFrame,
                  cats: List[str],
                  n_samples: int) -> np.array:
    scores = []

    with tqdm(total=len(cats)) as pbar:
        # NOTE: the reason we do this by category is to control memory usage
        for cat in cats:
            pbar.set_description('sampling...')
            subject_samples = sample(subject, cat, n_samples)
            target_samples = sample(target, cat, n_samples)
            control_samples = sample(control, cat, n_samples)

            pbar.set_description('getting indicators...')
            indicators = get_indicators(
                subject_samples, target_samples, control_samples)
            # (1, n_samples)
            indicators = np.expand_dims(indicators, axis=0)
            scores.append(indicators)

            pbar.update()

    # (n_cats, n_samples)
    not_reduced = np.concatenate(scores, axis=0)
    # (n_samples)
    scores = not_reduced.mean(axis=0)

    # scale to be in [-1, 1] and not [0, 1]
    scores = -1 + 2 * scores

    return scores


def sample(df: pd.DataFrame, cat: str, n_samples: int) -> np.array:
    """Perform tapered block bootstrap for a category.

    Args:
      df: pandas.DataFrame, the data to sample.
      cat: String, the column name representing the category to sample.
      n_samples: Int, the number of bootstrap samples.

    Returns:
      numpy.array: of shape (n_samples, n_observations).
    """
    # get optimal block size
    b_star = optimal_block_length(df[cat].values)
    b_star = math.ceil(b_star[0].b_star_cb)

    # (n_samples, n_observations)
    samples = tapered_block_bootstrap(df[cat].values,
                                      block_length=b_star,
                                      replications=n_samples)

    return samples
