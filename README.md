# Modularized, reusable implementations of media alignment algorithms

## Quickstart

1. Build the image in the `Dockerfile` - can run `./build_image.sh`.
2. Spin up a jupyter notebook with `./jupyter.sh`. Place your data in a folder 
   named `data` in the root of this project. The `jupyter.sh` script will 
   hook it up as a volume.
3. Run the algorithms in `ccpalign` as desired.

## Algorithms

### Lexical Category Alignment

Example:

```python
from ccpalign.lexcat import nlp4if_cats, score

_, score, ci_lower, ci_upper = score(
    subject=df_subject,
    target=df_target,
    control=df_control,
    cats=nlp4if_cats,
    n_samples=10000,
    alpha=0.05)
```

In our terminology, the `subject` is the media under study - i.e. that you 
want to score. The `target` is the alignment target, e.g. PRC or Russian state 
media. The `control` is a baseline, such as the mainstream media of the country 
the target comes from. These are `pandas.DataFrame`s, and need to have a `date` 
column, and columns with frequencies for each lexical category specific in 
`cats`. Each row represents a document - i.e. news article.

Note: subsequent algorithmic development suggests a vectorized calculation is 
far better, which we will update here in the future.

### Entity Coverage Alignment

```python
from ccpalign.entity.coverage import evaluate

_, scores, _ = evaluate(
    label_to_docs=label_to_docs,
    label_to_n=label_to_n,
    control_labels=control_labels,
    k=1000,
    num_samples=10000,
    random_seed=42)
```

In this module, we represent each document as a `collections.Counter` counting 
entity mentions. The `label_to_docs` variable needs to hold a dictionary 
mapping labels (e.g. media names) to lists of docs - i.e., lists of 
`Counter`s. Crucially, this dictionary must have a label for `subject` and 
`target`, following the same naming conventions described above for lexical 
category alignment. It must also have keys for each label in `control_labels`, 
which defines which labels belong to the set of control media. `k` specifies 
how many entities to take from each media (starting with the most frequently 
mentioned). `num_samples` controls how many randomizations to perform, and 
`random_seed` controls reproducibility.
