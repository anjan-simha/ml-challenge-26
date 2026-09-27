# Business Entity Resolution --- Proposed ML Solution

## 1. Objective

This document describes the proposed solution for the Business Entity
Resolution Challenge.

The task is to identify which records in **Source 2** and **Source 3**
refer to the same real-world business as each record in **Source 1**.

Source 1 is the deduplicated reference source. A Source 1 entity may
have:

-   zero matching records,
-   one matching record, or
-   multiple matching records across Source 2 and Source 3.

The training data contains ground-truth matching labels. The test data
does not contain ground truth, so the pipeline must generate the final
matches for every Source 1 test entity.

------------------------------------------------------------------------

## 2. Why a Multi-Stage Solution Is Required

The datasets are large, and business records contain noisy and
inconsistent information.

Typical variations include:

-   abbreviations such as `Pvt` vs `Private` and `Ltd` vs `Limited`
-   punctuation differences
-   spelling errors and typos
-   word-order changes
-   transliteration variations
-   partial addresses
-   missing address components
-   address abbreviations such as `Rd` vs `Road`
-   different address component ordering
-   landmark-based addresses

Because of these variations, exact string matching is insufficient.

At the same time, comparing every Source 1 record with every Source 2
and Source 3 record would be computationally impractical at large scale.

Therefore, the proposed architecture separates the problem into:

1.  Data preprocessing and normalization
2.  Traditional blocking
3.  Embedding-based candidate retrieval
4.  Candidate union and deduplication
5.  Detailed pairwise feature engineering
6.  Supervised ML matching
7.  Thresholding and singleton handling
8.  Output validation

------------------------------------------------------------------------

# 3. Proposed Architecture

``` text
                         TRAIN DATA
                             |
                             v
                  Data Preprocessing
                  + Normalization
                             |
              +--------------+--------------+
              |                             |
              v                             v
       Traditional Blocking          Embedding Generation
              |                             |
              |                         FAISS Index
              |                             |
              +-------------+---------------+
                            |
                            v
                    Candidate Union
                    + Deduplication
                            |
                            v
                 Pairwise Feature Engineering
                            |
       +--------------------+----------------------+
       |                    |                      |
       v                    v                      v
  Name Features       Address Features       Other Features
  - Jaccard          - Jaccard              - Country match
  - Levenshtein      - Levenshtein          - Postal/PIN match
  - TF-IDF cosine    - TF-IDF cosine        - Embedding cosine
       |                    |                      |
       +--------------------+----------------------+
                            |
                            v
                       XGBoost Model
                            |
                            v
                    Match Probability
                            |
                            v
                 Threshold + Decision Logic
                            |
                  +---------+---------+
                  |                   |
                  v                   v
                MATCH              NO MATCH
                  |                   |
                  +---------+---------+
                            |
                            v
                 matching_results.tsv
```

------------------------------------------------------------------------

# 4. Step 1 --- Data Loading

All challenge files are TSV files and must be read using a tab
separator.

Example:

``` python
import pandas as pd

source1 = pd.read_csv("train_source1.tsv", sep="\t")
source2 = pd.read_csv("train_source2.tsv", sep="\t")
source3 = pd.read_csv("train_source3.tsv", sep="\t")
ground_truth = pd.read_csv("train_ground_truth.tsv", sep="\t")
```

The source files contain:

-   `entity_id`
-   `business_name`
-   `business_address`
-   `country`

The source is identifiable from the `S1-`, `S2-`, and `S3-` entity ID
prefixes and the corresponding file.

------------------------------------------------------------------------

# 5. Step 2 --- Data Preprocessing and Normalization

## Why?

Two records representing the same business may have different textual
representations.

For example:

``` text
ABC Technologies Pvt. Ltd.
ABC Technologies Private Limited
ABC Tech Pvt Ltd
```

may refer to the same entity.

Similarly:

``` text
12 MG Road, Bangalore
12 M.G. Rd, Bengaluru
```

may represent the same address.

Normalization creates additional standardized representations without
destroying the original values.

## Proposed normalization

### Business name

Possible operations:

-   lowercase conversion
-   Unicode normalization
-   whitespace normalization
-   punctuation normalization
-   controlled abbreviation normalization
-   tokenization
-   removal/standardization of common legal suffixes where appropriate

### Address

Possible operations:

-   lowercase conversion
-   Unicode normalization
-   whitespace normalization
-   punctuation normalization
-   address abbreviation normalization
-   tokenization
-   preservation of useful numeric/address components

### Important constraint

Normalization must be performed using the provided data and local
processing only.

The challenge prohibits external data lookup, external databases,
geocoding APIs, and external data augmentation.

------------------------------------------------------------------------

# 6. Step 3 --- Traditional Blocking

## What is blocking?

Blocking is the first candidate-generation stage.

Instead of comparing one Source 1 record with every Source 2 and Source
3 record, blocking generates a smaller set of plausible candidates.

For example:

``` text
Source 1:
ABC Technologies Pvt Ltd
Bangalore
India

             |
             v

Candidate generation

S2-101
S2-503
S3-210
S3-912
```

Only these candidates proceed to the more expensive matching stage.

## Why?

Suppose there are hundreds of thousands or millions of records.

A full pairwise comparison would create an extremely large number of
comparisons.

Blocking dramatically reduces the number of pairs that need detailed
scoring.

## Proposed blocking signals

Multiple blocking rules can be used, for example:

-   country
-   normalized business-name tokens
-   name prefixes
-   character n-grams
-   address tokens
-   postal/PIN information when present
-   combinations of the above

The blocking stage should favor **high recall**. A true match that is
never included in the candidate set cannot be recovered by the
downstream model.

------------------------------------------------------------------------

# 7. Step 4 --- Embedding-Based Candidate Retrieval

## Why embeddings?

Traditional blocking can miss matches when two records use substantially
different wording.

For example:

``` text
ABC Technologies Private Limited

ABC Tech Pvt Ltd
```

A semantic embedding can represent these texts in a vector space where
related records may be close to each other.

Therefore, embeddings are proposed as a **second candidate-generation
mechanism**, rather than as the final matching decision.

## Proposed representation

A combined representation can be constructed from fields such as:

``` text
business_name + business_address + country
```

Example:

``` text
ABC Technologies Pvt Ltd | 12 MG Road Bangalore | India
```

The embedding model converts this text into a numerical vector.

## Pre-computation

Source 2 and Source 3 records can be embedded once:

``` text
Source 2 + Source 3
        |
        v
Embedding model
        |
        v
Vectors
        |
        v
FAISS index
```

The index can then be reused for Source 1 queries.

This avoids regenerating Source 2/3 embeddings for every Source 1
record.

------------------------------------------------------------------------

# 8. Why FAISS?

FAISS is a library for efficient similarity search over vectors.

It is proposed here because the requirement is primarily:

> Given a Source 1 embedding, retrieve the most similar Source 2/Source
> 3 embeddings.

A full vector database is not necessarily required for this competition
because the pipeline does not inherently need database-style operations
such as application-level CRUD, distributed storage, or complex metadata
management.

The architecture can therefore remain simple:

``` text
Original records -> Pandas/Parquet
Embeddings       -> FAISS index
Matching model   -> XGBoost
```

The exact FAISS index type should be selected after benchmarking on a
representative subset of the data.

------------------------------------------------------------------------

# 9. Step 5 --- Candidate Union

Traditional blocking and embedding retrieval provide complementary
candidate sets.

For example:

``` text
Traditional blocking:
S2-101
S2-503
S3-210

Embedding retrieval:
S2-503
S2-777
S3-912
```

The final candidate set becomes:

``` text
S2-101
S2-503
S2-777
S3-210
S3-912
```

This is the union of both candidate-generation approaches.

Duplicates are removed.

## Why use both?

Traditional blocking provides inexpensive, structured signals.

Embedding retrieval can recover candidates that have semantic similarity
despite textual differences.

Using both reduces dependence on a single candidate-generation method.

------------------------------------------------------------------------

# 10. Step 6 --- Pairwise Feature Engineering

Once the candidate set is sufficiently small, detailed features can be
calculated for every Source 1--candidate pair.

## Name features

Recommended features include:

-   Jaccard similarity
-   normalized Levenshtein similarity
-   TF-IDF cosine similarity
-   token overlap
-   character n-gram similarity
-   exact normalized-name match

## Address features

Recommended features include:

-   Jaccard similarity
-   normalized Levenshtein similarity
-   TF-IDF cosine similarity
-   token overlap
-   numeric component overlap
-   character n-gram similarity

## Additional features

Examples:

-   country equality
-   postal/PIN equality when available
-   embedding cosine similarity
-   name length difference
-   address length difference
-   number/token overlap

The challenge specifically recommends Jaccard, Levenshtein, and TF-IDF
cosine similarity for names and addresses.

------------------------------------------------------------------------

# 11. Step 7 --- Supervised Matching Model

The detailed pairwise features are provided to a supervised ML
classifier.

A proposed model is **XGBoost**.

Example input:

``` text
name_jaccard
name_levenshtein
name_tfidf
address_jaccard
address_levenshtein
address_tfidf
embedding_cosine
country_match
postal_match
token_overlap
```

The model outputs a match probability:

``` text
S1-001 <-> S2-103
        |
        v
     XGBoost
        |
        v
     0.96
```

The probability is then used by the final decision logic.

------------------------------------------------------------------------

# 12. Why XGBoost?

The problem contains structured numerical features produced from
multiple similarity methods.

XGBoost is suitable for this type of feature-based classification
because it can model nonlinear interactions between features.

For example:

``` text
High name similarity
+
High address similarity
+
Same country
+
High embedding similarity
```

may provide a stronger matching signal than any single feature alone.

The model should be trained only on information available within the
challenge data.

------------------------------------------------------------------------

# 13. Training Labels

`train_ground_truth.tsv` provides the training labels.

For each Source 1 entity, the file contains the matching Source 2 and/or
Source 3 entity IDs.

These labels can be transformed into pair-level training examples:

``` text
Source 1      Candidate      Label

S1-001        S2-101         1
S1-001        S2-102         0
S1-001        S3-201         1
S1-001        S3-202         0
```

Where:

-   `1` = genuine match
-   `0` = non-match

Negative examples should be generated carefully from candidate pairs
rather than relying on arbitrary random pairs alone.

------------------------------------------------------------------------

# 14. Validation Strategy

The test set does not contain ground-truth labels.

Therefore, model development should use a validation split derived from
the training data.

A recommended process is:

``` text
Training data
     |
     +-------------------+
     |                   |
     v                   v
Training split      Validation split
     |                   |
     v                   v
Train model          Generate candidates
     |                   |
     +---------+---------+
               |
               v
        Evaluate F_0.5
```

The validation process should reproduce the real inference pipeline as
closely as possible.

This is important because candidate generation itself affects the
maximum achievable recall.

------------------------------------------------------------------------

# 15. Candidate Recall Must Be Measured

For the validation set, measure:

``` text
Candidate Recall =
Number of true matches present in candidates
/
Number of true matches
```

Example:

``` text
True matches = 1000
Found inside candidate sets = 970

Candidate Recall = 97%
```

If the candidate recall is only 70%, the downstream XGBoost model cannot
achieve more than that level of match coverage, regardless of how
accurate the classifier is.

Therefore, blocking parameters and FAISS Top-K should be tuned using
validation data.

------------------------------------------------------------------------

# 16. Final Matching and Threshold

The XGBoost model produces a probability for each candidate pair.

Example:

``` text
S1-001 -> S2-101    0.97
S1-001 -> S2-503    0.92
S1-001 -> S3-210    0.21
S1-001 -> S3-912    0.07
```

A threshold is then applied:

``` text
probability >= threshold
        |
        v
      MATCH
```

The threshold should not be selected arbitrarily.

It should be tuned on the validation set against the challenge's
evaluation metric.

------------------------------------------------------------------------

# 17. Precision vs Recall

The challenge uses **F_0.5**, which is precision-heavy.

Therefore, false matches are particularly important.

For example:

``` text
True:
S1-001 -> S2-101, S3-210

Prediction:
S1-001 -> S2-101, S3-210, S2-999
```

The extra `S2-999` is a false positive.

The evaluation metric places greater emphasis on precision than recall.

Therefore, the final decision threshold should be selected using
validation results rather than simply maximizing the number of predicted
matches.

------------------------------------------------------------------------

# 18. Singleton Handling

A Source 1 entity may have no matching Source 2 or Source 3 record.

Such an entity is a singleton.

Example:

``` text
S1-005
matched_entity_ids = ""
```

The model must be able to output no match.

This is especially important because correctly predicting a singleton
receives full per-entity credit under the challenge's scoring
definition, while incorrectly assigning a match receives no credit for
that entity.

------------------------------------------------------------------------

# 19. Output Generation

The final pipeline produces:

## `matching_results.tsv`

``` text
source1_entity_id    matched_entity_ids

S1-00001             S2-00047,S2-00193,S3-00812
S1-00002             S3-00004
S1-00003
```

Every Source 1 test entity must have exactly one row.

The `matched_entity_ids` field contains a comma-separated list of
matching Source 2 and/or Source 3 IDs.

------------------------------------------------------------------------

# 20. `candidate_pairs.tsv`

This file records the final candidate set that is actually passed to the
matching model.

Example:

``` text
source1_entity_id    candidate_entity_ids

S1-00001             S2-00047,S2-00193,S3-00812,S3-00999
S1-00002             S3-00004
S1-00003
```

If multiple blocking stages are used, this file should represent the
**last candidate set immediately before model inference**, not an
earlier intermediate blocking result.

Every final match must therefore also appear in the corresponding
candidate list.

------------------------------------------------------------------------

# 21. End-to-End Inference Pipeline

For the test set:

``` text
test_source1.tsv
test_source2.tsv
test_source3.tsv
          |
          v
   Normalize records
          |
          +-------------------------+
          |                         |
          v                         v
Traditional blocking          FAISS retrieval
          |                         |
          +------------+------------+
                       |
                       v
               Candidate union
                       |
                       v
              Feature generation
                       |
                       v
                  XGBoost
                       |
                       v
              Probability scores
                       |
                       v
              Threshold + rules
                       |
                       v
             Singleton handling
                       |
             +---------+---------+
             |                   |
             v                   v
candidate_pairs.tsv       matching_results.tsv
```

------------------------------------------------------------------------

# 22. Computational Efficiency

The main reason for the multi-stage architecture is scalability.

A naive approach would attempt to compare every Source 1 record against
every Source 2/Source 3 record.

For large datasets, this results in an enormous number of pairwise
comparisons.

The proposed approach reduces computation in stages:

``` text
Millions of raw records
        |
        v
Cheap blocking
        |
        v
Smaller candidate pool
        |
        v
Embedding Top-K retrieval
        |
        v
Final candidate pool
        |
        v
Expensive similarity features
        |
        v
XGBoost
```

Embeddings for Source 2 and Source 3 should be generated once and
indexed for reuse.

The embedding dimension, model, FAISS index type, and Top-K value should
be benchmarked on a representative subset before running the complete
dataset.

Memory usage must also be considered because storing millions of dense
vectors can require several gigabytes of RAM depending on vector
dimension and datatype.

------------------------------------------------------------------------

# 23. Why Not Use Embeddings Alone?

Embeddings are useful for semantic retrieval, but semantic similarity
does not necessarily mean entity identity.

For example:

``` text
ABC Technologies Bangalore
ABC Technologies Mumbai
```

can be semantically very similar while potentially representing
different records/entities.

Entity resolution also benefits from exact and character-level evidence
such as:

-   name edit distance
-   token overlap
-   address similarity
-   country consistency
-   postal/PIN information

Therefore, embeddings are used as **candidate retrieval features**,
while the final decision is made using multiple signals.

------------------------------------------------------------------------

# 24. Why Not Use Only Traditional Blocking?

Traditional blocking is efficient and interpretable, but strict rules
can miss records with substantial textual variation.

For example:

``` text
ABC Technologies Private Limited

ABC Tech Pvt Ltd
```

may not satisfy a strict exact blocking rule.

Embedding retrieval provides a complementary mechanism for finding
semantically related candidates.

Therefore:

``` text
Traditional blocking
        +
Embedding retrieval
        =
More robust candidate generation
```

The two candidate sets are combined before final scoring.

------------------------------------------------------------------------

# 25. Fair-Play and Data Usage

The solution must use only the data supplied by the challenge.

The following are explicitly excluded:

-   commercial entity-resolution APIs
-   external business databases
-   government business-registration lookups
-   geocoding APIs
-   external internet-based data augmentation

No external information should be used to determine the identity of a
business.

All preprocessing, candidate generation, embedding creation, feature
engineering, and model inference should operate on the supplied datasets
and locally generated representations.

------------------------------------------------------------------------

# 26. Model and Dependency Considerations

The challenge specifies that the final model must satisfy its stated
licensing and parameter constraints.

Before selecting an embedding model or any other pretrained model,
verify:

1.  Model license
2.  Maximum parameter count
3.  Whether commercial/competition use is permitted
4.  Compatibility with the challenge requirements

The same verification should be performed for all major model components
included in the final submission.

------------------------------------------------------------------------

# 27. Proposed Project Structure

``` text
business_entity_resolution/
│
├── data/
│   ├── train/
│   └── test/
│
├── src/
│   ├── preprocessing.py
│   ├── blocking.py
│   ├── embeddings.py
│   ├── faiss_index.py
│   ├── features.py
│   ├── train.py
│   ├── inference.py
│   ├── thresholding.py
│   └── utils.py
│
├── models/
│   ├── xgboost_model/
│   └── faiss_index/
│
├── output/
│   ├── candidate_pairs.tsv
│   └── matching_results.tsv
│
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

# 28. Recommended Development Order

Do not immediately run the complete dataset.

Develop incrementally:

### Phase 1 --- Data understanding

-   inspect row counts
-   inspect missing values
-   inspect name/address patterns
-   inspect country distribution
-   inspect ground-truth match counts

### Phase 2 --- Baseline

Implement:

``` text
normalization
+
simple blocking
+
basic string similarity
```

Establish a baseline F_0.5.

### Phase 3 --- Improved blocking

Add multiple blocking rules and measure candidate recall and reduction
ratio.

### Phase 4 --- Embedding retrieval

Add embeddings and FAISS.

Measure whether embedding retrieval increases candidate recall without
making the candidate set unnecessarily large.

### Phase 5 --- ML matcher

Train XGBoost using detailed pairwise features.

### Phase 6 --- Threshold tuning

Tune the decision threshold using validation F_0.5.

### Phase 7 --- Full-scale inference

Run the complete test pipeline and generate:

``` text
candidate_pairs.tsv
matching_results.tsv
```

### Phase 8 --- Validation

Run the provided submission validator before submitting.

------------------------------------------------------------------------

# 29. Final Proposed Solution

The proposed solution is therefore:

``` text
                 BUSINESS RECORDS
                        |
                        v
             NORMALIZATION / CLEANING
                        |
             +----------+----------+
             |                     |
             v                     v
     TRADITIONAL BLOCKING     EMBEDDING SEARCH
             |                     |
             |                   FAISS
             |                     |
             +----------+----------+
                        |
                        v
               CANDIDATE UNION
                        |
                        v
             DETAILED FEATURES
                        |
        +---------------+----------------+
        |               |                |
        v               v                v
     NAME SIM       ADDRESS SIM      METADATA
        |               |                |
        +---------------+----------------+
                        |
                        v
                     XGBOOST
                        |
                        v
                 MATCH PROBABILITY
                        |
                        v
             THRESHOLD + SINGLETON
                        |
                        v
              FINAL MATCHES
                        |
              +---------+---------+
              |                   |
              v                   v
     candidate_pairs.tsv   matching_results.tsv
```

## Core Design Principle

> **Use cheap methods to narrow the search space, semantic retrieval to
> improve candidate recall, detailed similarity features to describe
> each candidate pair, and a supervised ML model to make the final
> entity-resolution decision.**

This architecture is designed specifically for a large-scale
entity-resolution problem where correctness, candidate recall,
computational efficiency, and precision all matter.

------------------------------------------------------------------------

## 30. Important Challenge-Specific Constraints

The implementation must preserve the challenge requirements:

-   Source 1 is the reference source.
-   Every Source 1 test entity must receive exactly one output row.
-   A Source 1 entity may have zero, one, or multiple matches.
-   Final matches must come only from Source 2 and Source 3.
-   `matching_results.tsv` contains the final predictions.
-   `candidate_pairs.tsv` contains the candidate set actually passed to
    the final matching model.
-   Every final match must be present in the candidate set.
-   The test set has no ground-truth labels.
-   Validation should therefore be performed using a held-out portion of
    the training data.
-   Country must be treated as an open-set string field; the pipeline
    must not be hard-coded to only the training countries.
-   External data lookup and external entity-resolution/data
    augmentation services are prohibited.
-   The final model must comply with the challenge's stated
    model-license and parameter constraints.
