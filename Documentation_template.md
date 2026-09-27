# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Team MVK  
**Submission Date:** September 2026  

---

## 1. Executive Summary

This solution presents a scalable, high-recall, precision-optimized Machine Learning pipeline designed for the Business Entity Resolution Challenge. Operating over millions of noisy business records across three independent data sources (Source 1 reference vs. Source 2 and Source 3), the architecture employs a multi-stage approach: (1) deterministic Unicode and legal abbreviation normalization, (2) an open-set country-partitioned multi-rule inverted index achieving >98% candidate recall, (3) a 16-dimensional pairwise fuzzy, token, and numeric feature engineering framework, (4) a gradient-boosted LightGBM classifier with entity-level GroupKFold cross-validation, and (5) an optimized decision threshold explicitly tuned for the precision-heavy macro-averaged $F_{0.5}$ metric with comprehensive singleton handling.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across the 3 sources identified four major noise patterns that make naive string matching ineffective:
1. **Corporate and Legal Variations:** Extreme variance in entity suffixes (`Pvt` vs. `Private`, `Ltd` vs. `Limited`, `Corp` vs. `Corporation`, `Inc.` vs. `Incorporated`, `Co.` vs. `Company`, `LLC`, `LLP`, `SARL`, `SAS`, `GmbH`).
2. **Missing and Corrupted Names:** Approximately 8.4% of candidate records in Source 2 and Source 3 have empty or heavily truncated business names where the entity is identified predominantly through its address components.
3. **Address Component Reordering and Noise:** High frequency of punctuation symbols (`##`, `***`, `,`, `.`), street abbreviations (`Rd` vs. `Road`, `St` vs. `Street`, `Ave` vs. `Avenue`, `Blvd` vs. `Boulevard`), landmark references (`Opp.`, `Near`, `Beside`), and municipal numbering formats.
4. **Country Partitioning & Open-Set Distribution:** Entities match strictly within their respective country. The test set introduces a third country (`France`), not present in the training set (`US` and `India`). Hardcoded country filters would fail on France; hence, country must be processed as an open-set partition.

### 2.2 Solution Strategy
- **Approach Type:** Multi-Stage Hybrid: Normalization $\rightarrow$ Country-Partitioned Multi-Rule Inverted Index Blocking $\rightarrow$ Candidate Union $\rightarrow$ Pairwise Feature Engineering $\rightarrow$ Supervised Gradient Boosted Classifier (LightGBM) $\rightarrow$ $F_{0.5}$ Threshold Optimization & Singleton Handling.
- **Core Innovation:** A joint name-and-address inverted index architecture that combines significant name tokens, character prefixes, and building number/locality tokens. This catches near-duplicate names while simultaneously capturing records with missing names that match strictly on address, pushing candidate recall to over 98% while keeping the candidate pool compact (~5–20 candidates per entity).

---

## 3. Candidate Generation (Blocking)

### 3.1 Blocking Keys Used
To ensure high recall without combinatorial explosion across ~12 million candidate comparisons, blocking is executed within country partitions using five complementary inverted index rules:
1. **Rule 1 (Exact Normalized Name):** Hash lookup on fully standardized, punctuation-free business names.
2. **Rule 2 (Significant Name Tokens):** Inverted index over non-generic, high-information tokens (filtering stop words like *services*, *technologies*, *consulting*, *solutions*).
3. **Rule 3 (Name Character Prefix):** First 5 alphanumeric characters of normalized name without whitespace (recovers spelling typos and suffix corruptions).
4. **Rule 4 (Address Number + Name Token):** Joint index on building/premise number and the primary name token.
5. **Rule 5 (Address Number + Significant Locality Token):** Captures candidates where the business name is missing or drastically altered, but street/building address matches.

### 3.2 Candidate Pool & Recall Performance
- **Candidate Pool Capping:** Postings lists for generic keys are capped at 150 to avoid explosive candidate blocks.
- **Candidate Retention:** Candidates are scored by rule hit count, retaining the top 20 most promising candidates per Source 1 entity.
- **Candidate Recall:** Reaches **98.2%** recall on held-out validation benchmarks, dramatically reducing the search space by a reduction ratio of >99.99%.
- **Candidate Set Audit:** All candidates passed to the classifier are recorded in `output/candidate_pairs.tsv` to ensure complete reproducibility and guarantee that all final matches in `output/matching_results.tsv` are strict subsets.

---

## 4. Matching Model

### 4.1 Feature Engineering (16 Dimensions)
For each candidate pair `(s1_rec, cand_rec)`, a 16-dimensional feature vector is computed:
- **Name Similarity Features:**
  - `name_jaccard`: Token Jaccard similarity.
  - `name_levenshtein`: Normalized Levenshtein similarity (via RapidFuzz C++ backend).
  - `name_exact_match`: Binary indicator for identical normalized names.
  - `name_token_overlap`: Count of shared significant name tokens.
  - `name_token_recall`: Fraction of significant tokens matched relative to the shorter string.
  - `name_prefix_match`: Binary indicator for matching 4-character clean prefixes.
  - `name_len_diff`: Absolute difference in normalized string character lengths.
- **Address Similarity Features:**
  - `addr_jaccard`: Address token Jaccard similarity.
  - `addr_levenshtein`: Normalized Levenshtein similarity over normalized address strings.
  - `addr_exact_match`: Binary indicator for identical normalized addresses.
  - `addr_token_overlap`: Count of shared significant address/locality tokens.
  - `addr_number_jaccard`: Jaccard similarity across extracted numeric components (building/PIN/suite).
  - `addr_number_exact`: Binary indicator for matching primary street/building number.
  - `addr_len_diff`: Absolute difference in normalized address string lengths.
- **Cross-Field & Metadata Features:**
  - `country_match`: Binary consistency check (1.0 for same country, 0.0 otherwise).
  - `combined_char_jaccard`: Character 3-gram Jaccard similarity over combined `name + address` text.

### 4.2 Model Type & Cross-Validation
- **Model Type:** LightGBM Binary Classifier (`learning_rate=0.06`, `num_leaves=31`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`).
- **Validation Splitting:** 5-fold `GroupKFold` strictly grouped by `source1_entity_id`. Splitting by raw pair is prohibited as it causes severe entity leakage between train and validation splits.
- **Loss Function:** Binary logloss with early stopping on the validation set.

### 4.3 Decision Threshold Optimization & Singleton Handling
- **Metric Formulation:** Macro-averaged $F_{0.5}$:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- **Precision Weighting:** Because $F_{0.5}$ penalizes false positives (false merges) twice as heavily as false negatives, the optimal threshold is shifted upwards towards high-precision candidates.
- **Singleton Handling:** Entities with zero matches in the candidate pool or where all candidate probabilities fall below the threshold are assigned an empty match list (`""`), receiving full credit (1.0) under the competition scoring formula.
- **Optimal Threshold:** Tuned via validation sweep to $t^* \approx 0.50 - 0.55$, achieving peak macro $F_{0.5}$.

---

## 5. Results & Error Analysis

- **Validation Macro $F_{0.5}$ Score:** $0.784$ on held-out entity splits.
- **Top Predictive Features by Gain:**
  1. `addr_jaccard` (Dominant signal in verifying shared physical location)
  2. `name_jaccard` (Primary business identity alignment)
  3. `name_levenshtein` (Handles typo variations and abbreviation expansions)
  4. `addr_levenshtein` (Handles street address formatting differences)
  5. `addr_number_exact` (Crucial differentiator against neighboring businesses)
- **Common False Positives (Wrong Merges):** Co-located businesses in large commercial complexes or IT parks sharing identical building numbers and localities, but possessing distinct business names. Penalized by checking `name_jaccard` and `name_levenshtein`.
- **Common False Negatives (Missed Matches):** Entities with completely missing business names in the source record paired with drastic transliteration variations in street names. Mitigated through address number and character 3-gram overlap.

---

## 6. Conclusion

The developed pipeline delivers a robust, scalable, and memory-efficient solution for large-scale Entity Resolution under noisy multi-source conditions. By coupling multi-rule inverted index blocking with rapid C++-backed string metrics, gradient boosted decision trees, and precision-heavy thresholding, the system processes multi-million record datasets while strictly adhering to open-set country requirements and formatting constraints.

---

## Appendix

### A. Code Artefacts
The full submission package is organized as follows:
```text
<team_name>_submission/
├── output/
│   ├── matching_results.tsv        # Final entity matches (submitted for scoring)
│   └── candidate_pairs.tsv         # Candidate pairs fed to the matching model
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── utils.py            # I/O, timers, and macro F_0.5 metric evaluation
│       │   ├── preprocessing.py    # Normalization and abbreviation expansion
│       │   ├── blocking.py         # Multi-rule inverted index candidate generator
│       │   ├── embeddings.py       # Dense/sparse text embeddings
│       │   ├── faiss_index.py      # FAISS vector similarity search
│       │   ├── features.py         # 16-dimensional pairwise similarity extraction
│       │   ├── train.py            # Supervised LightGBM training with GroupKFold
│       │   ├── thresholding.py     # Threshold optimization & singleton handling
│       │   └── inference.py        # Streaming country-partitioned inference pipeline
│       ├── models/                 # Saved model weights & threshold metadata
│       ├── requirements.txt        # Pinned dependencies
│       └── README.md               # End-to-end reproduction guide
└── Documentation_template.md       # Technical methodology write-up
```

### B. Reproduction Commands
- **Training:** `python -m src.train --train-dir dataset/train --model-dir models`
- **Inference:** `python -m src.inference --test-dir dataset/test --model-dir models --output-dir ../../output`
- **Validation:** `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`
