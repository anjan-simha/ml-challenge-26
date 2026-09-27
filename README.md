# ML Challenge 2026: Business Entity Resolution

<p align="center">
  <b>Team Name:</b> Team MVK &nbsp;|&nbsp;
  <b>Track:</b> Business Entity Resolution &nbsp;|&nbsp;
  <b>Evaluation Metric:</b> Macro-averaged $F_{0.5}$ Score
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Validation%20Macro%20F0.5-0.9741-brightgreen.svg" alt="Validation Macro F0.5" />
  <img src="https://img.shields.io/badge/Optimal%20Threshold-0.95-blue.svg" alt="Optimal Threshold" />
  <img src="https://img.shields.io/badge/Candidate%20Recall-%3E98.2%25-orange.svg" alt="Candidate Recall" />
  <img src="https://img.shields.io/badge/Model-LightGBM%20(16--D%20Features)-purple.svg" alt="Model" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License" />
</p>

---

## Table of Contents

1. [Overview & Problem Statement](#1-overview--problem-statement)
2. [End-to-End Pipeline Architecture](#2-end-to-end-pipeline-architecture)
3. [Repository & Submission Structure](#3-repository--submission-structure)
4. [Key Innovations & Technical Methodology](#4-key-innovations--technical-methodology)
   - [4.1 Preprocessing & Legal Entity Normalization](#41-preprocessing--legal-entity-normalization)
   - [4.2 Multi-Rule Inverted Index Blocking](#42-multi-rule-inverted-index-blocking)
   - [4.3 16-Dimensional RapidFuzz Feature Engineering](#43-16-dimensional-rapidfuzz-feature-engineering)
   - [4.4 LightGBM Classifier with GroupKFold Validation](#44-lightgbm-classifier-with-groupkfold-validation)
   - [4.5 Macro $F_{0.5}$ Threshold Optimization & Singleton Logic](#45-macro-f05-threshold-optimization--singleton-logic)
   - [4.6 Open-Set Country Partitioning & Streaming Inference](#46-open-set-country-partitioning--streaming-inference)
5. [Installation & Setup](#5-installation--setup)
6. [Step-by-Step Reproduction Guide](#6-step-by-step-reproduction-guide)
   - [Step 1: Supervised Model Training & Threshold Optimization](#step-1-supervised-model-training--threshold-optimization)
   - [Step 2: Full Test Inference](#step-2-full-test-inference)
   - [Step 3: Submission Format & Integrity Validation](#step-3-submission-format--integrity-validation)
7. [Validation Results & Feature Analysis](#7-validation-results--feature-analysis)
8. [Submission Deliverables Checklist](#8-submission-deliverables-checklist)
9. [Academic Integrity & Fair Play Statement](#9-academic-integrity--fair-play-statement)

---

## 1. Overview & Problem Statement

In large-scale commercial ecosystems, business identity data originates from multiple disparate sources—each capturing partial, noisy, and unlinked fragments of information about the same real-world entities. 

The goal of the **ML Challenge 2026: Business Entity Resolution** is to resolve entities across three independent data sources:
- **Source 1 (`S1-*`):** The deduplicated reference source. Predictions must be provided for **every single Source 1 entity**.
- **Source 2 (`S2-*`) & Source 3 (`S3-*`):** Unlinked, noisy incoming records. A Source 1 entity may match zero (singleton), one, or multiple records across Source 2 and Source 3.

### Primary Data Challenges Addressed:
1. **Severe Legal and Suffix Variance:** Disparities across entity types (`Pvt` vs. `Private`, `Ltd` vs. `Limited`, `Corp` vs. `Corporation`, `Inc.` vs. `Incorporated`, `Co.` vs. `Company`, `LLC`, `LLP`, `SARL`, `SAS`).
2. **Missing & Truncated Business Names:** ~8.4% of candidate records have empty, corrupted, or heavily truncated names, requiring identity resolution through street/building address components.
3. **Address Permutations & Formatting:** Extensive punctuation noise (`##`, `***`), abbreviations (`Rd` vs. `Road`, `St` vs. `Street`), municipal numbering, and landmark directions (`Opp.`, `Near`, `Beside`).
4. **Open-Set Country Distribution:** The training set contains entities from `US` and `India`, whereas the test set introduces a third country (`France`). Entities only match within their respective country.
5. **Precision-Heavy Metric ($F_{0.5}$):** The competition evaluates submissions using **Macro-Averaged $F_{0.5}$**:
   $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   False merges (false positives) are penalized **twice as heavily** as missed matches (false negatives). Singletons (entities with no matches) receive a full score of **1.0** when correctly identified with an empty match list (`""`), and **0.0** if falsely matched.

---

## 2. End-to-End Pipeline Architecture

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      Raw Data Sources (S1, S2, S3)                     │
 └────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                   Data Preprocessing & Normalization                   │
 │   - Unicode NFKD normalization, lowercase conversion                   │
 │   - Punctuation removal & controlled legal/street abbreviation map     │
 │   - High-information token extraction & building digit isolation       │
 └────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                 Open-Set Country-Partitioned Blocking                  │
 │   Rule 1: Exact Normalized Name                                        │
 │   Rule 2: Significant Name Token Inverted Index                        │
 │   Rule 3: Cleaned Name 5-Character Prefix Index                        │
 │   Rule 4: Building Number + Primary Name Token Index                   │
 │   Rule 5: Building Number + Significant Locality Token Index          │
 │   (Frequency capped at 150; top 20 candidates retained per entity)    │
 └────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                   Candidate Pool Union & TSV Stream                    │
 │               (Output to output/candidate_pairs.tsv)                   │
 └────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │            16-Dimensional RapidFuzz Pairwise Feature Extraction        │
 │   - Name: Jaccard, Levenshtein, exact match, token overlap/recall,     │
 │           prefix match, length difference                              │
 │   - Address: Jaccard, Levenshtein, exact match, token overlap, numeric │
 │              Jaccard, primary number exact match, length difference    │
 │   - Cross-Field: Character 3-gram Jaccard, country consistency         │
 └────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │               LightGBM Gradient Boosted Matcher (5-Fold)               │
 │           (GroupKFold strictly grouped by Source 1 entity ID)          │
 └────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │           Optimal Thresholding & Singleton Logic (t* = 0.95)           │
 │   - Tuned on held-out validation entities for Macro F_0.5              │
 │   - Strict singleton assignment (empty list "" -> 1.0 score credit)    │
 └────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                       Final Submission TSV                             │
 │               (Output to output/matching_results.tsv)                  │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository & Submission Structure

The repository conforms strictly to the ML Challenge 2026 packaging rules:

```text
team_mvk_submission/
├── output/
│   ├── matching_results.tsv             # Final predicted matches (scored on leaderboard)
│   └── candidate_pairs.tsv              # Blocking candidate set fed to matching model
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py              # Package initializer
│       │   ├── utils.py                 # TSV streaming, timing, & macro F_0.5 evaluation
│       │   ├── preprocessing.py         # Unicode, legal abbreviation & digit normalizer
│       │   ├── blocking.py              # 5-rule inverted index candidate generator
│       │   ├── embeddings.py            # Dense/sparse semantic representations
│       │   ├── faiss_index.py           # FAISS vector similarity search utilities
│       │   ├── features.py              # 16-D RapidFuzz pairwise similarity features
│       │   ├── train.py                 # LightGBM training with GroupKFold & tuning
│       │   ├── thresholding.py          # Macro F_0.5 threshold sweep & singleton logic
│       │   └── inference.py             # Memory-efficient streaming test inference
│       ├── models/
│       │   ├── entity_matcher_lgb.pkl   # Serialized trained LightGBM model weights
│       │   └── threshold_config.json    # Optimal threshold config (0.95, Macro F0.5: 0.9741)
│       ├── requirements.txt             # Pinned environment dependencies
│       └── README.md                    # Reproduction instructions for code package
├── student_resource/                    # Challenge dataset & evaluation tools
│   └── student_resource/
│       ├── dataset/
│       │   ├── train/                   # S1, S2, S3 & ground truth TSVs
│       │   └── test/                    # Test S1, S2, S3 TSVs (US, India, France)
│       └── utils/
│           └── validate_submission.py   # Official submission integrity validator
├── Documentation_template.md            # Technical methodology write-up
├── Business_Entity_Resolution_Proposed_Solution.md # Architectural specification
└── README.md                            # Master project documentation (this file)
```

---

## 4. Key Innovations & Technical Methodology

### 4.1 Preprocessing & Legal Entity Normalization
- **Unicode & Case Normalization:** Full NFKD normalization followed by lowercase conversion and strip operations.
- **Legal Entity Mapping:** Standardizes 30+ regional and international corporate suffixes (`private limited` $\rightarrow$ `pvt ltd`, `corporation`/`incorporated` $\rightarrow$ `corp`, `llc`, `sarl`, `gmbh`, `sa`).
- **Street Address Normalization:** Expands directional and street tokens (`rd` $\rightarrow$ `road`, `st` $\rightarrow$ `street`, `ave` $\rightarrow$ `avenue`, `blvd` $\rightarrow$ `boulevard`, `opp` $\rightarrow$ `opposite`).
- **Digit Extraction:** Extracts numerical sequences representing building numbers, suites, and postal PIN codes for exact location verification.

### 4.2 Multi-Rule Inverted Index Blocking
Naive pairwise comparison across millions of entities requires ~12 trillion comparisons ($O(N^2)$). We employ a country-partitioned multi-rule inverted index achieving **>98.2% candidate recall** with a **>99.99% reduction ratio**:
1. **Rule 1 (Exact Normalized Name):** Direct hash lookup on standardized business name.
2. **Rule 2 (Significant Name Tokens):** Inverted index over informative name tokens (stopwords like *enterprise*, *solutions*, *services* filtered).
3. **Rule 3 (Cleaned Name 5-Prefix):** First 5 alphanumeric characters to catch phonetic typos and transcription errors.
4. **Rule 4 (Premise Number + Primary Name Token):** Joint index on building number and primary name token.
5. **Rule 5 (Premise Number + Locality Token):** Crucial fallback mechanism that captures entities with **missing or completely blank names** that match based on physical address.
- **Posting Frequency Capping:** Lists capped at 150 entries to prevent generic tokens from flooding candidate blocks.
- **Candidate Pool Capping:** Top 20 candidates per Source 1 entity retained based on multi-rule hit count.

### 4.3 16-Dimensional RapidFuzz Feature Engineering
Using the C++-accelerated `RapidFuzz` backend, the pipeline computes 16 pairwise similarity features per candidate pair:
- **Name Metrics:** Token Jaccard, normalized Levenshtein similarity, exact match boolean, token overlap count, token recall ratio, 4-character prefix match boolean, length difference.
- **Address Metrics:** Address token Jaccard, normalized Levenshtein similarity, exact match boolean, token overlap count, building/PIN number Jaccard, primary building number exact match boolean, address length difference.
- **Cross-Field Metrics:** Character 3-gram Jaccard similarity across concatenated `name + address`, country consistency indicator.

### 4.4 LightGBM Classifier with GroupKFold Validation
- **Model Type:** LightGBM Binary Classifier (`num_leaves=31`, `learning_rate=0.06`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`).
- **GroupKFold (5 Folds):** Validation splits are grouped strictly by `source1_entity_id`. Standard random pair splitting leads to catastrophic entity leakage; GroupKFold guarantees that validation entities have never been seen during training.
- **Training Efficiency:** Trains in under 3 minutes with early stopping on binary logloss.

### 4.5 Macro $F_{0.5}$ Threshold Optimization & Singleton Logic
- **Metric-Aligned Optimization:** Because the evaluation metric is Macro $F_{0.5}$, precision is weighted twice as heavily as recall.
- **Threshold Sweep:** A systematic grid sweep over validation entities identifies the optimal probability threshold ($t^* = 0.95$). Lower thresholds boost recall at the expense of precision, which severely degrades macro $F_{0.5}$.
- **Singleton Handling:** A Source 1 entity with zero candidate hits or whose candidates all fall below $t^*$ is output with an empty match string (`""`). This earns a **1.0 score** under the challenge formula.

### 4.6 Open-Set Country Partitioning & Streaming Inference
- **Dynamic Country Discovery:** The pipeline scans `test_source1.tsv` dynamically, discovering `US`, `India`, and `France` without hardcoded assumptions.
- **Country Partitioning:** Matches are strictly within-country, cutting candidate comparison spaces by orders of magnitude.
- **Memory Streaming:** Reads large source files in chunked streams (`chunksize=150,000`), discarding processed records per country and maintaining peak memory usage strictly under **1.5 GB RAM**.

---

## 5. Installation & Setup

### Prerequisites
- **Operating System:** Windows, Linux, or macOS
- **Python:** Python 3.10+ recommended

### Environment Setup

1. **Clone or Navigate to the Workspace:**
   ```bash
   cd team_mvk_submission
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   # On Windows (PowerShell):
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # On Linux/macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r code/business_entity_resolution/requirements.txt
   ```

---

## 6. Step-by-Step Reproduction Guide

The complete pipeline can be executed either from the project root or from inside `code/business_entity_resolution/`.

### Step 1: Supervised Model Training & Threshold Optimization

Train the LightGBM classifier on the training set with entity-grouped GroupKFold validation and find the optimal $F_{0.5}$ threshold:

```bash
cd code/business_entity_resolution

python -m src.train \
    --train-dir ../../student_resource/student_resource/dataset/train \
    --model-dir models \
    --sample-size 40000 \
    --num-rounds 300
```

**Outputs Produced:**
- `models/entity_matcher_lgb.pkl`: Serialized LightGBM model weights.
- `models/threshold_config.json`: Tuned decision threshold ($t^* = 0.95$) and validation score.

---

### Step 2: Full Test Inference

Generate `matching_results.tsv` and `candidate_pairs.tsv` across all test countries (`US`, `India`, `France`):

```bash
python -m src.inference \
    --test-dir ../../student_resource/student_resource/dataset/test \
    --model-dir models \
    --output-dir ../../output
```

**Outputs Produced:**
- `output/candidate_pairs.tsv`: Tab-separated candidate pool generated during blocking.
- `output/matching_results.tsv`: Final predicted matches thresholded for Macro $F_{0.5}$.

---

### Step 3: Submission Format & Integrity Validation

Validate both output files against the official challenge validator to ensure zero formatting errors, full entity coverage, and strict candidate subset compliance:

```bash
# Return to workspace root
cd ../..

python student_resource/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/student_resource/dataset/test
```

**Expected Validator Result:**
```text
PASS: output/matching_results.tsv and output/candidate_pairs.tsv are valid and ready for submission.
```

---

## 7. Validation Results & Feature Analysis

### Validation Performance
- **Validation Macro $F_{0.5}$ Score:** **0.9741**
- **Optimal Decision Threshold ($t^*$):** **0.95**
- **Candidate Pool Recall:** **>98.2%**
- **Search Space Reduction Ratio:** **>99.99%**

### Top Predictive Features by Information Gain:
| Rank | Feature | Importance Description |
|:---:|:---|:---|
| 1 | `addr_jaccard` | Dominant location signal; verifies shared physical address tokens |
| 2 | `name_jaccard` | Primary identity alignment across significant business name tokens |
| 3 | `name_levenshtein` | Handles minor typographical errors and abbreviation expansions |
| 4 | `addr_levenshtein` | Reconciles street ordering and localized suffix variations |
| 5 | `addr_number_exact` | Crucial discriminator against neighboring businesses in the same locality |
| 6 | `combined_char_jaccard` | Robust against missing names where address dominates the text |

---

## 8. Submission Deliverables Checklist

- [x] **`output/matching_results.tsv`**: Tab-separated, exact header `source1_entity_id\tmatched_entity_ids`, exactly one row per test S1 entity, empty strings for singletons.
- [x] **`output/candidate_pairs.tsv`**: Tab-separated candidate set, exact header `source1_entity_id\tcandidate_entity_ids`, superset of final matches.
- [x] **`code/business_entity_resolution/src/`**: Self-contained, modular, reproducible Python package.
- [x] **`code/business_entity_resolution/requirements.txt`**: Pinned dependencies for clean reproduction.
- [x] **`Documentation_template.md`**: Completed technical methodology report following challenge guidelines.
- [x] **`README.md`**: Master reproduction and architecture documentation.

---

## 9. Academic Integrity & Fair Play Statement

This solution strictly adheres to the fair-play and academic integrity guidelines of the ML Challenge 2026:
- **No External Data Lookups:** Zero external databases, government registries, commercial APIs, or geocoding services were used. All predictions are generated purely from the provided dataset.
- **Model Constraints:** Model uses LightGBM, fully conforming to MIT/Apache 2.0 open-source licensing and well below the 8-billion parameter ceiling.
- **Open-Set Compliance:** No hardcoded country filters; dynamic handling of unseen countries (`France`).
- **Reproducibility:** All scripts run deterministically with pinned random seeds.

---

## 10. Team & License

- **Team Name:** Team MVK
- **Competition:** ML Challenge 2026 — Business Entity Resolution
- **License:** MIT License
