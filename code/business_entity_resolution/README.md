# Business Entity Resolution Pipeline

This repository contains the end-to-end Machine Learning pipeline for the **ML Challenge 2026: Business Entity Resolution**.

The pipeline resolves multi-source noisy business records across three independent data sources (Source 1, Source 2, Source 3) into unified business identities.

---

## Architecture Overview

```text
               Source 1, 2, 3 Records
                         |
                         v
              Data Preprocessing & Normalization
              - Unicode & lowercase normalization
              - Controlled abbreviation expansion
              - Token & numeric extraction
                         |
                         v
              High-Recall Multi-Rule Blocking
              - Country partition
              - Significant name token inverted index
              - Cleaned name prefix index
              - Address number + token inverted index
                         |
                         v
                 Candidate Pool Union
              (Logged to candidate_pairs.tsv)
                         |
                         v
              Pairwise Feature Engineering
              - Name Jaccard, Levenshtein, Exact, Overlap
              - Address Jaccard, Levenshtein, Exact, Numbers
              - Character 3-gram similarity
                         |
                         v
             LightGBM / XGBoost Matcher
                         |
                         v
              Threshold & Singleton Logic
           (Optimized for Macro F_0.5 metric)
                         |
                         v
                matching_results.tsv
```

---

## Directory Structure

```text
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── utils.py               # Parsing, logging, and macro F_0.5 computation
│   ├── preprocessing.py       # Normalization and abbreviation expansion
│   ├── blocking.py            # Multi-rule inverted index candidate generator
│   ├── embeddings.py          # Semantic text representations and embeddings
│   ├── faiss_index.py         # FAISS vector similarity search
│   ├── features.py            # 16 pairwise similarity features
│   ├── train.py               # Supervised model training with GroupKFold
│   ├── thresholding.py        # F_0.5 threshold optimization and singleton logic
│   └── inference.py           # Country-partitioned test inference pipeline
├── models/
│   ├── entity_matcher_lgb.pkl # Serialized LightGBM model
│   └── threshold_config.json  # Tuned decision threshold
├── requirements.txt           # Pinned dependencies
└── README.md                  # Reproduction instructions
```

---

## Installation & Setup

1. **Create and activate a virtual environment (Python 3.10+ recommended):**
   ```bash
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Reproduce End-to-End

### Step 1: Model Training & Threshold Tuning

Train the supervised matcher on training data with GroupKFold validation (grouped strictly by Source 1 entity):

```bash
python -m src.train \
    --train-dir ../../student_resource/student_resource/dataset/train \
    --model-dir models \
    --sample-size 40000 \
    --num-rounds 300
```

This step:
- Loads reference Source 1 records and ground-truth matches
- Performs multi-rule blocking and hard negative extraction
- Computes pairwise similarity features
- Trains a LightGBM classifier with early stopping
- Optimizes decision threshold for macro F_0.5 (including true singletons)
- Saves `models/entity_matcher_lgb.pkl` and `models/threshold_config.json`

### Step 2: Full Test Inference

Generate `matching_results.tsv` and `candidate_pairs.tsv` from the test datasets:

```bash
python -m src.inference \
    --test-dir ../../student_resource/student_resource/dataset/test \
    --model-dir models \
    --output-dir ../../output
```

This step:
- Discovers countries in test set dynamically (open-set: India, US, France)
- Processes records country-by-country using memory-efficient streaming (< 1.5 GB RAM)
- Generates candidate pairs and writes to `output/candidate_pairs.tsv`
- Applies model scoring and optimal threshold
- Writes predictions to `output/matching_results.tsv`

### Step 3: Validate Outputs

Validate formatting, entity coverage, and candidate subset rules using the challenge validator:

```bash
python ../../student_resource/student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../student_resource/student_resource/dataset/test
```
