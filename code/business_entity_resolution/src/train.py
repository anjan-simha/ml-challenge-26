"""
ML Challenge 2026: Business Entity Resolution
Supervised Classifier Training & Validation Pipeline

Trains a LightGBM/XGBoost gradient boosted classifier on pairwise similarity features.
Uses GroupKFold on Source 1 entities to prevent cross-pair data leakage.
Tunes the decision threshold on validation set for macro-averaged F_0.5.
Saves the trained model and optimal threshold to the models/ directory.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from .utils import Timer, macro_f05_score, parse_id_list
from .preprocessing import preprocess_record
from .blocking import MultiRuleBlocker
from .features import compute_pair_features, FEATURE_NAMES
from .thresholding import tune_decision_threshold, save_threshold_config


def train_pipeline(train_dir, model_dir, sample_size=40000, num_boost_round=300):
    """
    Execute end-to-end model training, validation, and threshold optimization.
    
    Args:
        train_dir: path to directory containing train_source1.tsv, train_source2.tsv,
                   train_source3.tsv, train_ground_truth.tsv
        model_dir: directory to save trained model artifacts and threshold config
        sample_size: number of S1 training entities to use for model development
        num_boost_round: maximum number of boosting iterations
    """
    os.makedirs(model_dir, exist_ok=True)
    
    with Timer(f"Loading {sample_size} S1 records and Ground Truth"):
        s1_path = os.path.join(train_dir, "train_source1.tsv")
        gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

        s1_df = pd.read_csv(s1_path, sep="\t", nrows=sample_size)
        gt_df = pd.read_csv(gt_path, sep="\t", nrows=sample_size)

        # Build ground truth dictionary: s1_id -> set(matched_ids)
        s1_ids_loaded = set(s1_df["entity_id"])
        gt_df = gt_df[gt_df["source1_entity_id"].isin(s1_ids_loaded)]
        
        gt_dict = {}
        all_target_cands = set()
        for _, row in gt_df.iterrows():
            m_list = parse_id_list(row.get("matched_entity_ids"))
            gt_dict[row["source1_entity_id"]] = set(m_list)
            all_target_cands.update(m_list)

        print(f"Loaded {len(s1_df)} S1 entities with {len(all_target_cands)} unique true matches.", flush=True)

    with Timer("Preprocessing S1 records"):
        s1_records = {}
        s1_by_country = {}
        for _, row in s1_df.iterrows():
            rec = preprocess_record(row.to_dict())
            s1_records[rec["entity_id"]] = rec
            s1_by_country.setdefault(rec["country"], []).append(rec)

    with Timer("Scanning Source 2 and Source 3 for matching candidates & distractors"):
        cand_records = {}
        cand_by_country = {}
        needed_remaining = set(all_target_cands)

        # Scan S2 and S3 in chunks
        for s_file in ["train_source2.tsv", "train_source3.tsv"]:
            path = os.path.join(train_dir, s_file)
            if not os.path.exists(path):
                continue
            
            # Read first 300,000 rows for broad negative pool + true matches
            for chunk in pd.read_csv(path, sep="\t", chunksize=150000):
                # Always grab true matches
                true_hits = chunk[chunk["entity_id"].isin(needed_remaining)]
                for _, row in true_hits.iterrows():
                    rec = preprocess_record(row.to_dict())
                    cand_records[rec["entity_id"]] = rec
                    cand_by_country.setdefault(rec["country"], []).append(rec)
                    needed_remaining.discard(rec["entity_id"])

                # Also grab distractor records for negative candidate pool
                distractors = chunk.sample(n=min(len(chunk), 15000), random_state=42)
                for _, row in distractors.iterrows():
                    cid = row["entity_id"]
                    if cid not in cand_records:
                        rec = preprocess_record(row.to_dict())
                        cand_records[cid] = rec
                        cand_by_country.setdefault(rec["country"], []).append(rec)

                if len(needed_remaining) == 0 or len(cand_records) >= 200000:
                    break

        print(f"Candidate pool size: {len(cand_records)} records.", flush=True)

    with Timer("Building Multi-Rule Blockers & Generating Candidate Pairs"):
        candidate_pairs = []
        found_true_matches = 0
        total_evaluable_true = 0

        for country, s1_list in s1_by_country.items():
            cands_in_country = cand_by_country.get(country, [])
            if not cands_in_country:
                continue

            blocker = MultiRuleBlocker(max_candidates_per_entity=20, max_postings_per_key=150)
            for c_rec in cands_in_country:
                blocker.add_candidate(c_rec)
            blocker.finalize_index()

            for s1_rec in s1_list:
                s1_id = s1_rec["entity_id"]
                true_matches = gt_dict.get(s1_id, set()).intersection(cand_records.keys())
                total_evaluable_true += len(true_matches)

                retrieved = set(blocker.retrieve_candidates(s1_rec))
                # Measure blocking recall
                if true_matches:
                    found_true_matches += len(true_matches.intersection(retrieved))

                # Inject true matches into training pairs to ensure balanced supervision
                all_cands_for_s1 = retrieved.union(true_matches)
                for cid in all_cands_for_s1:
                    label = 1 if cid in true_matches else 0
                    candidate_pairs.append((s1_id, cid, label))

            blocker.clear()

        recall = (found_true_matches / total_evaluable_true) if total_evaluable_true > 0 else 0.0
        print(f"Blocking Candidate Recall on evaluable matches: {recall:.2%}", flush=True)
        print(f"Total candidate pairs generated: {len(candidate_pairs)}", flush=True)

    with Timer("Extracting pairwise similarity features"):
        pairs_df = pd.DataFrame(candidate_pairs, columns=["source1_entity_id", "candidate_id", "label"])
        feature_rows = []
        for row in pairs_df.itertuples(index=False):
            rec1 = s1_records[row.source1_entity_id]
            rec2 = cand_records[row.candidate_id]
            feature_rows.append(compute_pair_features(rec1, rec2))

        X = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
        y = pairs_df["label"]
        groups = pairs_df["source1_entity_id"]

    with Timer("Training LightGBM model with GroupKFold validation"):
        # Split by Source 1 entity (strict non-leakage guarantee)
        gkf = GroupKFold(n_splits=5)
        train_idx, val_idx = next(gkf.split(X, y, groups))

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        val_pairs_df = pairs_df.iloc[val_idx].copy()

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.06,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": -1,
            "random_state": 42,
            "n_jobs": -1,
        }

        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=num_boost_round,
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
        )

        val_pairs_df["prob"] = model.predict(X_val)

    with Timer("Tuning Decision Threshold for Macro F_0.5 on held-out validation entities"):
        val_s1_entities = set(val_pairs_df["source1_entity_id"])
        # Subset ground truth for validation entities (including singletons)
        val_gt = {s1_id: gt_dict.get(s1_id, set()) for s1_id in val_s1_entities}

        best_t, best_score, history = tune_decision_threshold(val_pairs_df, val_gt)
        print(f"\n==========================================", flush=True)
        print(f"Optimal Decision Threshold: {best_t:.4f}", flush=True)
        print(f"Validation Macro F_0.5    : {best_score:.4f}", flush=True)
        print(f"==========================================\n", flush=True)

    # Save artifacts
    model_path = os.path.join(model_dir, "entity_matcher_lgb.pkl")
    thresh_path = os.path.join(model_dir, "threshold_config.json")
    joblib.dump(model, model_path)
    save_threshold_config(best_t, best_score, thresh_path)
    print(f"Saved trained model to {model_path}", flush=True)
    print(f"Saved threshold configuration to {thresh_path}", flush=True)

    return model, best_t, best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Entity Resolution Supervised Model")
    parser.add_argument("--train-dir", type=str, default="dataset/train", help="Path to train data directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save model artifacts")
    parser.add_argument("--sample-size", type=int, default=40000, help="Number of S1 entities to train on")
    parser.add_argument("--num-rounds", type=int, default=300, help="Maximum boosting iterations")
    args = parser.parse_args()

    train_pipeline(
        train_dir=args.train_dir,
        model_dir=args.model_dir,
        sample_size=args.sample_size,
        num_boost_round=args.num_rounds,
    )
