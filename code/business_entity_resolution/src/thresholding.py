"""
ML Challenge 2026: Business Entity Resolution
Decision Threshold Optimization & Singleton Handling Module

Optimizes the probability decision threshold to maximize macro-averaged F_0.5.
Handles singleton Source 1 entities with no matches, earning full credit (1.0).
"""

import json
import os
import numpy as np
from collections import defaultdict
from .utils import macro_f05_score


def tune_decision_threshold(pair_df, ground_truth, min_thresh=0.25, max_thresh=0.95, step=0.02):
    """
    Search for the decision threshold that maximizes Macro-averaged F_0.5 on validation data.
    
    Args:
        pair_df: DataFrame with columns ['source1_entity_id', 'candidate_id', 'prob']
        ground_truth: dict mapping source1_entity_id -> collection of true match IDs
        
    Returns:
        tuple (best_threshold, best_f05_score, threshold_scores_dict)
    """
    thresholds = np.arange(min_thresh, max_thresh + step, step)
    best_threshold = 0.50
    best_score = -1.0
    history = {}

    # Pre-group pairs by source1_entity_id for fast sweeping
    grouped = defaultdict(list)
    for row in pair_df.itertuples(index=False):
        grouped[row.source1_entity_id].append((row.candidate_id, row.prob))

    all_s1_ids = list(ground_truth.keys())

    for t in thresholds:
        preds = {}
        for s1_id in all_s1_ids:
            if s1_id in grouped:
                cands = [cid for cid, p in grouped[s1_id] if p >= t]
                preds[s1_id] = cands
            else:
                preds[s1_id] = []

        score = macro_f05_score(preds, ground_truth)
        history[round(float(t), 4)] = round(float(score), 4)

        if score > best_score:
            best_score = score
            best_threshold = round(float(t), 4)

    return best_threshold, best_score, history


def apply_decision_threshold(pair_df, threshold, all_s1_ids=None):
    """
    Apply threshold to candidate pair probabilities to produce final entity match lists.
    
    Args:
        pair_df: DataFrame with ['source1_entity_id', 'candidate_id', 'prob']
        threshold: float cutoff probability
        all_s1_ids: optional list/set of all required Source 1 IDs to ensure every S1 has a row
        
    Returns:
        dict mapping source1_entity_id -> list of matched candidate IDs (empty for singletons)
    """
    matches = defaultdict(list)
    
    if pair_df is not None and not pair_df.empty:
        filtered = pair_df[pair_df["prob"] >= threshold]
        for row in filtered.itertuples(index=False):
            matches[row.source1_entity_id].append(row.candidate_id)

    # Ensure every requested S1 entity is present in output mapping
    final_mapping = {}
    target_ids = all_s1_ids if all_s1_ids is not None else list(matches.keys())
    for s1_id in target_ids:
        # Deduplicate matches while preserving order
        seen = set()
        deduped = []
        for mid in matches.get(s1_id, []):
            if mid not in seen:
                seen.add(mid)
                deduped.append(mid)
        final_mapping[s1_id] = deduped

    return final_mapping


def save_threshold_config(threshold, f05_score, filepath):
    """Save threshold metadata to JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    data = {
        "optimal_threshold": float(threshold),
        "validation_macro_f05": float(f05_score),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_threshold_config(filepath, default_threshold=0.50):
    """Load threshold metadata from JSON file, falling back to default."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("optimal_threshold", default_threshold)
    return default_threshold
