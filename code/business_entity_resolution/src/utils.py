"""
ML Challenge 2026: Business Entity Resolution
Utility functions for data I/O, evaluation metrics (F_0.5 macro score), and parsing.
"""

import os
import sys
import time
import pandas as pd
import numpy as np


def clean_str(val):
    """Safely convert value to stripped string."""
    if val is None or pd.isna(val):
        return ""
    return str(val).strip()


def parse_id_list(id_str):
    """Parse comma-separated entity IDs into a list of cleaned IDs."""
    if id_str is None or pd.isna(id_str):
        return []
    s = str(id_str).strip()
    if not s:
        return []
    return [item.strip() for item in s.split(",") if item.strip()]


def format_id_list(id_collection):
    """Format a list or set of entity IDs into a comma-separated string."""
    if not id_collection:
        return ""
    # Sort for deterministic output
    return ",".join(sorted(set(id_collection)))


def compute_f05(precision, recall):
    """
    Compute F_0.5 score:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    Weights precision 2x over recall.
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    denom = 0.25 * precision + recall
    if denom <= 0.0:
        return 0.0
    return (1.25 * precision * recall) / denom


def evaluate_entity_f05(pred_ids, true_ids):
    """
    Compute F_0.5 score for a single Source 1 entity according to challenge specification:
    - If entity has no true matches (singleton):
        returns 1.0 if pred_ids is empty, else 0.0.
    - If entity has true matches:
        returns 0.0 if pred_ids is empty;
        otherwise calculates precision, recall, and F_0.5.
    """
    pred_set = set(pred_ids) if pred_ids else set()
    true_set = set(true_ids) if true_ids else set()

    if not true_set:
        # Singleton entity
        return 1.0 if not pred_set else 0.0

    if not pred_set:
        # Missed all matches
        return 0.0

    true_positives = len(pred_set.intersection(true_set))
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(pred_set)
    recall = true_positives / len(true_set)
    return compute_f05(precision, recall)


def macro_f05_score(predictions, ground_truth):
    """
    Compute macro-averaged F_0.5 across all Source 1 entities in ground_truth.
    
    Args:
        predictions: dict mapping source1_entity_id -> collection of predicted match IDs
        ground_truth: dict mapping source1_entity_id -> collection of true match IDs
        
    Returns:
        float: macro-averaged F_0.5 score in [0.0, 1.0]
    """
    if not ground_truth:
        return 0.0

    scores = []
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, [])
        score = evaluate_entity_f05(pred_matches, true_matches)
        scores.append(score)

    return float(np.mean(scores))


class Timer:
    """Context manager and utility for logging elapsed execution time."""
    def __init__(self, description="Operation"):
        self.description = description
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Starting: {self.description}...", flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"[{time.strftime('%H:%M:%S')}] Completed: {self.description} in {elapsed:.2f}s", flush=True)
