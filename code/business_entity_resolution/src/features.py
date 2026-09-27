"""
ML Challenge 2026: Business Entity Resolution
Pairwise Feature Engineering Module

Computes detailed textual, fuzzy, token, and numeric similarity features
for candidate pairs (Source 1 <-> Candidate from Source 2 / Source 3).
"""

import numpy as np
from rapidfuzz.distance import Levenshtein


FEATURE_NAMES = [
    "name_jaccard",
    "name_levenshtein",
    "name_exact_match",
    "name_token_overlap",
    "name_token_recall",
    "name_prefix_match",
    "name_len_diff",
    "addr_jaccard",
    "addr_levenshtein",
    "addr_exact_match",
    "addr_token_overlap",
    "addr_number_jaccard",
    "addr_number_exact",
    "addr_len_diff",
    "country_match",
    "combined_char_jaccard",
]


def jaccard_similarity(set_a, set_b):
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def extract_char_ngrams(text, n=3):
    """Extract character n-grams from a string."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def compute_pair_features(s1_rec, cand_rec):
    """
    Extract pairwise feature vector for a (Source 1, Candidate) pair.
    
    Args:
        s1_rec: dict of preprocessed features for Source 1 entity
        cand_rec: dict of preprocessed features for Candidate entity
        
    Returns:
        list of float values matching FEATURE_NAMES
    """
    n1_norm, n2_norm = s1_rec["name_norm"], cand_rec["name_norm"]
    a1_norm, a2_norm = s1_rec["addr_norm"], cand_rec["addr_norm"]

    t1_tokens = set(s1_rec["name_tokens"])
    t2_tokens = set(cand_rec["name_tokens"])
    s1_sig = set(s1_rec["name_sig"])
    s2_sig = set(cand_rec["name_sig"])

    # 1. Name Features
    name_jacc = jaccard_similarity(t1_tokens, t2_tokens)
    name_lev = Levenshtein.normalized_similarity(n1_norm, n2_norm) if (n1_norm or n2_norm) else 0.0
    name_exact = 1.0 if (n1_norm and n1_norm == n2_norm) else 0.0

    sig_intersection = len(s1_sig.intersection(s2_sig))
    name_tok_overlap = float(sig_intersection)
    min_sig_len = min(len(s1_sig), len(s2_sig)) if (s1_sig and s2_sig) else 0
    name_tok_recall = (sig_intersection / min_sig_len) if min_sig_len > 0 else 0.0

    p1, p2 = s1_rec["name_prefix"], cand_rec["name_prefix"]
    name_prefix_match = 1.0 if (p1 and p2 and p1 == p2) else 0.0
    name_len_diff = float(abs(len(n1_norm) - len(n2_norm)))

    # 2. Address Features
    a1_tokens = set(s1_rec["addr_tokens"])
    a2_tokens = set(cand_rec["addr_tokens"])
    a1_sig = set(s1_rec["addr_sig"])
    a2_sig = set(cand_rec["addr_sig"])

    addr_jacc = jaccard_similarity(a1_tokens, a2_tokens)
    addr_lev = Levenshtein.normalized_similarity(a1_norm, a2_norm) if (a1_norm or a2_norm) else 0.0
    addr_exact = 1.0 if (a1_norm and a1_norm == a2_norm) else 0.0
    addr_tok_overlap = float(len(a1_sig.intersection(a2_sig)))

    nums1 = set(s1_rec["addr_numbers"])
    nums2 = set(cand_rec["addr_numbers"])
    addr_num_jacc = jaccard_similarity(nums1, nums2)
    
    first_num1 = s1_rec["addr_numbers"][0] if s1_rec["addr_numbers"] else ""
    first_num2 = cand_rec["addr_numbers"][0] if cand_rec["addr_numbers"] else ""
    addr_num_exact = 1.0 if (first_num1 and first_num1 == first_num2) else 0.0
    addr_len_diff = float(abs(len(a1_norm) - len(a2_norm)))

    # 3. Country Match
    c1 = s1_rec["country"]
    c2 = cand_rec["country"]
    country_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0

    # 4. Combined Character 3-Gram Jaccard
    comb1 = f"{n1_norm} {a1_norm}".strip()
    comb2 = f"{n2_norm} {a2_norm}".strip()
    char_ngrams1 = extract_char_ngrams(comb1, n=3)
    char_ngrams2 = extract_char_ngrams(comb2, n=3)
    comb_char_jacc = jaccard_similarity(char_ngrams1, char_ngrams2)

    return [
        name_jacc,
        name_lev,
        name_exact,
        name_tok_overlap,
        name_tok_recall,
        name_prefix_match,
        name_len_diff,
        addr_jacc,
        addr_lev,
        addr_exact,
        addr_tok_overlap,
        addr_num_jacc,
        addr_num_exact,
        addr_len_diff,
        country_match,
        comb_char_jacc,
    ]


def batch_compute_features(pairs, s1_dict, cand_dict):
    """
    Compute feature matrix for a list of (s1_id, cand_id) pairs.
    
    Args:
        pairs: list of tuples (s1_id, cand_id)
        s1_dict: dict mapping s1_id -> preprocessed s1 record
        cand_dict: dict mapping cand_id -> preprocessed candidate record
        
    Returns:
        np.ndarray of shape (len(pairs), len(FEATURE_NAMES))
    """
    rows = []
    for s1_id, cand_id in pairs:
        rec1 = s1_dict[s1_id]
        rec2 = cand_dict[cand_id]
        rows.append(compute_pair_features(rec1, rec2))

    return np.array(rows, dtype=np.float32)
