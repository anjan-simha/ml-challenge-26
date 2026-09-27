"""
ML Challenge 2026: Business Entity Resolution
End-to-End Inference Pipeline

Processes test records country-by-country using high-performance streaming I/O.
Executes multi-rule blocking, feature engineering, and model inference.
Generates:
  1. output/candidate_pairs.tsv (candidate set passed to matching model)
  2. output/matching_results.tsv (final entity match predictions)
"""

import os
import sys
import gc
import argparse
import joblib
import numpy as np
from collections import defaultdict

from .utils import Timer, format_id_list
from .preprocessing import preprocess_record
from .blocking import MultiRuleBlocker
from .features import compute_pair_features, FEATURE_NAMES
from .thresholding import load_threshold_config


def stream_tsv_records(filepath, target_country=None, max_rows=None):
    """
    High-performance TSV line reader.
    Yields (entity_id, business_name, business_address, country).
    """
    if not os.path.exists(filepath):
        return

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        # Skip header
        header = f.readline()
        count = 0
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue

            entity_id, name, addr, country = parts[0], parts[1], parts[2], parts[3]
            c_clean = country.strip().upper()

            if target_country is not None and c_clean != target_country:
                continue

            yield {
                "entity_id": entity_id,
                "business_name": name,
                "business_address": addr,
                "country": c_clean,
            }

            count += 1
            if max_rows and count >= max_rows:
                break


def run_country_inference(country, s1_records, cand_records, model, threshold, cand_out_fp, match_out_fp):
    """
    Execute blocking, feature calculation, and thresholded prediction for one country partition.
    Streams results directly to candidate and matching output file descriptors.
    """
    print(f"[{country}] Building candidate indexes for {len(cand_records):,} candidates...", flush=True)
    blocker = MultiRuleBlocker(max_candidates_per_entity=20, max_postings_per_key=150)
    for c_rec in cand_records.values():
        blocker.add_candidate(c_rec)

    print(f"[{country}] Generating candidates and scoring {len(s1_records):,} S1 entities...", flush=True)

    batch_size = 5000
    s1_items = list(s1_records.items())
    total_s1 = len(s1_items)

    pairs_batch = []
    s1_cand_map = {}

    for i, (s1_id, s1_rec) in enumerate(s1_items):
        cands = blocker.retrieve_candidates(s1_rec)
        s1_cand_map[s1_id] = cands

        for cid in cands:
            pairs_batch.append((s1_id, cid))

        # Score in batches to avoid large memory footprints
        if len(pairs_batch) >= batch_size or (i == total_s1 - 1 and pairs_batch):
            # Extract features
            features = []
            for sid, cid in pairs_batch:
                rec1 = s1_records[sid]
                rec2 = cand_records[cid]
                features.append(compute_pair_features(rec1, rec2))

            X_batch = np.array(features, dtype=np.float32)
            probs = model.predict(X_batch)

            # Map probabilities back to pairs
            pair_probs = defaultdict(list)
            for (sid, cid), prob in zip(pairs_batch, probs):
                if prob >= threshold:
                    pair_probs[sid].append(cid)

            # Write predictions for processed entities
            for sid in set(p[0] for p in pairs_batch):
                cands_for_sid = s1_cand_map.pop(sid, [])
                matches_for_sid = pair_probs.get(sid, [])

                # Format strings
                cand_str = format_id_list(cands_for_sid)
                match_str = format_id_list(matches_for_sid)

                cand_out_fp.write(f"{sid}\t{cand_str}\n")
                match_out_fp.write(f"{sid}\t{match_str}\n")

            pairs_batch.clear()

        # Handle any entities in batch that had 0 candidates (singletons)
        for sid in list(s1_cand_map.keys()):
            if not s1_cand_map[sid]:
                cand_out_fp.write(f"{sid}\t\n")
                match_out_fp.write(f"{sid}\t\n")
                s1_cand_map.pop(sid)

        # Log progress periodically
        if (i + 1) % 50000 == 0 or (i + 1) == total_s1:
            print(f"[{country}] Processed {i + 1:,} / {total_s1:,} entities ({(i + 1) / total_s1:.1%})", flush=True)

    blocker.clear()
    gc.collect()


def run_inference_pipeline(test_dir, model_dir, output_dir, max_s1_limit=None):
    """
    Run full end-to-end inference over test datasets.
    
    Args:
        test_dir: directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv
        model_dir: directory containing entity_matcher_lgb.pkl and threshold_config.json
        output_dir: directory to write candidate_pairs.tsv and matching_results.tsv
        max_s1_limit: optional limit on S1 entities for rapid testing
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "entity_matcher_lgb.pkl")
    thresh_path = os.path.join(model_dir, "threshold_config.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run training first.")

    print(f"Loading model from {model_path}...", flush=True)
    model = joblib.load(model_path)
    threshold = load_threshold_config(thresh_path, default_threshold=0.89)
    print(f"Using decision threshold: {threshold:.4f}", flush=True)

    cand_out_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_out_path = os.path.join(output_dir, "matching_results.tsv")

    # Step 1: Discover all countries from test_source1.tsv and load S1 partitioned by country
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    print(f"Scanning test Source 1 from {s1_path}...", flush=True)

    s1_by_country = defaultdict(dict)
    total_s1_loaded = 0

    for raw_rec in stream_tsv_records(s1_path, max_rows=max_s1_limit):
        rec = preprocess_record(raw_rec)
        country = rec["country"]
        s1_by_country[country][rec["entity_id"]] = rec
        total_s1_loaded += 1

    countries = sorted(s1_by_country.keys())
    print(f"Loaded {total_s1_loaded:,} Source 1 test records across countries: {countries}", flush=True)

    # Initialize output TSVs with exact required headers
    with open(cand_out_path, "w", encoding="utf-8") as f_cand, \
         open(match_out_path, "w", encoding="utf-8") as f_match:

        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        # Process country-by-country
        for country in countries:
            s1_country_records = s1_by_country[country]
            print(f"\n==========================================", flush=True)
            print(f"Processing Country: {country} ({len(s1_country_records):,} S1 entities)", flush=True)
            print(f"==========================================", flush=True)

            # Load S2 and S3 for this country using streaming
            cand_records = {}
            max_cand_limit = (max_s1_limit * 100) if max_s1_limit else None
            for s_file in ["test_source2.tsv", "test_source3.tsv"]:
                s_path = os.path.join(test_dir, s_file)
                print(f"Streaming {s_file} for country {country}...", flush=True)
                for raw_rec in stream_tsv_records(s_path, target_country=country, max_rows=max_cand_limit):
                    rec = preprocess_record(raw_rec)
                    cand_records[rec.entity_id] = rec

            print(f"Loaded {len(cand_records):,} candidates for country {country}.", flush=True)

            # Run inference for this country partition
            run_country_inference(
                country=country,
                s1_records=s1_country_records,
                cand_records=cand_records,
                model=model,
                threshold=threshold,
                cand_out_fp=f_cand,
                match_out_fp=f_match,
            )

            # Free memory
            del cand_records
            s1_by_country[country].clear()
            gc.collect()

    print(f"\nInference completed successfully!", flush=True)
    print(f"Output files generated:")
    print(f"  - {match_out_path}")
    print(f"  - {cand_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Entity Resolution Inference on Test Data")
    parser.add_argument("--test-dir", type=str, default="dataset/test", help="Path to test data directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory containing trained model")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save output TSVs")
    parser.add_argument("--max-entities", type=int, default=None, help="Optional limit for rapid testing")
    args = parser.parse_args()

    run_inference_pipeline(
        test_dir=args.test_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        max_s1_limit=args.max_entities,
    )
