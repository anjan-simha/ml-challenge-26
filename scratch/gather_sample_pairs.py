import pandas as pd
import re
from collections import defaultdict

train_dir = r'student_resource\student_resource\dataset\train'

print("Loading S1 and Ground Truth (50,000 rows)...")
s1 = pd.read_csv(f'{train_dir}/train_source1.tsv', sep='\t', nrows=50000)
gt = pd.read_csv(f'{train_dir}/train_ground_truth.tsv', sep='\t', nrows=50000)

# Build map: s1_id -> set of matched_ids
gt_map = {}
needed_matches = set()
for _, r in gt.iterrows():
    if pd.notna(r.matched_entity_ids) and str(r.matched_entity_ids).strip():
        m_list = str(r.matched_entity_ids).split(',')
        gt_map[r.source1_entity_id] = m_list
        needed_matches.update(m_list)

print(f"Targeting {len(needed_matches)} true matching IDs for 50,000 S1 entities.")

# Scan S2 and S3 in chunks to find the true matching records
matched_records = []
for source_file in ['train_source2.tsv', 'train_source3.tsv']:
    print(f"Scanning {source_file} in chunks of 500,000...")
    for chunk in pd.read_csv(f'{train_dir}/{source_file}', sep='\t', chunksize=500000):
        found = chunk[chunk.entity_id.isin(needed_matches)]
        if len(found) > 0:
            matched_records.append(found)
            print(f"  Found {len(found)} true matches (Total accumulated: {sum(len(x) for x in matched_records)})")
        if sum(len(x) for x in matched_records) >= 2000:
            break

matched_df = pd.concat(matched_records, ignore_index=True)
print(f"Total true matching records gathered: {len(matched_df)}")

# Now save this small paired dataset for rapid offline development & analysis!
s1_subset = s1[s1.entity_id.isin(gt_map.keys())]
matched_df.to_pickle('sample_matches_s23.pkl')
s1_subset.to_pickle('sample_s1.pkl')
gt.to_pickle('sample_gt.pkl')
print("Saved sample_matches_s23.pkl, sample_s1.pkl, sample_gt.pkl successfully!")
