import re
import pandas as pd
import numpy as np
from collections import defaultdict
import time

print("Starting Blocking Exploration & Recall Benchmark...")
t0 = time.time()

# Common legal and generic stop words
STOP_WORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'llc', 'llp', 'gmbh', 'sa', 'sarl', 'bv', 'the', 'and', 'of', 'in',
    'services', 'service', 'solutions', 'enterprises', 'enterprise', 'technologies',
    'technology', 'consulting', 'consultants', 'group', 'holdings', 'international',
    'industries', 'india', 'us', 'usa'
}

ABBREVIATIONS = {
    r'\bpvt\.?\b': 'private',
    r'\bltd\.?\b': 'limited',
    r'\binc\.?\b': 'incorporated',
    r'\bcorp\.?\b': 'corporation',
    r'\bco\.?\b': 'company',
    r'\brd\.?\b': 'road',
    r'\bst\.?\b': 'street',
    r'\bave\.?\b': 'avenue',
    r'\bblvd\.?\b': 'boulevard',
    r'\bln\.?\b': 'lane',
    r'\bdr\.?\b': 'drive',
    r'\bhwy\.?\b': 'highway',
    r'\bct\.?\b': 'court',
    r'\bpl\.?\b': 'place',
    r'\bapt\.?\b': 'apartment',
    r'\bste\.?\b': 'suite',
    r'\bsec\.?\b': 'sector',
    r'\bfl\.?\b': 'floor',
    r'\bbldg\.?\b': 'building',
}

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for pattern, repl in ABBREVIATIONS.items():
        text = re.sub(pattern, repl, text)
    # Remove punctuation
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_tokens(norm_text):
    return [t for t in norm_text.split() if len(t) > 1]

def get_significant_tokens(norm_text):
    return [t for t in get_tokens(norm_text) if t not in STOP_WORDS and len(t) > 2]

def get_numbers(text):
    if not isinstance(text, str):
        return []
    return re.findall(r'\b\d+\b', text)

train_dir = r'student_resource\student_resource\dataset\train'

print("Loading 25,000 S1 records, 100,000 S2, 100,000 S3...")
s1 = pd.read_csv(f'{train_dir}/train_source1.tsv', sep='\t', nrows=25000)
gt = pd.read_csv(f'{train_dir}/train_ground_truth.tsv', sep='\t', nrows=25000)

# Filter gt for S1 entities loaded
s1_set = set(s1.entity_id)
gt = gt[gt.source1_entity_id.isin(s1_set)]

# Parse ground truth
gt_dict = {}
all_matched_ids = set()
for _, r in gt.iterrows():
    if pd.notna(r.matched_entity_ids) and str(r.matched_entity_ids).strip():
        m_list = str(r.matched_entity_ids).split(',')
        gt_dict[r.source1_entity_id] = set(m_list)
        all_matched_ids.update(m_list)
    else:
        gt_dict[r.source1_entity_id] = set()

print(f"Loaded {len(s1)} S1 entities. Ground truth has {len(all_matched_ids)} total matched IDs.")

# Load matching S2 and S3 records
print("Loading S2 and S3...")
s2 = pd.read_csv(f'{train_dir}/train_source2.tsv', sep='\t', nrows=200000)
s3 = pd.read_csv(f'{train_dir}/train_source3.tsv', sep='\t', nrows=200000)

s23 = pd.concat([s2, s3], ignore_index=True)
s23_dict = set(s23.entity_id)

# Only evaluate recall on matches that are present in s23 sample
evaluable_gt = {}
evaluable_matches_count = 0
for s1_id, mset in gt_dict.items():
    present_matches = mset.intersection(s23_dict)
    evaluable_gt[s1_id] = present_matches
    evaluable_matches_count += len(present_matches)

print(f"Evaluable ground truth matches in S2/S3 sample: {evaluable_matches_count}")

# Preprocess
print("Preprocessing records...")
s1['name_norm'] = s1.business_name.apply(normalize_text)
s1['addr_norm'] = s1.business_address.apply(normalize_text)
s1['sig_tokens'] = s1.name_norm.apply(get_significant_tokens)
s1['addr_nums'] = s1.business_address.apply(get_numbers)

s23['name_norm'] = s23.business_name.apply(normalize_text)
s23['addr_norm'] = s23.business_address.apply(normalize_text)
s23['sig_tokens'] = s23.name_norm.apply(get_significant_tokens)
s23['addr_nums'] = s23.business_address.apply(get_numbers)

print("Building Inverted Index on S2/S3...")
# Index 1: (country, token) for top significant name tokens
token_index = defaultdict(list)
# Index 2: (country, name_prefix)
prefix_index = defaultdict(list)
# Index 3: (country, addr_num, first_token)
num_token_index = defaultdict(list)

for idx, r in s23.iterrows():
    cand_id = r.entity_id
    c = r.country
    
    # Significant tokens (up to 3)
    for tok in r.sig_tokens[:3]:
        token_index[(c, tok)].append(cand_id)
        
    # Name prefix (first 6 chars of normalized name without spaces)
    n_clean = r.name_norm.replace(' ', '')
    if len(n_clean) >= 4:
        prefix_index[(c, n_clean[:5])].append(cand_id)
        
    # Address number + first token
    if r.addr_nums and r.sig_tokens:
        num_token_index[(c, r.addr_nums[0], r.sig_tokens[0])].append(cand_id)

print(f"Indexed {len(s23)} candidates. Index size: {len(token_index)} token keys, {len(prefix_index)} prefix keys.")

# Query S1 against index
print("Querying S1 candidates...")
candidates = defaultdict(set)
for idx, r in s1.iterrows():
    s1_id = r.entity_id
    c = r.country
    
    # 1. Sig tokens
    for tok in r.sig_tokens[:3]:
        key = (c, tok)
        if key in token_index:
            # Cap candidates per key to prevent explosive blocks
            cand_list = token_index[key]
            if len(cand_list) <= 100:
                candidates[s1_id].update(cand_list)
            else:
                candidates[s1_id].update(cand_list[:50])
                
    # 2. Name prefix
    n_clean = r.name_norm.replace(' ', '')
    if len(n_clean) >= 4:
        key = (c, n_clean[:5])
        if key in prefix_index:
            candidates[s1_id].update(prefix_index[key][:50])
            
    # 3. Addr num + token
    if r.addr_nums and r.sig_tokens:
        key = (c, r.addr_nums[0], r.sig_tokens[0])
        if key in num_token_index:
            candidates[s1_id].update(num_token_index[key][:50])

# Measure recall
found_matches = 0
total_candidates = sum(len(cset) for cset in candidates.values())
avg_candidates = total_candidates / len(s1)

for s1_id, mset in evaluable_gt.items():
    if mset:
        cset = candidates[s1_id]
        found_matches += len(mset.intersection(cset))

recall = found_matches / evaluable_matches_count if evaluable_matches_count > 0 else 0
print(f"\n=== BLOCKING BENCHMARK RESULTS ===")
print(f"Evaluable True Matches: {evaluable_matches_count}")
print(f"Recovered by Blocking : {found_matches}")
print(f"Candidate Recall      : {recall:.2%}")
print(f"Avg Candidates / S1   : {avg_candidates:.1f}")
print(f"Total time            : {time.time() - t0:.1f}s")
