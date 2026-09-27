import pandas as pd
import re
from rapidfuzz.distance import Levenshtein

s1 = pd.read_pickle('sample_s1.pkl')
s23 = pd.read_pickle('sample_matches_s23.pkl')
gt = pd.read_pickle('sample_gt.pkl')

print(f"Loaded {len(s1)} S1, {len(s23)} S2/S3 true matches.")

# Invert ground truth: cand_id -> s1_id
gt_cand_to_s1 = {}
for _, r in gt.iterrows():
    if pd.notna(r.matched_entity_ids):
        for mid in str(r.matched_entity_ids).split(','):
            gt_cand_to_s1[mid] = r.source1_entity_id

s1_dict = s1.set_index('entity_id').to_dict('index')

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

STOP_WORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'llc', 'llp', 'gmbh', 'sa', 'sarl', 'the', 'and', 'of', 'in',
    'services', 'service', 'solutions', 'enterprises', 'enterprise', 'technologies',
    'technology', 'consulting', 'consultants', 'group', 'holdings'
}

def norm_name(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for p, r in ABBREVIATIONS.items():
        text = re.sub(p, r, text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_sig_tokens(text):
    norm = norm_name(text)
    return [t for t in norm.split() if t not in STOP_WORDS and len(t) > 1]

def get_nums(text):
    if not isinstance(text, str):
        return set()
    return set(re.findall(r'\b\d+\b', text))

pairs = []
for _, cand in s23.iterrows():
    cand_id = cand.entity_id
    if cand_id in gt_cand_to_s1:
        s1_id = gt_cand_to_s1[cand_id]
        if s1_id in s1_dict:
            pairs.append((s1_dict[s1_id], cand))

print(f"Formed {len(pairs)} matched pairs for analysis.")

exact_name_cnt = 0
token_overlap_cnt = 0
prefix_overlap_cnt = 0
num_overlap_cnt = 0
lev_name_sum = 0
lev_addr_sum = 0

missed_cases = []

for s1_rec, cand_rec in pairs:
    n1 = norm_name(s1_rec['business_name'])
    n2 = norm_name(cand_rec['business_name'])
    
    t1 = set(get_sig_tokens(s1_rec['business_name']))
    t2 = set(get_sig_tokens(cand_rec['business_name']))
    
    nums1 = get_nums(s1_rec['business_address'])
    nums2 = get_nums(cand_rec['business_address'])
    
    # Exact
    if n1 == n2:
        exact_name_cnt += 1
        
    # Token overlap
    tok_match = bool(t1.intersection(t2))
    if tok_match:
        token_overlap_cnt += 1
        
    # Prefix (first 4 chars of cleaned name)
    p1 = n1.replace(' ', '')[:4]
    p2 = n2.replace(' ', '')[:4]
    pref_match = (p1 and p2 and p1 == p2)
    if pref_match:
        prefix_overlap_cnt += 1
        
    # Number overlap
    if nums1 and nums2 and nums1.intersection(nums2):
        num_overlap_cnt += 1
        
    lev_name = Levenshtein.normalized_similarity(n1, n2)
    lev_addr = Levenshtein.normalized_similarity(norm_name(s1_rec['business_address']), norm_name(cand_rec['business_address']))
    lev_name_sum += lev_name
    lev_addr_sum += lev_addr
    
    # Check if BOTH token AND prefix missed
    if not tok_match and not pref_match:
        missed_cases.append((s1_rec, cand_rec, n1, n2))

total = len(pairs)
print("\n=== MATCHING PAIR ANALYSIS ===")
print(f"Exact Normalized Name Match: {exact_name_cnt}/{total} ({exact_name_cnt/total:.2%})")
print(f"At Least 1 Sig Token Match : {token_overlap_cnt}/{total} ({token_overlap_cnt/total:.2%})")
print(f"4-Char Name Prefix Match   : {prefix_overlap_cnt}/{total} ({prefix_overlap_cnt/total:.2%})")
print(f"Token OR Prefix Match      : {(total - len(missed_cases))}/{total} ({(total - len(missed_cases))/total:.2%})")
print(f"Address Number Overlap     : {num_overlap_cnt}/{total} ({num_overlap_cnt/total:.2%})")
print(f"Mean Name Levenshtein Sim  : {lev_name_sum/total:.4f}")
print(f"Mean Addr Levenshtein Sim  : {lev_addr_sum/total:.4f}")

print(f"\nTotal missed by Token AND 4-char Prefix: {len(missed_cases)} ({len(missed_cases)/total:.2%})")
print("Here are 10 sample missed cases:")
for s1_rec, cand_rec, n1, n2 in missed_cases[:10]:
    try:
        print(f"\n  S1  : '{n1}' | Addr: '{s1_rec['business_address']}'")
        print(f"  Cand: '{n2}' | Addr: '{cand_rec['business_address']}'")
    except Exception:
        pass
