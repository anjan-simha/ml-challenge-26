import pandas as pd
import numpy as np
import re
from rapidfuzz.distance import Levenshtein
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import time

print("Testing End-to-End Feature Extraction and Model Training...")

s1 = pd.read_pickle('sample_s1.pkl')
s23 = pd.read_pickle('sample_matches_s23.pkl')
gt = pd.read_pickle('sample_gt.pkl')

# Ground truth map: s1_id -> set of matched_ids
gt_map = {}
for _, r in gt.iterrows():
    if pd.notna(r.matched_entity_ids) and str(r.matched_entity_ids).strip():
        gt_map[r.source1_entity_id] = set(str(r.matched_entity_ids).split(','))
    else:
        gt_map[r.source1_entity_id] = set()

# Preprocessing helpers
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

ADDR_STOP = {
    'road', 'street', 'avenue', 'lane', 'drive', 'floor', 'plot', 'sector', 'block',
    'unit', 'near', 'beside', 'opposite', 'city', 'district', 'state', 'india', 'us',
    'usa', 'france', 'and', 'the', 'in', 'of'
}

def clean_norm(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    for p, r in ABBREVIATIONS.items():
        text = re.sub(p, r, text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_tokens(norm_text):
    return [t for t in norm_text.split() if len(t) > 1]

def get_sig_tokens(norm_text):
    return [t for t in get_tokens(norm_text) if t not in STOP_WORDS and len(t) > 2]

def get_addr_sig(norm_text):
    return [t for t in get_tokens(norm_text) if t not in ADDR_STOP and len(t) > 2]

def get_numbers(text):
    if not isinstance(text, str):
        return set()
    return set(re.findall(r'\b\d+\b', text))

def jaccard(set1, set2):
    if not set1 or not set2:
        return 0.0
    u = len(set1.union(set2))
    return len(set1.intersection(set2)) / u if u > 0 else 0.0

# Prepare dictionaries
s1['name_norm'] = s1.business_name.apply(clean_norm)
s1['addr_norm'] = s1.business_address.apply(clean_norm)
s1['name_tok'] = s1.name_norm.apply(get_tokens)
s1['name_sig'] = s1.name_norm.apply(get_sig_tokens)
s1['addr_tok'] = s1.addr_norm.apply(get_tokens)
s1['addr_sig'] = s1.addr_norm.apply(get_addr_sig)
s1['nums'] = s1.business_address.apply(get_numbers)

s23['name_norm'] = s23.business_name.apply(clean_norm)
s23['addr_norm'] = s23.business_address.apply(clean_norm)
s23['name_tok'] = s23.name_norm.apply(get_tokens)
s23['name_sig'] = s23.name_norm.apply(get_sig_tokens)
s23['addr_tok'] = s23.addr_norm.apply(get_tokens)
s23['addr_sig'] = s23.addr_norm.apply(get_addr_sig)
s23['nums'] = s23.business_address.apply(get_numbers)

s1_dict = s1.set_index('entity_id').to_dict('index')
s23_dict = s23.set_index('entity_id').to_dict('index')

# Build Candidate Pairs (Positives + Hard Negatives from blocking)
print("Building blocking index on S23...")
from collections import defaultdict
token_index = defaultdict(list)
num_token_index = defaultdict(list)

for idx, r in s23.iterrows():
    cid = r.entity_id
    c = r.country
    for tok in r.name_sig[:2]:
        token_index[(c, tok)].append(cid)
    if r.nums and r.name_sig:
        num_token_index[(c, list(r.nums)[0], r.name_sig[0])].append(cid)
    if r.nums and r.addr_sig:
        num_token_index[(c, list(r.nums)[0], r.addr_sig[0])].append(cid)

print("Forming pairs...")
pairs_list = []
# Pick 1,000 S1 records that have matches in our s23 pool, plus some singletons
sample_s1_ids = list(s1_dict.keys())[:3000]

for s1_id in sample_s1_ids:
    s1_rec = s1_dict[s1_id]
    c = s1_rec['country']
    true_m = gt_map.get(s1_id, set())
    
    # Retrieve candidates
    cand_ids = set()
    for tok in s1_rec['name_sig'][:2]:
        cand_ids.update(token_index.get((c, tok), [])[:30])
    if s1_rec['nums'] and s1_rec['name_sig']:
        cand_ids.update(num_token_index.get((c, list(s1_rec['nums'])[0], s1_rec['name_sig'][0]), [])[:20])
    if s1_rec['nums'] and s1_rec['addr_sig']:
        cand_ids.update(num_token_index.get((c, list(s1_rec['nums'])[0], s1_rec['addr_sig'][0]), [])[:20])
        
    # Also add true matches present in s23 so training has positives
    for tm in true_m:
        if tm in s23_dict:
            cand_ids.add(tm)
            
    for cid in cand_ids:
        label = 1 if cid in true_m else 0
        pairs_list.append((s1_id, cid, label))

print(f"Total candidate pairs: {len(pairs_list)}")
pairs_df = pd.DataFrame(pairs_list, columns=['s1_id', 'cand_id', 'label'])
print("Label distribution:")
print(pairs_df.label.value_counts())

# Compute Features
print("Computing pairwise features...")
features = []
for _, r in pairs_df.iterrows():
    rec1 = s1_dict[r.s1_id]
    rec2 = s23_dict[r.cand_id]
    
    # Name features
    name_jacc = jaccard(set(rec1['name_tok']), set(rec2['name_tok']))
    name_lev = Levenshtein.normalized_similarity(rec1['name_norm'], rec2['name_norm'])
    name_exact = 1.0 if rec1['name_norm'] and rec1['name_norm'] == rec2['name_norm'] else 0.0
    name_overlap = len(set(rec1['name_sig']).intersection(set(rec2['name_sig'])))
    p1 = rec1['name_norm'].replace(' ', '')[:4]
    p2 = rec2['name_norm'].replace(' ', '')[:4]
    pref_match = 1.0 if p1 and p2 and p1 == p2 else 0.0
    name_len_diff = abs(len(rec1['name_norm']) - len(rec2['name_norm']))
    
    # Address features
    addr_jacc = jaccard(set(rec1['addr_tok']), set(rec2['addr_tok']))
    addr_lev = Levenshtein.normalized_similarity(rec1['addr_norm'], rec2['addr_norm'])
    addr_overlap = len(set(rec1['addr_sig']).intersection(set(rec2['addr_sig'])))
    num_jacc = jaccard(rec1['nums'], rec2['nums'])
    addr_exact = 1.0 if rec1['addr_norm'] and rec1['addr_norm'] == rec2['addr_norm'] else 0.0
    addr_len_diff = abs(len(rec1['addr_norm']) - len(rec2['addr_norm']))
    
    country_match = 1.0 if rec1['country'] == rec2['country'] else 0.0
    
    features.append([
        name_jacc, name_lev, name_exact, name_overlap, pref_match, name_len_diff,
        addr_jacc, addr_lev, addr_overlap, num_jacc, addr_exact, addr_len_diff,
        country_match
    ])

feat_cols = [
    'name_jacc', 'name_lev', 'name_exact', 'name_overlap', 'pref_match', 'name_len_diff',
    'addr_jacc', 'addr_lev', 'addr_overlap', 'num_jacc', 'addr_exact', 'addr_len_diff',
    'country_match'
]
X = pd.DataFrame(features, columns=feat_cols)
y = pairs_df.label
groups = pairs_df.s1_id

# GroupKFold Split
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X, y, groups))

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
val_pairs = pairs_df.iloc[val_idx].copy()

# Train LightGBM model
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.08,
    'num_leaves': 31,
    'verbosity': -1,
    'random_state': 42
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=200,
    callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
)

val_pairs['prob'] = model.predict(X_val)

# Threshold Tuning for F_0.5
def compute_f05(precision, recall):
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

val_s1_set = set(val_pairs.s1_id)
# Also include singletons in validation evaluation!
val_truth = {sid: gt_map.get(sid, set()) for sid in val_s1_set}

print("\n--- Tuning Decision Threshold for Macro F_0.5 ---")
best_thresh = 0.5
best_macro_f05 = 0.0

for t in np.arange(0.30, 0.95, 0.05):
    # Predict matches for each S1
    preds = defaultdict(set)
    for sid, group in val_pairs.groupby('s1_id'):
        matched = set(group[group.prob >= t]['cand_id'])
        preds[sid] = matched
        
    scores = []
    for sid, truth in val_truth.items():
        pred = preds.get(sid, set())
        if not truth:  # singleton
            scores.append(1.0 if not pred else 0.0)
        else:
            if not pred:
                scores.append(0.0)
            else:
                inter = len(pred.intersection(truth))
                p = inter / len(pred)
                r = inter / len(truth)
                scores.append(compute_f05(p, r))
                
    macro_f05 = np.mean(scores)
    print(f"Threshold: {t:.2f} -> Macro F_0.5: {macro_f05:.4f}")
    if macro_f05 > best_macro_f05:
        best_macro_f05 = macro_f05
        best_thresh = t

print(f"\n>>> Best Threshold: {best_thresh:.2f} with Macro F_0.5: {best_macro_f05:.4f} <<<")

# Feature importances
print("\nFeature Importances:")
imp = pd.Series(model.feature_importance(importance_type='gain'), index=feat_cols).sort_values(ascending=False)
print(imp)
