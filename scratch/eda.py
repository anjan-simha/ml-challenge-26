import pandas as pd

train_dir = r'student_resource\student_resource\dataset\train'
test_dir = r'student_resource\student_resource\dataset\test'

s1 = pd.read_csv(f'{train_dir}/train_source1.tsv', sep='\t', nrows=20000)
s2 = pd.read_csv(f'{train_dir}/train_source2.tsv', sep='\t', nrows=50000)
s3 = pd.read_csv(f'{train_dir}/train_source3.tsv', sep='\t', nrows=50000)
gt = pd.read_csv(f'{train_dir}/train_ground_truth.tsv', sep='\t', nrows=20000)

print("=== S1 Info ===")
print(s1.info())
print("\nS1 Country counts:")
print(s1['country'].value_counts(dropna=False))

print("\n=== Test S1 Info ===")
test_s1 = pd.read_csv(f'{test_dir}/test_source1.tsv', sep='\t', nrows=20000)
print(test_s1['country'].value_counts(dropna=False))

# Create lookup dicts
s1_dict = s1.set_index('entity_id').to_dict('index')
s2_dict = s2.set_index('entity_id').to_dict('index')
s3_dict = s3.set_index('entity_id').to_dict('index')

print("\n=== 15 SAMPLE MATCHING PAIRS ===")
samples_shown = 0
for _, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    matched = str(row['matched_entity_ids'])
    if matched and matched != 'nan' and s1_id in s1_dict:
        m_ids = matched.split(',')
        # Check if we have at least one match loaded
        loaded_m = [mid for mid in m_ids if (mid in s2_dict or mid in s3_dict)]
        if loaded_m:
            s1_rec = s1_dict[s1_id]
            print(f"\n--- Entity S1: {s1_id} ({s1_rec['country']}) ---")
            print(f"  S1 Name   : {s1_rec['business_name']}")
            print(f"  S1 Address: {s1_rec['business_address']}")
            for mid in loaded_m[:3]:
                rec = s2_dict[mid] if mid in s2_dict else s3_dict[mid]
                print(f"  -> Match {mid} ({rec['country']}):")
                print(f"     Name   : {rec['business_name']}")
                print(f"     Address: {rec['business_address']}")
            samples_shown += 1
            if samples_shown >= 12:
                break
