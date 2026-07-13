import os, shutil, random

RAW = r"G:\KMS\rag-bot-fastapi\server\data\raw"
CURATED = r"G:\KMS\rag-bot-fastapi\server\data\curated"

os.makedirs(CURATED, exist_ok=True)
os.makedirs(os.path.join(CURATED, "clinical_trials"), exist_ok=True)
os.makedirs(os.path.join(CURATED, "faers"), exist_ok=True)

# 1. Copy PubMed abstracts (all of it, it's small and valuable)
pubmed_src = os.path.join(RAW, "pubmed_abstracts.txt")
if os.path.exists(pubmed_src):
    shutil.copy(pubmed_src, CURATED)
    print("Copied PubMed abstracts.")

# 2. Sample ~40,000 clinical trial files from nested subfolders
trials_root = os.path.join(RAW, "clinical_trials")
all_trial_files = []
for subfolder in os.listdir(trials_root):
    subfolder_path = os.path.join(trials_root, subfolder)
    if os.path.isdir(subfolder_path):
        for f in os.listdir(subfolder_path):
            if f.endswith(".json"):
                all_trial_files.append(os.path.join(subfolder_path, f))

print(f"Found {len(all_trial_files)} total trial files.")
sample_size = min(40000, len(all_trial_files))
sampled = random.sample(all_trial_files, sample_size)

for i, src_path in enumerate(sampled):
    dest_path = os.path.join(CURATED, "clinical_trials", os.path.basename(src_path))
    shutil.copy(src_path, dest_path)
    if i % 5000 == 0:
        print(f"Copied {i}/{sample_size} trial files...")

print(f"Copied {sample_size} clinical trial files.")

# 3. Copy 3 of the 6 FAERS files (~600MB-700MB)
faers_root = os.path.join(RAW, "faers")
faers_files = sorted([f for f in os.listdir(faers_root) if f.endswith(".json")])
for f in faers_files[:3]:
    shutil.copy(os.path.join(faers_root, f), os.path.join(CURATED, "faers", f))
    print(f"Copied {f}")

print("Curation complete.")