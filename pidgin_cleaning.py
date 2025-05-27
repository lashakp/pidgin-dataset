import pandas as pd
from datasets import load_dataset
import re

# --------------------------------------------
# 🧩 STEP 1: Load Dataset from Hugging Face Hub
# --------------------------------------------
dataset = load_dataset("Tommy0201/Pidgin_English_AmazonMTurk", split="train")
print("Dataset loaded:\n", dataset)

# --------------------------------------------
# 🔍 STEP 2: Preview Sample Entry
# --------------------------------------------
print("\nSample entry:\n", dataset[0])

# --------------------------------------------
# 📊 STEP 3: Convert to DataFrame for Analysis
# --------------------------------------------
df = dataset.to_pandas()
print(df.head())
print(df.shape)
print(df.columns)
print(df.describe())
print(df.dtypes)
print(df.isnull().sum())
print(df.sample(5))
df.info()

# --------------------------------------------
# ✅ STEP 4: Validate Text Entries
# --------------------------------------------
def is_valid(sample):
    return sample.get('text') is not None and isinstance(sample['text'], str) and sample['text'].strip() != ""

# Remove samples with missing or invalid text
cleaned_dataset = dataset.filter(is_valid)

# --------------------------------------------
# 🔠 STEP 5: Normalize the Text
# --------------------------------------------
def normalize_text(sample):
    text = sample["text"].lower()
    text = re.sub(r"[^a-z0-9\s']", "", text)  # Remove special characters and digits
    sample["text"] = text.strip()
    return sample

# Apply normalization to valid data
original_dataset = dataset
cleaned_dataset = cleaned_dataset.map(normalize_text)

# --------------------------------------------
# 🔁 STEP 6: Preview Changes (First 5 Samples)
# --------------------------------------------
for i in range(5):
    original = dataset[i]["text"]
    cleaned = cleaned_dataset[i]["text"]
    print(f"\nOriginal: {original}")
    print(f"Cleaned : {cleaned}")

# --------------------------------------------
# 📈 STEP 7: Analyze Cleaned Dataset
# --------------------------------------------
cleaned_df = cleaned_dataset.to_pandas()
cleaned_size = len(cleaned_df)
cleaned_avg_len = cleaned_df['text'].apply(lambda x: len(x.split())).mean()

original_df = dataset.to_pandas()
original_size = len(original_df)
original_avg_len = original_df['text'].apply(lambda x: len(x.split())).mean()

# --------------------------------------------
# 📉 STEP 8: Cleaning Summary
# --------------------------------------------
print("\n🧹 CLEANING SUMMARY")
print(f"Original dataset size: {original_size}")
print(f"Cleaned dataset size : {cleaned_size}")
print(f"Records removed      : {original_size - cleaned_size}")
print(f"Reduction (%)        : {(1 - cleaned_size/original_size) * 100:.2f}%")
print(f"Original avg length  : {original_avg_len:.2f} words")
print(f"Cleaned avg length   : {cleaned_avg_len:.2f} words")

# --------------------------------------------
# 🔍 STEP 9: Side-by-Side Text Comparison
# --------------------------------------------
print("\n📝 Sample Comparison:")
for i in range(5):
    try:
        original = df['text'].iloc[i]
        cleaned = cleaned_df['text'].iloc[i]
        print(f"\nOriginal: {original}")
        print(f"Cleaned : {cleaned}")
    except IndexError:
        break

# --------------------------------------------
# 📊 STEP 10: Track Total Changes Made
# --------------------------------------------
changed_count = 0
total = len(original_dataset)

for i in range(total):
    original = original_dataset[i]["text"]
    cleaned = cleaned_dataset[i]["text"]
    if original != cleaned:
        changed_count += 1

print(f"\nChanged sentences: {changed_count} / {total}")
print(f"Percentage changed: {changed_count / total * 100:.2f}%")
# save cleaned dataset
cleaned_dataset.save_to_disk("data/cleaned_dataset")

# export cleaned dataset to csv
cleaned_df.to_csv("data/cleaned_dataset.csv", index=False)