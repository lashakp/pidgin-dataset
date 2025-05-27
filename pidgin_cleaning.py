# 📦 Import necessary libraries
import pandas as pd                      # Used for data manipulation and analysis
from datasets import load_dataset        # To load datasets directly from Hugging Face Hub
import re                                # Provides regular expressions for text cleaning

# --------------------------------------------
# 🧩 STEP 1: Load Dataset from Hugging Face Hub
# --------------------------------------------
dataset = load_dataset("Tommy0201/Pidgin_English_AmazonMTurk", split="train")  # Load the 'train' split of the dataset
print("Dataset loaded:\n", dataset)                                            # Display basic information about the dataset

# --------------------------------------------
# 🔍 STEP 2: Preview a Sample Entry
# --------------------------------------------
print("\nSample entry:\n", dataset[0])  # Print the first entry in the dataset to examine structure

# --------------------------------------------
# 📊 STEP 3: Convert to DataFrame for Analysis
# --------------------------------------------
df = dataset.to_pandas()               # Convert Hugging Face Dataset object to pandas DataFrame for easier analysis

print(df.head())                       # Display first 5 rows of the DataFrame
print(df.shape)                        # Print total number of rows and columns
print(df.columns)                      # List column names (usually 'text' and 'label')
print(df.describe())                   # Show summary statistics (only for numeric columns, e.g., 'label')
print(df.dtypes)                       # Display data types of each column
print(df.isnull().sum())               # Show count of missing values in each column
print(df.sample(5))                    # Show 5 random rows to get a feel of the data
df.info()                              # Print concise summary: column types, non-null counts, etc.

# --------------------------------------------
# ✅ STEP 4: Validate Text Entries
# --------------------------------------------
# Define a function to check if a sample's 'text' is valid (non-empty and a string)
def is_valid(sample):
    return sample.get('text') is not None and isinstance(sample['text'], str) and sample['text'].strip() != ""

# Apply filter to remove rows with missing or invalid 'text'
cleaned_dataset = dataset.filter(is_valid)

# --------------------------------------------
# 🔠 STEP 5: Normalize the Text
# --------------------------------------------
# Define a function to normalize text by lowercasing and removing unwanted characters
def normalize_text(sample):
    text = sample["text"].lower()                          # Convert to lowercase
    text = re.sub(r"[^a-z0-9\s']", "", text)               # Remove special characters and digits (keep a-z, 0-9, space, apostrophe)
    sample["text"] = text.strip()                          # Strip leading/trailing whitespace
    return sample

# Keep the original dataset before cleaning
original_dataset = dataset

# Apply normalization function to each sample in the cleaned dataset
cleaned_dataset = cleaned_dataset.map(normalize_text)

# --------------------------------------------
# 🔁 STEP 6: Preview First 5 Changes
# --------------------------------------------
# Show side-by-side comparison of original vs cleaned text for the first 5 entries
for i in range(5):
    original = dataset[i]["text"]                          # Original text from uncleaned dataset
    cleaned = cleaned_dataset[i]["text"]                   # Cleaned text from cleaned dataset
    print(f"\nOriginal: {original}")                       # Print original
    print(f"Cleaned : {cleaned}")                          # Print cleaned version

# --------------------------------------------
# 📈 STEP 7: Analyze Cleaned Dataset
# --------------------------------------------
cleaned_df = cleaned_dataset.to_pandas()                   # Convert cleaned dataset to DataFrame

# Calculate number of entries and average sentence length for cleaned data
cleaned_size = len(cleaned_df)                            
cleaned_avg_len = cleaned_df['text'].apply(lambda x: len(x.split())).mean()

# Also get stats for the original dataset
original_df = dataset.to_pandas()
original_size = len(original_df)
original_avg_len = original_df['text'].apply(lambda x: len(x.split())).mean()

# --------------------------------------------
# 📉 STEP 8: Cleaning Summary
# --------------------------------------------
# Print comparison summary of original vs cleaned dataset
print("\n🧹 CLEANING SUMMARY")
print(f"Original dataset size: {original_size}")                          # Total entries before cleaning
print(f"Cleaned dataset size : {cleaned_size}")                           # Total entries after cleaning
print(f"Records removed      : {original_size - cleaned_size}")           # Number of records removed
print(f"Reduction (%)        : {(1 - cleaned_size/original_size) * 100:.2f}%")  # Percent reduction
print(f"Original avg length  : {original_avg_len:.2f} words")             # Average words per sentence before
print(f"Cleaned avg length   : {cleaned_avg_len:.2f} words")              # Average words per sentence after

# --------------------------------------------
# 🔍 STEP 9: Side-by-Side Text Comparison
# --------------------------------------------
# Print side-by-side comparison of 5 random rows
print("\n📝 Sample Comparison:")
for i in range(5):
    try:
        original = df['text'].iloc[i]                     # Get original text
        cleaned = cleaned_df['text'].iloc[i]              # Get cleaned text
        print(f"\nOriginal: {original}")                  # Show original version
        print(f"Cleaned : {cleaned}")                     # Show cleaned version
    except IndexError:
        break                                             # In case fewer than 5 entries

# --------------------------------------------
# 📊 STEP 10: Track Total Changes Made
# --------------------------------------------
changed_count = 0                                         # Counter for modified sentences
total = len(original_dataset)                            # Total number of samples in original dataset

# Loop through all samples and count how many texts were actually changed
for i in range(total):
    original = original_dataset[i]["text"]                # Text from original dataset
    cleaned = cleaned_dataset[i]["text"]                  # Text from cleaned dataset
    if original != cleaned:
        changed_count += 1                                # Increment if there's a change

# Show total and percentage of changed entries
print(f"\nChanged sentences: {changed_count} / {total}")                    
print(f"Percentage changed: {changed_count / total * 100:.2f}%")

# --------------------------------------------
# 💾 STEP 11: Save the Cleaned Dataset Locally
# --------------------------------------------
cleaned_dataset.save_to_disk("data/cleaned_dataset")     # Save the cleaned dataset in Hugging Face format locally

# Also export the cleaned data as CSV file (for use in tools like Excel or notebooks)
cleaned_df.to_csv("data/cleaned_dataset.csv", index=False)
print("\nCleaned dataset saved to 'data/cleaned_dataset' and 'data/cleaned_dataset.csv'")  # Confirmation message
