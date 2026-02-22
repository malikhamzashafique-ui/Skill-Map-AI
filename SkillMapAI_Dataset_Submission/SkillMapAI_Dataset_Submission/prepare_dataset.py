import pandas as pd
import ast
from pathlib import Path

# Load dataset
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "data" / "all_job_post.csv"

df = pd.read_csv(data_path)

print("Original Shape:", df.shape)

# Convert string list to actual Python list
df["job_skill_set"] = df["job_skill_set"].apply(ast.literal_eval)

print("\nConverted skill column to list.")
print(type(df["job_skill_set"][0]))

# Normalize skills
def clean_skills(skill_list):
    return [skill.lower().strip() for skill in skill_list]

df["job_skill_set"] = df["job_skill_set"].apply(clean_skills)

print("\nSample cleaned skills:")
print(df["job_skill_set"].head())

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)            # remove line breaks
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)   # remove symbols & numbers
    text = re.sub(r'\s+', ' ', text)           # remove extra spaces
    return text.strip()

df["clean_description"] = df["job_description"].apply(clean_text)

print("\nCleaned description sample:\n")
print(df["clean_description"].head(2))
# Keep only useful columns
df = df[["category", "job_title", "clean_description", "job_skill_set"]]

print("\nFinal dataset columns:")
print(df.columns)

print("Final shape:", df.shape)

# Safety check — remove jobs that somehow have no skills
df = df[df["job_skill_set"].map(len) > 0]

print("\nAfter removing empty-skill jobs:")
print(df.shape)

# ---------------- SAVE FINAL DATASET ----------------

output_path = BASE_DIR / "data" / "processed_jobs.csv"
df.to_csv(output_path, index=False)

print("\nSUCCESS: Processed dataset saved at:")
print(output_path)
