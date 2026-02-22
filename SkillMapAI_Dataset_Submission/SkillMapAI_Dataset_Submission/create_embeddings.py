import pandas as pd
from pathlib import Path
import ast
from sentence_transformers import SentenceTransformer
import numpy as np

# ---------------- Load Dataset ----------------

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "data" / "processed_jobs.csv"

df = pd.read_csv(data_path)

# Convert skills back to list
df["job_skill_set"] = df["job_skill_set"].apply(ast.literal_eval)

print("Dataset loaded:", df.shape)

# ---------------- Prepare Text for AI ----------------

def combine_text(row):
    skills = " ".join(row["job_skill_set"])
    return f"{row['job_title']} {skills} {row['clean_description']}"

df["combined_text"] = df.apply(combine_text, axis=1)

print("Text prepared for embedding.")

# ---------------- Load AI Model ----------------

print("Loading AI model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- Create Embeddings ----------------

print("Creating embeddings...")
embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)

print("Embeddings shape:", embeddings.shape)

# ---------------- Save Embeddings ----------------

embedding_path = BASE_DIR / "data" / "job_embeddings.npy"
np.save(embedding_path, embeddings)

print("\nSUCCESS: Embeddings saved at:")
print(embedding_path)
