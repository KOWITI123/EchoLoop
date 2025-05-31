# src/embedding_generator.py
import numpy as np
import os

def load_embedding(artist, motif_id, embeddings_dir):
    embedding_path = os.path.join(embeddings_dir, f"{artist}_{motif_id}.npy")
    if os.path.exists(embedding_path):
        print(f"Loading existing embedding: {embedding_path}")
        return np.load(embedding_path)
    raise FileNotFoundError(f"Embedding not found: {embedding_path}")