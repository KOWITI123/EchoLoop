# main.py
import os
import numpy as np
from src.feature_extraction import extract_features_batch
from src.embedding_generator import load_embedding
from src.database import store_in_mongodb, upload_to_gridfs
from src.search import test_vector_search
from sklearn.decomposition import PCA
import pickle

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MJ_MOTIF_DIR = os.path.join(BASE_DIR, "processed", "micheal_jackson", "motifs")
BEATLES_MOTIF_DIR = os.path.join(BASE_DIR, "processed", "the_beatles", "melodies")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

MONGO_URI = "mongodb+srv://allankowiti55:ZITWjvIHsJ3Wcbnr@cluster0.9yu6rwa.mongodb.net/Data?retryWrites=true&w=majority"
DB_NAME = "Data"
MOTIFS_COLLECTION = "Motifs"
EMBEDDING_DIM = 128
PCA_MODEL_PATH = os.path.join(BASE_DIR, "pca_model.pkl")

def main():
    all_embeddings = []
    motif_metadata = []

    # Process Michael Jackson (audio)
    mj_files = [f for f in os.listdir(MJ_MOTIF_DIR) if f.islower() and f.endswith(".wav")]
    print(f"Found {len(mj_files)} Michael Jackson motifs")
    mj_paths = [os.path.join(MJ_MOTIF_DIR, fname) for fname in mj_files]
    mj_embeddings = extract_features_batch(mj_paths)
    for fname, embedding in zip(mj_files, mj_embeddings):
        motif_id = fname.replace(".wav", "")
        all_embeddings.append(embedding)
        motif_metadata.append({
            "artist": "Michael Jackson",
            "motif_id": motif_id,
            "metadata": {"tempo": 100, "key": "A minor"},
            "file_path": os.path.join(MJ_MOTIF_DIR, fname)
        })

    # Process The Beatles (audio)
    beatles_files = [f for f in os.listdir(BEATLES_MOTIF_DIR) if f.islower() and f.endswith(".wav")]
    print(f"Found {len(beatles_files)} Beatles motifs")
    beatles_paths = [os.path.join(BEATLES_MOTIF_DIR, fname) for fname in beatles_files]
    beatles_embeddings = extract_features_batch(beatles_paths)
    for fname, embedding in zip(beatles_files, beatles_embeddings):
        motif_id = fname.replace(".wav", "")
        all_embeddings.append(embedding)
        motif_metadata.append({
            "artist": "The Beatles",
            "motif_id": motif_id,
            "metadata": {"tempo": 120, "key": "E major"},
            "file_path": os.path.join(BEATLES_MOTIF_DIR, fname)
        })

    # Convert embeddings to numpy array
    all_embeddings = np.array(all_embeddings)  # Shape: (89, 768)
    print(f"Generated embeddings with shape: {all_embeddings.shape}")

    # Apply PCA to reduce to min(n_samples, n_features)
    n_samples = all_embeddings.shape[0]  # 89
    pca_dim = min(n_samples, all_embeddings.shape[1])  # 89
    pca = PCA(n_components=pca_dim)
    embeddings = pca.fit_transform(all_embeddings)  # Shape: (89, 89)

    # Save the PCA model for future use
    with open(PCA_MODEL_PATH, 'wb') as f:
        pickle.dump(pca, f)
    print(f"Saved PCA model to {PCA_MODEL_PATH}")

    # Pad embeddings to EMBEDDING_DIM (128)
    embeddings_padded = np.zeros((embeddings.shape[0], EMBEDDING_DIM))
    embeddings_padded[:, :pca_dim] = embeddings

    # Normalize embeddings
    embeddings = embeddings_padded / (np.linalg.norm(embeddings_padded, axis=1, keepdims=True) + 1e-8)

    # Prepare embeddings data for MongoDB
    embeddings_data = []
    for i, (embedding, meta) in enumerate(zip(embeddings, motif_metadata)):
        artist = meta["artist"]
        motif_id = meta["motif_id"]
        embedding_path = os.path.join(EMBEDDINGS_DIR, f"{artist}_{motif_id}.npy")
        np.save(embedding_path, embedding)
        print(f"Saved embedding locally: {embedding_path}")
        embeddings_data.append({
            "artist": artist,
            "motif_id": motif_id,
            "embedding": embedding.tolist(),
            "metadata": meta["metadata"]
        })

    # Store embeddings in MongoDB Atlas
    db, client = store_in_mongodb(embeddings_data, MONGO_URI, DB_NAME, MOTIFS_COLLECTION)

    # Verify insertion
    collection = db[MOTIFS_COLLECTION]
    print(f"Inserted {collection.count_documents({})} documents into MongoDB Atlas")

    # Upload stems to GridFS
    for meta in motif_metadata:
        artist = meta["artist"]
        stem_id = meta["motif_id"]
        file_path = meta["file_path"]
        upload_to_gridfs(db, file_path, stem_id, artist)

    # Test vector search
    if mj_files:
        query_embedding = load_embedding("Michael Jackson", "mj_motif_0", EMBEDDINGS_DIR)
        test_vector_search(db, query_embedding, "Michael Jackson", MOTIFS_COLLECTION)

    client.close()

if __name__ == "__main__":
    main()