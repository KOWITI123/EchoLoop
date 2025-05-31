# src/database.py
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import gridfs

def store_in_mongodb(embeddings_data, mongo_uri, db_name, collection_name):
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        # Get existing motif_ids
        existing_ids = {doc["motif_id"] for doc in collection.find({}, {"motif_id": 1})}
        # Filter out documents that already exist
        new_docs = [doc for doc in embeddings_data if doc["motif_id"] not in existing_ids]
        if new_docs:
            collection.insert_many(new_docs)
            print(f"Inserted {len(new_docs)} new embeddings into MongoDB Atlas")
        else:
            print("No new embeddings to insert; all motif_ids already exist.")
        return db, client
    except ConnectionFailure as e:
        print(f"Error connecting to MongoDB Atlas: {e}")
        raise

def upload_to_gridfs(db, file_path, stem_id, artist):
    fs = gridfs.GridFS(db)
    # Check if the file already exists in GridFS
    existing_file = fs.find_one({"metadata.stem_id": stem_id, "metadata.artist": artist})
    if existing_file:
        print(f"File already exists in GridFS: {stem_id}, File ID: {existing_file._id}")
        return existing_file._id
    # Upload the file if it doesn't exist
    with open(file_path, 'rb') as f:
        file_id = fs.put(f, filename=stem_id, metadata={"stem_id": stem_id, "artist": artist})
    print(f"Uploaded to GridFS: {stem_id}, File ID: {file_id}")
    return file_id