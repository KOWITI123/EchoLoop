# src/search.py
def test_vector_search(db, query_embedding, artist, collection_name):
    collection = db[collection_name]
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding.tolist(),
                "numCandidates": 100,  # Increased from 10
                "limit": 5,
                "filter": {"artist": artist}
            }
        },
        {
            "$project": {
                "artist": 1,
                "motif_id": 1,
                "score": {"$meta": "vectorSearchScore"},
                "_id": 0
            }
        }
    ]
    print(f"Executing vector search for artist: {artist}")
    results = list(collection.aggregate(pipeline))
    print("Vector Search Results:")
    if not results:
        print("No results found.")
    for result in results:
        print(result)