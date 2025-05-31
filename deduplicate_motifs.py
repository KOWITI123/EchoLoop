# deduplicate_motifs.py
from pymongo import MongoClient

# Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://allankowiti55:ZITWjvIHsJ3Wcbnr@cluster0.9yu6rwa.mongodb.net/Data")
db = client["Data"]
collection = db["Motifs"]

# Run the pipeline to get duplicate _ids
pipeline = [
    {"$group": {"_id": "$motif_id", "uniqueIds": {"$addToSet": "$_id"}, "firstDoc": {"$first": "$$ROOT"}}},
    {"$unwind": "$uniqueIds"},
    {"$match": {"$expr": {"$ne": ["$uniqueIds", "$firstDoc._id"]}}},
    {"$project": {"_id": "$uniqueIds"}}
]
duplicates = list(collection.aggregate(pipeline))
duplicate_ids = [doc["_id"] for doc in duplicates]

# Delete duplicates
if duplicate_ids:
    collection.delete_many({"_id": {"$in": duplicate_ids}})
    print(f"Deleted {len(duplicate_ids)} duplicate documents.")
else:
    print("No duplicates found.")

client.close()