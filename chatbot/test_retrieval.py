from pymongo import MongoClient

class TextRetrieval:
    def __init__(self, URI, db_name, collection_name, initial_vector_ids, initial_texts):
        self.client = MongoClient(URI)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

        # Drop the collection to start fresh
        self.collection.drop()

        self.add_texts(initial_vector_ids, initial_texts)

    def add_texts(self, vector_ids, texts):
        # insert all the texts into the collection
        pairs = [{"vector_id": int(vector_id), "text": text} for vector_id, text in zip(vector_ids, texts)]
        self.collection.insert_many(pairs)

    def get_text(self, vector_id):
        pair = self.collection.find_one({"vector_id": int(vector_id)})
        if pair is None:
            raise RuntimeError(f"Text not found for vector_id: {vector_id}")
        return pair["text"]
