from .vector_search import VectorSearch
from .text_retrieval import TextRetrieval

class KnowledgeFlow:
    def __init__(self, index_type, dimension, vectors, texts, URI, db_name, collection_name):
        self.vector_search = VectorSearch(index_type, dimension)
        initial_vector_ids = self.vector_search.create_index(vectors)
        self.text_retrieval = TextRetrieval(URI, db_name, collection_name, initial_vector_ids, texts)

    def search(self, query_vector, k):
        distances, indices = self.vector_search.search(query_vector, k)
        
        texts = [self.text_retrieval.get_text(index) for index in indices]
        return texts
    
    # add texts
    def add(self, vectors, texts):
        vector_ids = self.vector_search.add_vectors(vectors)
        self.text_retrieval.add_texts(vector_ids, texts)
