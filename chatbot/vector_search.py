import faiss
import numpy as np

class VectorSearch:
    def __init__(self, index_type, dimension):
        self.index_type = index_type
        self.dimension = dimension
        self.index = None

    def create_index(self, vectors):
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf":
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(self.dimension), self.dimension, 1)
            self.index.train(vectors) # only for IVF not Flat
        self.index.add(vectors)
        return list(range(self.index.ntotal))

    
    def search(self, query_vector, k):
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        return distances[0], indices[0]
    
    def add_vectors(self, vectors):
        # If vectors is a 1D array, reshape it to a 2D array
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        try:
            first_vector_id = self.index.ntotal
            self.index.add(vectors)
            last_vector_id = self.index.ntotal
            return list(range(first_vector_id, last_vector_id))
        except Exception as e:
            raise RuntimeError(f"Error adding vectors: {e}")
        

        
    
