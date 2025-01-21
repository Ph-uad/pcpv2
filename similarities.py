
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_cosine_similarity_matrix(data, target_index, n = 5):
    
    similarity_scores = cosine_similarity(data)
    
    target_similarity = similarity_scores[target_index]
    
    target_similarity[target_index] = -1
    
    top_n_indices = np.argsort(target_similarity)[-n:][::-1]
    
    top_n_similarity = [(index , target_index) for index, target_index in top_n_indices]
    
    return top_n_similarity
    
    
    