import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix

class SimilarArtists:
    def __init__(self, df, name_column, exclude_features = ["artists","id","name","loudness","release_date"]):
        self.df = df
        # Name of the column that contains the artist names in our case "artists"
        self.name_column = name_column
        self.exclude_features = exclude_features
        self.artist = df[name_column].tolist()
        
        # self.target_column = target_column
        
        
    def get_random_artist_observation_index_by_name(self, name):
        indexes = []
        
        # find all indexes of artists that contain the name
        for x in range(len(self.artist)):
            if name.lower() in str(self.artist[x]).lower():
                indexes.append(x)
        
        # if no matches found, return None
        if len(indexes) == 0:
            return None
        else:
            # return a random index from the found indexes 
            return  indexes[random.randint(0, len(indexes))]
        
        
    def get_cosine_similarity_matrix(self, name, n = 5):
        
        target_index = self.get_random_artist_observation_index_by_name(name)
        
        if target_index is None:
            return None
        
        if self.exclude_features is not None:
          data = self.df.drop(columns=self.exclude_features).values
        else:
            data = self.df.values
        
        normalized_data = normalize(data)
        sparse_data = csr_matrix(normalized_data)
        similarity_scores = cosine_similarity(sparse_data, dense_output=False)
        target_similarity = similarity_scores[target_index].toarray().flatten()
        target_similarity[target_index] = -1
        top_n_indices = np.argsort(target_similarity)[-n:][::-1]

        return top_n_indices
    
    


# def get_cosine_similarity_matrix(df, target_index, exclude_features = None, n = 5):
    
#     if exclude_features is not None:
#        data = df.drop(columns=exclude_features).values
#     else:
#         data = df.values
    
#     similarity_scores = cosine_similarity(data)
    
#     target_similarity = similarity_scores[target_index]
    
#     target_similarity[target_index] = -1
    
#     top_n_indices = np.argsort(target_similarity)[-n:][::-1]
    
#     return top_n_indices



# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# def get_cosine_similarity_matrix(df, target_index, exclude_features = None, n = 5):
    
#     if exclude_features is not None:
#        data = df.drop(columns=exclude_features).values
#     else:
#         data = df.values
    
#     similarity_scores = cosine_similarity(data)
    
#     target_similarity = similarity_scores[target_index]
    
#     target_similarity[target_index] = -1
    
#     top_n_indices = np.argsort(target_similarity)[-n:][::-1]
    
#     return top_n_indices
    
#for genres
# get_cosine_similarity_matrix(pd, 0, 10, ["genres", "duration_ms"])

# for artist
# get_cosine_similarity_matrix(pd, 2, ["artists","id","name","loudness","release_date"])

# def get_random_artist_observation_index_by_name(name, artist):
#     indexes = []
    
#     # find all indexes of artists that contain the name
#     for x in range(len(artist)):
#         if name.lower() in str(artist[x]).lower():
#             indexes.append(x)
    
#     # if no matches found, return None
#     if len(indexes) == 0:
#         return None
#     else:
#         # return a random index from the found indexes
#         return  indexes[random.randint(0, len(indexes))]
    
    

# get_random_artist_observation_index_by_name('Mehmet')

# def get_cosine_similarity_matrix(df, target_index, exclude_features = None, n = 5):
    
#     if exclude_features is not None:
    #    data = df.drop(columns=exclude_features).values
#     else:
#         data = df.values
    
#     similarity_scores = cosine_similarity(data)
    
#     target_similarity = similarity_scores[target_index]
    
#     target_similarity[target_index] = -1
    
#     top_n_indices = np.argsort(target_similarity)[-n:][::-1]
    
#     return top_n_indices
    