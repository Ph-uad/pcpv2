import random

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import seaborn as sns


class Recommendation_Module:

    def __init__(self, df, name_column = "artists"):
        self.df = df
        self.data = df
        # Name of the column that contains the artist names in our case "artists"
        self.artist = self.data[name_column].tolist()

        self.df = self.extract_features()

    def extract_features(self):
        if "genres" not in self.df.columns:
            return self.df.drop(columns=["artists", "id", "name", "release_date"])
        else:
            return self.df.drop(columns=["id"])

    def split_features(self):
        x_train, y_train = train_test_split(
            self.df, test_size=0.2, random_state=42)
        return x_train, y_train

    def create_cluster(self):
        # Usinjg K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=100)
        cluster = kmeans.fit_predict(self.df)
        self.df['cluster'] = cluster
        self.data['cluster'] = cluster
        # self.df['cluster'] = kmeans.fit_predict(self.df)

    def visulize_cluster(self, based_on=['popularity', 'tempo']):
        sns.scatterplot(x=based_on[0], y=based_on[1],
                        hue='cluster', data=self.df, palette='viridis')
        plt.title('K-Means clustering')
        plt.show()

        plt.scatter(self.df['cluster'], self.df['popularity'])
        plt.show()

    def get_recommendation_by_cluster(self, name, n=5):
        target_index = self.get_random_artist_observation_index_by_name(name)

        if target_index is None:
            return None

        # get the cluster value of the target artist
        target_cluster_data = self.data.iloc[target_index]
        recommendations = self.data["cluster"] == target_cluster_data["cluster"]
        recommended = self.data[recommendations]

        target_popularity = target_cluster_data["popularity"]

        # Filter for songs released within +10 or -10 popularity of the target artist
        filtered = recommended[(recommended['popularity'] >= target_popularity - 10) & (recommended['popularity'] <= target_popularity + 10)]

        # If fewer than n observations, include additional ones
        if len(filtered) < n:
            additional = recommended[~recommended.index.isin(filtered.index)]  # Exclude already selected
            remaining_needed = n - len(filtered)
            filtered = pd.concat([filtered, additional.head(remaining_needed)])
        
        return filtered.head(n)

        
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
            return indexes[random.randint(0, len(indexes))]

    def normalize_data(self):
        return normalize(self.df)


    def get_top_n_similar_variables_by_similarity(self, name, n=5, chunk_size=None):
        print("Getting top n similar variables by cosine similarity")

        # Get the index of the target variable
        target_index = self.get_random_artist_observation_index_by_name(name)
        if target_index is None:
            return None

        # Normalize the dataset
        normalized_data = self.normalize_data()
        sparse_data = csr_matrix(normalized_data)

        # Split the dataset into chunks if chunk_size is provided
        if chunk_size and sparse_data.shape[0] > chunk_size:
            start_idx = (target_index // chunk_size) * chunk_size
            end_idx = min(start_idx + chunk_size, sparse_data.shape[0])
            chunk_data = sparse_data[start_idx:end_idx]
            target_row = sparse_data[target_index]

            # Compute similarity only for the chunk
            similarity_scores = cosine_similarity(chunk_data, target_row, dense_output=False).toarray().flatten()
        else:
            # Compute similarity for the entire dataset
            similarity_scores = cosine_similarity(sparse_data, dense_output=False)[target_index].toarray().flatten()

        # Exclude the target variable itself if it is within chunk_size
        if target_index <= len(similarity_scores):
            similarity_scores[target_index] = -1

        # Find top-n similar indices
        top_n_indices = np.argpartition(similarity_scores, -n)[-n:]
        top_n_indices = top_n_indices[np.argsort(similarity_scores[top_n_indices])[::-1]]
        
        #get the observsations by using the indices
        return self.data.iloc[top_n_indices]




#https://youtu.be/7QYxUv3j1gI?si=Q1Z9J9Q9Q9q
#https://youtu.be/iNlZ3IU5Ffw?si=TK5_aHZcTlo7C7i4
