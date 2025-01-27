import random

from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import seaborn as sns


class Recommendation_Module:

    def __init__(self, df, name_column):
        self.df = df
        # Name of the column that contains the artist names in our case "artists"
        self.name_column = name_column
        self.artist = df[name_column].tolist()
        
        self.df = self.extract_features()
        
        
    def extract_features(self):
        if "genres" not in  self.df.columns:
            return self.df.drop(columns = ["artists", "id", "name", "release_date"])
        else:
            return self.df.drop(columns = ["id"])

    def split_features(self):
        x_train, y_train = train_test_split(self.df, test_size=0.2, random_state=42)
        return x_train, y_train
    
    def create_cluster(self):
        # Usinjg K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=100)
        cluster = kmeans.fit_predict(self.df)
        self.df['Cluster'] = cluster
        # self.df['Cluster'] = kmeans.fit_predict(self.df)
        
    def visulize_cluster(self, based_on = ['popularity', 'tempo']):
        sns.scatterplot(x='feature1', y='feature2', hue='Cluster', data=self.df, palette='viridis')
        plt.title('K-Means Clustering')
        plt.show()

        plt.scatter(self.df['Cluster'], self.df['acousticness'])
        plt.show()
        
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

    def get_top_n_similar_variables(self, name, n=5):

        target_index = self.get_random_artist_observation_index_by_name(name)

        if target_index is None:
            return None

        normalized_data = self.normalize_data()

        sparse_data = csr_matrix(normalized_data)
        similarity_scores = cosine_similarity(sparse_data, dense_output=False)
        target_similarity = similarity_scores[target_index].toarray().flatten()
        target_similarity[target_index] = -1
        top_n_indices = np.argsort(target_similarity)[-n:][::-1]

        return top_n_indices

    def generate_recommendation_data(self, name, n=5):
        scaler = StandardScaler()
        
        normalized_data = self.normalize_data()
        scaled_features = scaler.fit_transform(normalized_data)

        # Apply K-Means
        kmeans = KMeans(n_clusters=3, random_state=100)
        self.df['Cluster'] = kmeans.fit_predict(scaled_features)

        # Recommend items from the same cluster
        target_cluster = 0
        recommendations = self.df[self.df['Cluster'] == self.df[self.df[self.name_column] == name]['Cluster']].head(n)
        
        recommendations = self.df[self.df['Cluster'] == target_cluster]
        print("Items in Cluster 0 for Recommendation:")
        print(recommendations)
 
        return recommendations

        
    def optimize_k_means(self, max_clusters=10):      
        means_values = []
        inertia_values = []
        
        for k in range(1, max_clusters):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.normalize_data())
            
            means_values.append(k)
            inertia_values.append(kmeans.inertia_)
            
        fig =plt.subplots(figsize=(10, 6))
        plt.plot(means_values, inertia_values, 'o-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.grid(True)
        plt.show()
        
        
        
        
        
#https://youtu.be/iNlZ3IU5Ffw?si=TK5_aHZcTlo7C7i4
