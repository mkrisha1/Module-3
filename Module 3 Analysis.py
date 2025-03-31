# Module 3 Analysis

import pandas as pd
import scipy.spatial.distance
import ast

movie_genre = {}
# Load CSV file and remove rows with missing values
df = pd.read_csv("imdb_movies.csv", encoding="latin1")
for _, row in df.iterrows():
    genres = str(row['genres']).split(", ")  # Split genres assuming they are comma-separated
    actors = ast.literal_eval(row['actors']) if isinstance(row['actors'], str) else []  # Convert string to list safely

    for actor_id, actor_name in actors:
        if actor_id not in movie_genre:
            movie_genre[actor_id] = {}
        for genre in genres:
            if genre not in movie_genre[actor_id]:
                movie_genre[actor_id][genre] = 0
            movie_genre[actor_id][genre] += 1


index = movie_genre.keys()
rows = [movie_genre[k] for k in index]
df_genre = pd.DataFrame(rows, index=index)
df = df.fillna
df
df_norm = df.divide(df.sum(axis = 1), axis = 0)
df_norm.head(10)

# Normalize the data
df_norm = df_genre.divide(df_genre.sum(axis=1), axis=0)

# Compute cosine similarity for "Cocaine Bear"
target_movie = "Cocaine Bear"
if target_movie in df_norm.index:
    target_vector = df_norm.loc[target_movie].values.reshape(1, -1)
    distances = scipy.spatial.distance.cdist(df_norm, target_vector, metric="cosine").flatten()
    query_distances = list(zip(df_norm.index, distances))

    # Print the top 10 most similar movies
    for similar_movie in sorted(query_distances, key=lambda x: x[1], reverse=False)[1:11]:
        print(similar_movie)
else:
    print(f"Movie '{target_movie}' not found in dataset.")
