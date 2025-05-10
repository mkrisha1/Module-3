import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import ast

# Load and clean dataset
df = pd.read_csv("imdb_movies.csv", encoding="latin1")
df.columns = df.columns.str.lower()

# Clean genre and crew
df["genre"] = df["genre"].str.replace("Â", "", regex=False)
df["genre"] = df["genre"].apply(lambda x: x.split(",") if isinstance(x, str) else [])

df["crew"] = df["crew"].astype(str)

# Simplify actor parsing by just splitting by comma
df["actors"] = df["crew"].apply(lambda x: [actor.strip() for actor in x.split(",")] if isinstance(x, str) else [])

# Drop rows with missing values needed for similarity
df = df.dropna(subset=["genre", "actors", "score"])

# Find Cocaine Bear movie
target_movie = df[df['names'].str.lower() == "cocaine bear"]
if target_movie.empty:
    raise ValueError("Cocaine Bear not found in dataset.")
target_movie = target_movie.iloc[0]

# ---------- GENRE SIMILARITY ----------
mlb_genre = MultiLabelBinarizer()
genre_matrix = mlb_genre.fit_transform(df["genre"])
genre_df = pd.DataFrame(genre_matrix, index=df.index, columns=mlb_genre.classes_)

genre_sim = cosine_similarity([genre_df.loc[target_movie.name]], genre_df)[0]
df["genre_similarity"] = genre_sim

top_genre = df[df["names"] != "Cocaine Bear"].sort_values("genre_similarity", ascending=False).head(10)

# ---------- ACTOR SIMILARITY ----------
mlb_actor = MultiLabelBinarizer()
actor_matrix = mlb_actor.fit_transform(df["actors"])
actor_df = pd.DataFrame(actor_matrix, index=df.index, columns=mlb_actor.classes_)

actor_sim = cosine_similarity([actor_df.loc[target_movie.name]], actor_df)[0]
df["actor_similarity"] = actor_sim

top_actors = df[df["names"] != "Cocaine Bear"].sort_values("actor_similarity", ascending=False).head(10)

# ---------- USER RATING SIMILARITY ----------
df["rating_similarity"] = -abs(df["score"] - target_movie["score"])
top_rating = df[df["names"] != "Cocaine Bear"].sort_values("rating_similarity", ascending=False).head(10)

# ---------- DISPLAY RESULTS ----------
print("\nTop 10 Most Similar Movies to Cocaine Bear Based on Genre:")
print(top_genre[["names", "genre_similarity"]])

print("\nTop 10 Most Similar Movies to Cocaine Bear Based on Actors:")
print(top_actors[["names", "actor_similarity"]])

print("\nTop 10 Most Similar Movies to Cocaine Bear Based on User Rating:")
print(top_rating[["names", "score", "rating_similarity"]])
