import os
from pathlib import Path
import pickle
import random
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import requests

st.title("Movie Recommender System")

# load models and data
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


df = load_pickle(MODELS_DIR / "df.pkl")
indices = load_pickle(MODELS_DIR / "indices.pkl")
tfidf_matrix = load_pickle(MODELS_DIR / "tfidf_matrix.pkl")

top_movies = df.sort_values("vote_average", ascending=False).head(50)


def enrich_movies_with_details(movies, include_relevance=False):
    enriched_movies = []
    tmdb_available = headers is not None

    for movie in movies:
        movie_title = movie["title"]
        movie_id = movie["id"]
        relevance = movie.get("relevance")

        details = None
        genres = []
        poster_url = None

        if tmdb_available:
            details, should_disable_tmdb = fetch_movie_details(movie_id)
            if should_disable_tmdb:
                tmdb_available = False

        if details:
            genres = [genre["name"] for genre in details.get("genres", [])]
            poster_path = details.get("poster_path")
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"

        payload = {
            "title": movie_title,
            "genres": genres,
            "poster_url": poster_url,
        }

        if include_relevance and relevance is not None:
            payload["relevance"] = relevance

        enriched_movies.append(payload)

    return enriched_movies


def top_5_movies_data():
    sample_size = min(5, len(top_movies))
    sampled = top_movies.sample(
        n=sample_size,
        random_state=random.randint(1, 99999),
    )
    fallback_movies = [
        {
            "title": row["original_title"],
            "id": row["id"],
            "relevance": float(row["vote_average"]),
        }
        for _, row in sampled.iterrows()
    ]
    return enrich_movies_with_details(fallback_movies, include_relevance=True)


# TMDB API setup
TMDB_API_KEY = os.getenv('TMDB_API')

if TMDB_API_KEY:
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_KEY}",
    }
else:
    headers = None
    st.warning(
        "TMDB_API environment variable is not set. "
        "Posters and genres will be unavailable."
    )


def fetch_movie_details(movie_id: int):
    if headers is None:
        return None, False

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.json(), False
    except requests.exceptions.Timeout:
        st.warning("TMDB timeout. Showing titles without extra TMDB data.")
        return None, True
    except requests.exceptions.ConnectionError:
        st.warning(
            "TMDB connection failed. "
            "Showing titles without extra TMDB data."
        )
        return None, True
    except requests.exceptions.RequestException as e:
        st.error(f"TMDB request failed: {e}")
    return None, False


# movie recommendation function
def recommend_movies(movie_title, data, matrix, title_to_index, n=5):
    title = movie_title.strip().replace(' ', '').lower()
    if title not in title_to_index:
        return None

    idx = title_to_index[title]
    sim_scores = cosine_similarity(matrix[idx], matrix).flatten()
    sim_idx = sim_scores.argsort()[::-1][1:n + 1]
    recommendations = (
        data[["original_title", "id"]].iloc[sim_idx].values.tolist()
    )
    recommendation_movies = [
        {"title": rec_title, "id": movie_id}
        for rec_title, movie_id in recommendations
    ]
    return enrich_movies_with_details(recommendation_movies)


def render_movies(movies, show_relevance=False):
    st.write("Movies:")
    for i, movie in enumerate(movies, start=1):
        st.write(f"{i}. {movie['title']}")
        if show_relevance and "relevance" in movie:
            st.caption(f"Relevance: {movie['relevance']:.1f}")
        if movie["genres"]:
            st.caption(", ".join(movie["genres"]))
        if movie["poster_url"]:
            st.image(movie["poster_url"], width=160)


movie_title = st.text_input("Enter movie title")

# Show random top-5 on every load/refresh by default.
default_movies = top_5_movies_data()


if st.button("Recommend"):
    selected_title = movie_title.strip()

    if not selected_title:
        print("No title entered. Showing random top-rated movies.")
        st.info("No title entered. Showing random top-rated movies.")
        fallback_movies = top_5_movies_data()
        render_movies(fallback_movies, show_relevance=True)
        st.stop()

    recommendations = recommend_movies(
        selected_title, df, tfidf_matrix, indices, n=10
    )
    if recommendations is None:
        st.info(
            f"Movie '{selected_title}' not found in dataset. "
            "Showing random top-rated movies."
        )
        fallback_movies = top_5_movies_data()
        render_movies(fallback_movies, show_relevance=True)
    else:
        print(f"Recommendations for '{selected_title}':")
        st.write("Recommended Movies:")
        render_movies(recommendations, show_relevance=False)




