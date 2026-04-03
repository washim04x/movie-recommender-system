import json
import os
from pathlib import Path
import random
import shutil
import threading

import joblib
import mlflow
import requests
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.title("Movie Recommender System")

# load models and data
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Lazy model loading: don't block app startup
_df = None
_indices = None
_tfidf_matrix = None
_load_lock = threading.Lock()
_warmup_started = False
_load_error = None


def load_joblib(file_path):
    return joblib.load(file_path)


def load_model_info(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def configure_mlflow_tracking():
    dagshub_key = os.getenv("DAGSHUB_MOVIE")
    if not dagshub_key:
        raise ValueError("Dagshub_movie environment variable not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_key
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_key
    mlflow.set_tracking_uri(
        "https://dagshub.com/washim04x/movie-recommender-system.mlflow"
    )


def ensure_local_artifacts():
    targets = {
        "tfidf_matrix": MODELS_DIR / "tfidf_matrix.joblib",
        "indices": MODELS_DIR / "indices.joblib",
        "df": MODELS_DIR / "df.joblib",
    }

    missing = [name for name, path in targets.items() if not path.exists()]
    if not missing:
        return

    model_info_path = REPORTS_DIR / "model_info.json"
    model_info = load_model_info(model_info_path)
    artifact_uris = model_info.get("artifact_uris", {})

    configure_mlflow_tracking()

    for name in missing:
        artifact_uri = artifact_uris.get(name)
        if not artifact_uri:
            raise FileNotFoundError(
                f"Missing artifact URI for '{name}' in {model_info_path}"
            )

        downloaded_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri
        )
        target_path = targets[name]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded_path, target_path)


def _load_artifacts():
    global _df, _indices, _tfidf_matrix, _load_error
    try:
        ensure_local_artifacts()
        _df = load_joblib(MODELS_DIR / "df.joblib")
        _indices = load_joblib(MODELS_DIR / "indices.joblib")
        _tfidf_matrix = load_joblib(MODELS_DIR / "tfidf_matrix.joblib")
    except Exception as e:
        _load_error = str(e)
        print(f"[warmup] Artifact loading failed: {e}")


def _ensure_artifacts_loaded():
    global _df, _indices, _tfidf_matrix, _load_error
    if _df is not None and _indices is not None and _tfidf_matrix is not None:
        return
    if _load_error is not None:
        return

    with _load_lock:
        if _df is None or _indices is None or _tfidf_matrix is None:
            _load_artifacts()


def _warmup_async():
    global _warmup_started
    if _warmup_started:
        return
    _warmup_started = True

    def _bg():
        try:
            _ensure_artifacts_loaded()
            print("[warmup] Artifacts loaded successfully")
        except Exception as e:
            print(f"[warmup] Artifact warmup failed: {e}")

    threading.Thread(target=_bg, daemon=True).start()


# Start background warmup
_warmup_async()

top_movies = None


def get_top_movies():
    global top_movies
    if top_movies is None:
        _ensure_artifacts_loaded()
        if _df is None:
            return None
        top_movies = _df.sort_values("vote_average", ascending=False).head(50)
    return top_movies


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
    movies_df = get_top_movies()
    if movies_df is None:
        return None

    sample_size = min(5, len(movies_df))
    sampled = movies_df.sample(
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
def recommend_movies(movie_title, n=5):
    _ensure_artifacts_loaded()
    if _df is None or _tfidf_matrix is None or _indices is None:
        return None

    title = movie_title.strip().replace(' ', '').lower()
    if title not in _indices:
        return None

    idx = _indices[title]
    sim_scores = cosine_similarity(_tfidf_matrix[idx], _tfidf_matrix).flatten()
    sim_idx = sim_scores.argsort()[::-1][1:n + 1]
    recommendations = (
        _df[["original_title", "id"]].iloc[sim_idx].values.tolist()
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
    # Ensure artifacts are loaded before using them
    _ensure_artifacts_loaded()

    if _df is None or _load_error is not None:
        st.warning(
            "Models are still loading. Please wait a moment and try again..."
        )
        if _load_error:
            st.error(f"Error loading artifacts: {_load_error}")
        st.stop()

    selected_title = movie_title.strip()

    if not selected_title:
        print("No title entered. Showing random top-rated movies.")
        st.info("No title entered. Showing random top-rated movies.")
        fallback_movies = top_5_movies_data()
        render_movies(fallback_movies, show_relevance=True)
        st.stop()

    recommendations = recommend_movies(
        selected_title, n=10
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

