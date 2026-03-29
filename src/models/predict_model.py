import random
import pandas as pd
from pathlib import Path
import yaml
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies(movie_title, df, tfidf_matrix, indices, n=10):
    title = movie_title.strip().replace(' ', '').lower()
    if title not in indices:
        return f"Movie '{movie_title}' not found in the dataset."
    
    idx = indices[title]
    
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_idx = sim_scores.argsort()[::-1][1:n+1]
    return df['original_title'].iloc[sim_idx]

def main():
    curr_dir =Path(__file__)
    parent_dir = curr_dir.parent.parent.parent
    params_path = parent_dir / 'params.yaml'
    params = yaml.safe_load(open(params_path))

    models_path = parent_dir / 'models'
    tfidf_matrix_path = models_path / 'tfidf_matrix.pkl'
    indices_path = models_path / 'indices.pkl'
    df_path = models_path / 'df.pkl'

    tfidf_matrix = pickle.load(open(tfidf_matrix_path, 'rb'))
    indices = pickle.load(open(indices_path, 'rb'))
    df = pickle.load(open(df_path, 'rb'))
    
    movie_title = params['predict']['movie_title']

    recommendations = recommend_movies(movie_title, df, tfidf_matrix, indices, n=params['predict']['n_recommendations'])
    print(f"Recommendations for '{movie_title}':")
    print(recommendations)

if __name__ == "__main__":
    main()


