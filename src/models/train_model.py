import pandas as pd
from pathlib import Path
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib





def recommend_movies(movie_title, df, tfidf_matrix, indices, n=10):
    title = movie_title.strip().replace(' ', '').lower()
    if title not in indices:
        return f"Movie '{movie_title}' not found in the dataset."
    
    idx = indices[title]
    
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_idx = sim_scores.argsort()[::-1][1:n+1]
    return df['original_title'].iloc[sim_idx]

def save_model(model_dir, df ,tfidf_matrix, indices):
    with open(model_dir / 'tfidf_matrix.joblib', 'wb') as f:
        joblib.dump(tfidf_matrix, f)
    with open(model_dir / 'indices.joblib', 'wb') as f:
        joblib.dump(indices, f)
    with open(model_dir / 'df.joblib', 'wb') as f:
        joblib.dump(df, f)

def main():
    curr_dir =Path(__file__)
    parent_dir = curr_dir.parent.parent.parent

    params_path = parent_dir / 'params.yaml'
    params= yaml.safe_load(params_path.read_text())

    data_path = parent_dir / params['data']['processed_data_path'] / params['data']['features_data_file']
    df = pd.read_csv(data_path)

    df['name']= df['original_title'].str.strip().str.replace(' ', '').str.lower()
    indices = pd.Series(df.index, index=df['name'])
    indices = indices[~indices.index.duplicated(keep='first')]

    tfidf = TfidfVectorizer(max_features=params['parameters']['model']['n_features'],ngram_range=tuple(params['parameters']['model']['n_grams']))
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    

    movie_title = "The Dark Knight"
    recommendations = recommend_movies(movie_title, df, tfidf_matrix, indices)
    print(f"Recommendations for '{movie_title}':")
    print(recommendations)
    print("Model training completed successfully.")

    model_dir = parent_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    save_model(model_dir,df,tfidf_matrix, indices)

    
if __name__ == "__main__":
    main()
    
