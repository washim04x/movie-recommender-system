import pandas as pd
from pathlib import Path
import yaml
import ast
import nltk 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


import string
import re

def preprocess_text(text):
    text = str(text.lower())
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and w not in string.punctuation]
    tokens = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(tokens)



def extract_names(value):
        if pd.isna(value):
            return ""
        if isinstance(value, list):
            return " ".join([item.get('name', '') for item in value if isinstance(item, dict)])
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return ""
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return " ".join([item.get('name', '') for item in parsed if isinstance(item, dict)])
                return value
            except (ValueError, SyntaxError):
                return value
        return str(value)

def build_features(df):
    # Extract names from 'genres' and 'production_companies' columns
    df['genres'] = df['genres'].apply(extract_names)
    df['production_companies'] = df['production_companies'].apply(extract_names)
    df['combined_features'] = df['overview'] + ' ' + df['tagline'] + ' ' + df['genres'] + ' ' + df['production_companies']
    df=df[['original_title', 'combined_features', 'vote_average']]
    df['combined_features'] = df['combined_features'].apply(preprocess_text)
    df['combined_features'] = df['combined_features'].str.strip()
    df = df[df['combined_features'] != '']
    df = df[df['combined_features'].notna()]
    df = df.reset_index(drop=True)
    
    return df

def main():
    curr_dir =Path(__file__)
    parent_dir = curr_dir.parent.parent.parent
    params_path = parent_dir / 'params.yaml'
    params = yaml.safe_load(open(params_path))

    data_path = parent_dir / params['data']['processed_data_path'] / params['data']['processed_data_file']
    df = pd.read_csv(data_path)
    df = build_features(df)
    print(df.info())
    output_path = parent_dir / params['data']['processed_data_path'] / params['data']['features_data_file']
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()