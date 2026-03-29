import pandas as pd
from pathlib import Path
import yaml

def process_data(df):
    # Drop colums that more then 80% of the values are missing
    df.drop(columns=['belongs_to_collection', 'homepage'], inplace=True)
    df = df.drop_duplicates().reset_index(drop = True)

    # Remove duplicates based on the 'original_title' and 'release_date' columns
    df['name'] = df['original_title'].str.strip().str.replace(' ', '').str.lower()
    df.drop_duplicates(subset=['name','release_date'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Select only the relevant columns for the recommendation system
    cols = ['original_title', 'overview', 'tagline', 'genres', 'vote_average', 'production_companies']
    df = df[cols]
     
    # Fill missing values with empty strings or the mean for 'vote_average'
    df['overview'] = df['overview'].fillna(' ')
    df['tagline'] = df['tagline'].fillna(' ') 
    df['production_companies'] = df['production_companies'].fillna(' ')
    df['vote_average'] = df['vote_average'].fillna(df['vote_average'].mean())

    return df




def main():
    # Load the dataset
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    params_path = home_dir / 'params.yaml'
    params = yaml.safe_load(open(params_path))

    df_path = home_dir / params['data']['raw_data_path'] / params['data']['raw_data_file']
    df = pd.read_csv(df_path, low_memory=False)

    # Process the data
    processed_df = process_data(df)

    processed_df_path = home_dir / params['data']['processed_data_path'] / params['data']['processed_data_file']
    # Save the processed data to a new CSV file
    processed_df.to_csv(processed_df_path, index=False)
    print(f"Processed data saved to {processed_df_path}")

if __name__ == "__main__":
    main()