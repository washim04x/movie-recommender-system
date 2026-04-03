import json
import os
import joblib
from pathlib import Path

import mlflow
import yaml
from sklearn.metrics.pairwise import cosine_similarity


# setup dagshub logging
dagshub_key = os.getenv("DAGSHUB_MOVIE")
if not dagshub_key:
    raise ValueError("DAGSHUB_MOVIE environment variable not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_key
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_key
mlflow.set_tracking_uri(
    "https://dagshub.com/washim04x/movie-recommender-system.mlflow"
)


def save_model_info(run_id, model_path, file_path, artifact_uris=None) -> None:
    """Save the model run ID, path, and artifact URIs to a JSON file."""
    try:
        model_info = {
            "run_id": run_id,
            "model_path": str(model_path),
            "artifact_uris": artifact_uris or {},
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        print(f"Model info saved to {file_path}")
    except Exception as e:
        print(f"Error occurred while saving the model info: {e}")
        raise


def recommend_movies(movie_title, df, tfidf_matrix, indices, n=10):
    title = movie_title.strip().replace(" ", "").lower()
    if title not in indices:
        return f"Movie '{movie_title}' not found in the dataset."

    idx = indices[title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_idx = sim_scores.argsort()[::-1][1:n + 1]
    return df["original_title"].iloc[sim_idx]


def main():
    mlflow.set_experiment("Model_Evaluation_Experiment")
    with mlflow.start_run() as run:
        try:
            curr_dir = Path(__file__)
            parent_dir = curr_dir.parent.parent.parent
            params_path = parent_dir / "params.yaml"
            params = yaml.safe_load(open(params_path))

            models_path = parent_dir / "models"
            tfidf_matrix_path = models_path / "tfidf_matrix.joblib"
            indices_path = models_path / "indices.joblib"
            df_path = models_path / "df.joblib"

            tfidf_matrix = joblib.load(tfidf_matrix_path)
            indices = joblib.load(indices_path)
            df = joblib.load(df_path)

            movie_title = params["predict"]["movie_title"]
            recommendations = recommend_movies(
                movie_title,
                df,
                tfidf_matrix,
                indices,
                n=params["predict"]["n_recommendations"],
            )
            print(f"Recommendations for '{movie_title}':")
            print(recommendations)

            if recommendations is None:
                model_params = params["parameters"]["model"]
                for param_name, param_value in model_params.items():
                    mlflow.log_param(param_name, param_value)

            artifact_path = "model_artifacts"
            mlflow.log_artifact(
                str(tfidf_matrix_path),
                artifact_path=artifact_path,
            )
            mlflow.log_artifact(
                str(indices_path),
                artifact_path=artifact_path,
            )
            mlflow.log_artifact(
                str(df_path),
                artifact_path=artifact_path,
            )

            artifact_uris = {
                "tfidf_matrix": (
                    f"runs:/{run.info.run_id}/{artifact_path}/"
                    f"{tfidf_matrix_path.name}"
                ),
                "indices": (
                    f"runs:/{run.info.run_id}/{artifact_path}/"
                    f"{indices_path.name}"
                ),
                "df": (
                    f"runs:/{run.info.run_id}/{artifact_path}/"
                    f"{df_path.name}"
                ),
            }

            save_model_info(
                run.info.run_id,
                models_path,
                "reports/model_info.json",
                artifact_uris=artifact_uris,
            )
            print("Joblib artifacts logged successfully to MLflow")

            mlflow.log_artifact(
                "reports/model_info.json",
                artifact_path="model_evaluation",
            )
        except Exception:
            raise


if __name__ == "__main__":
    main()