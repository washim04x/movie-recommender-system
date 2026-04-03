import os
import mlflow

def promote_model():

    
    dagshub_key = os.getenv("Dagshub_movie")
    if not dagshub_key:
        raise ValueError("Dagshub_movie environment variable not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_key
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_key

    mlflow.set_tracking_uri("https://dagshub.com/washim04x/movie-recommender-system.mlflow")

    client = mlflow.MlflowClient()

    # List of models to promote
    models_to_promote = ["tfidf_matrix", "indices", "df"]
    
    for model_name in models_to_promote:
        try:
            # Get the latest version in staging
            latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[0].version

            # Archive the current production model
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            for version in prod_versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )

            # Promote the new model to production
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version_staging,
                stage="Production"
            )
            print(f"{model_name} version {latest_version_staging} promoted to Production")
        except Exception as e:
            print(f"Error promoting {model_name}: {e}")

if __name__ == "__main__":
    promote_model()
