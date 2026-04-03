import mlflow
import json
from pathlib import Path
import os

dagshub_key = os.getenv("DAGSHUB_MOVIE")
if not dagshub_key:
    raise ValueError("DAGSHUB_MOVIE environment variable not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_key
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_key

mlflow.set_tracking_uri(
    "https://dagshub.com/washim04x/movie-recommender-system.mlflow"
)


def load_model_info(file_path):
    """Load model metadata from JSON."""
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
            print(f"Model info loaded from {file_path}")
            return model_info
    except Exception as e:
        print(f"Error occurred while loading the model info: {e}")
        raise


def register_model_uri(model_name, model_uri):
    """Register the model in MLflow Model Registry."""
    print(f"Registering {model_name} from URI: {model_uri}")

    try:
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
        )
        print(
            f"{model_name} version {model_version.version} "
            "registered and promoted to Staging"
        )
    except Exception as e:
        print(f"Error occurred while registering {model_name}: {e}")
        raise


def handle_artifact_workflow(model_info):
    """Handle workflow where model_info stores raw .joblib artifact URIs."""
    artifact_uris = model_info.get("artifact_uris", {})
    if not artifact_uris:
        return False

    print("Detected artifact-only workflow in model_info.json")
    for name, uri in artifact_uris.items():
        print(f"- {name}: {uri}")

    print(
        "Raw .joblib artifacts are ready for app download. "
        "Skipping registry registration because these URIs "
        "are not MLflow model folders."
    )
    return True


if __name__ == "__main__":
    curr_dir = Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    model_info_path = home_dir.as_posix() + "/reports/model_info.json"
    model_info = load_model_info(model_info_path)

    if handle_artifact_workflow(model_info):
        raise SystemExit(0)

    run_id = model_info["run_id"]
    register_model_uri("tfidf_matrix", f"runs:/{run_id}/tfidf_matrix")
    register_model_uri("indices", f"runs:/{run_id}/indices")
    register_model_uri("df", f"runs:/{run_id}/df")
