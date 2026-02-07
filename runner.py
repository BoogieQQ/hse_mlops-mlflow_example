import mlflow

from scripts import evaluate, process_data, train
from datetime import datetime
from constants import MY_SURNAME, MLFLOW_TRACKING_URL


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    mlflow.set_experiment(f"homework_{MY_SURNAME}")

    with mlflow.start_run(run_name=f"best_run_{timestamp}") as run:
        process_data()
        train()
        evaluate()
