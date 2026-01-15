import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Assume:
# - model is a trained scikit-learn model
# - X_test, y_test are available for plotting confusion matrix

with mlflow.start_run():

    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 10)

    # Log metrics
    mlflow.log_metric("accuracy", 0.89)
    mlflow.log_metric("f1_score", 0.85)

    # Log the trained model
    mlflow.sklearn.log_model(model, artifact_path="model")
    
    
#######################################################################
    
    mlflow.register_model(
    model_uri="runs:/<run_id>/model",
    name="BestClassifierModel"
)
    
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="BestClassifierModel",
    version=1,
    stage="Staging"
)


######################################
@task
def load_data():
    # Load raw data
    return data

@task
def preprocess(data):
    # Clean / transform data
    return processed_data

@task
def train_model(processed_data):
    # Train ML model
    model = "trained_model"
    return model

@task
def evaluate(model):
    # Evaluate model
    accuracy = 0.83  # example result

    if accuracy < 0.85:
        raise ValueError(f"Model accuracy too low: {accuracy}")

    return accuracy

@flow(
    name="daily-ml-training-pipeline",
    schedule=CronSchedule(cron="0 2 * * *")  # Runs daily at 2 AM
)
def ml_pipeline():
    data = load_data()
    processed_data = preprocess(data)
    model = train_model(processed_data)
    accuracy = evaluate(model)
    return accuracy