"""
Wine Quality Prediction with ElasticNet Regression
This script trains an ElasticNet model and logs everything to MLflow
"""
import os
os.environ["MLFLOW_ENABLE_ARTIFACTS"] = "false"
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"
import warnings
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

mlflow.set_experiment("wine-quality-elasticnet")

def eval_metrics(actual, pred):
    """Calculate RMSE, MAE, and R2 score"""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train ElasticNet model on Wine Quality data')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Regularization strength (default: 0.5)')
    parser.add_argument('--l1_ratio', type=float, default=0.5,
                        help='ElasticNet mixing parameter (default: 0.5)')
    args = parser.parse_args()

    # Load the wine quality dataset from UCI repository
    print("Loading wine quality dataset...")
    data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    try:
        data = pd.read_csv(data_url, sep=";")
        print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    except Exception as e:
        print(f"ERROR: Unable to download data from {data_url}")
        print(f"Error details: {e}")
        return

    # Split the data into training and test sets (75% train, 25% test)
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    
    # Separate features (X) and target (y)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    print(f"\nTraining set: {train_x.shape[0]} samples")
    print(f"Test set: {test_x.shape[0]} samples")
    print(f"Features: {list(train_x.columns)}")

    # Start MLflow run
    with mlflow.start_run():
        print(f"\n Training ElasticNet model with alpha={args.alpha}, l1_ratio={args.l1_ratio}")
        
        # Train the ElasticNet model
        lr = ElasticNet(alpha=args.alpha, l1_ratio=args.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Make predictions on test set
        predicted_qualities = lr.predict(test_x)

        # Evaluate the model
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        # Print results
        print(f"\n Model Performance:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")

        # Log parameters to MLflow
        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("l1_ratio", args.l1_ratio)

        # Log metrics to MLflow
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log the trained model
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="model",
            )

        print("\n Model logged to MLflow")

        # Create feature importance visualization
        print("\n Creating feature importance plot...")
        feature_importance = pd.DataFrame({
            'feature': train_x.columns,
            'coefficient': lr.coef_[0] if len(lr.coef_.shape) > 1 else lr.coef_
        }).sort_values('coefficient', key=abs, ascending=False)

        plt.figure(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in feature_importance['coefficient']]
        plt.barh(feature_importance['feature'], feature_importance['coefficient'], color=colors)
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.title(f'Feature Coefficients (alpha={args.alpha}, l1_ratio={args.l1_ratio})')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        # Save and log the plot
        plot_path = "feature_importance.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(plot_path)
        plt.close()
        print(f" Feature importance plot saved and logged")

        # Create predictions vs actual plot
        plt.figure(figsize=(8, 6))
        plt.scatter(test_y, predicted_qualities, alpha=0.5)
        plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--', lw=2)
        plt.xlabel('Actual Quality')
        plt.ylabel('Predicted Quality')
        plt.title('Predictions vs Actual Values')
        plt.tight_layout()
        
        plot_path = "predictions_vs_actual.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(plot_path)
        plt.close()
        print(f"Predictions plot saved and logged")

        print(f"\n Run completed! Check MLflow UI to see your results.")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()