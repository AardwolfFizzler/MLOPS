import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import boto3
import joblib
from datetime import datetime
import os

# Download necessary data - Iris data from sklearn library
# We define a function to download the data
def download_data():
  from sklearn.datasets import load_iris
  iris = load_iris()
  features = pd.DataFrame(iris.data, columns=iris.feature_names)
  target = pd.Series(iris.target)
  return features, target

# Define a function to preprocess the data
# In this case, preprocessing will be just splitting the data into training and testing sets
def preprocess_data(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  return X_train, X_test, y_train, y_test

# Define a function to train the model
def train_model(X_train, y_train):
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)
  return model

# Define a function to save the model both locally and in S3
def save_model_to_s3(model, bucket_name, object_key):
  # Save model locally first
  local_filename = "model.joblib"
  joblib.dump(model, local_filename)
  
  # Upload to S3
  s3_client = boto3.client('s3')
  s3_client.upload_file(local_filename, bucket_name, object_key)
  
  # Clean up local file
  os.remove(local_filename)

# Putting all functions together
def main():
  # Download data
  X, y = download_data()
  X_train, X_test, y_train, y_test = preprocess_data(X, y)
  
  # Train model
  model = train_model(X_train, y_train)
  
  # Evaluate model
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f'Model accuracy: {accuracy}')
  
  # Save the model to S3
  bucket_name = os.environ.get('AWS_S3_BUCKET', 'your-bucket-name')
  timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
  object_key = f"trained_models/model_{timestamp}.joblib"
  save_model_to_s3(model, bucket_name, object_key)
  print(f"Model saved to s3://{bucket_name}/{object_key}")
  
if __name__ == "__main__":
  main()
