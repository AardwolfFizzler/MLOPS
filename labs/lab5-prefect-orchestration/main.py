"""
Lab 5: ML Workflow Orchestration with Prefect
Main workflow file with tasks and flows
"""

import pandas as pd
import skops.io as sio
from prefect import flow, task
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


@task
def load_data(filename: str):
    """
    Load and preprocess the bank churn dataset
    
    Args:
        filename: Path to the CSV file
    
    Returns:
        DataFrame with preprocessed data
    """
    print(f"Loading data from {filename}...")
    bank_df = pd.read_csv(filename, index_col="id", nrows=1000)
    bank_df = bank_df.drop(["CustomerId", "Surname"], axis=1)
    bank_df = bank_df.sample(frac=1, random_state=42)
    print(f"Data loaded successfully. Shape: {bank_df.shape}")
    return bank_df


@task
def preprocessing(bank_df: pd.DataFrame):
    """
    Preprocess data: impute missing values, encode categorical features, and scale numerical features
    
    Args:
        bank_df: Input DataFrame
    
    Returns:
        Preprocessed DataFrame
    """
    print("Starting preprocessing...")
    cat_col = [1, 2]  # Geography, Gender
    num_col = [0, 3, 4, 5, 6, 7, 8, 9]  # Numerical columns

    # Filling missing categorical values
    cat_impute = SimpleImputer(strategy="most_frequent")
    bank_df.iloc[:, cat_col] = cat_impute.fit_transform(bank_df.iloc[:, cat_col])

    # Filling missing numerical values
    num_impute = SimpleImputer(strategy="median")
    bank_df.iloc[:, num_col] = num_impute.fit_transform(bank_df.iloc[:, num_col])

    # Encode categorical features as an integer array
    cat_encode = OrdinalEncoder()
    bank_df.iloc[:, cat_col] = cat_encode.fit_transform(bank_df.iloc[:, cat_col])

    # Scaling numerical values
    scaler = MinMaxScaler()
    bank_df.iloc[:, num_col] = scaler.fit_transform(bank_df.iloc[:, num_col])
    
    print("Preprocessing completed successfully")
    return bank_df


@task
def data_split(bank_df: pd.DataFrame):
    """
    Split data into training and testing sets
    
    Args:
        bank_df: Preprocessed DataFrame
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    print("Splitting data into train and test sets...")
    # Splitting data into training and testing sets
    X = bank_df.drop(["Exited"], axis=1)
    y = bank_df.Exited

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=125
    )
    
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


@task
def train_model(X_train, X_test, y_train):
    """
    Select best features and train the logistic regression model
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
    
    Returns:
        Trained model and transformed test set
    """
    print("Training model...")
    # Selecting the best features
    KBest = SelectKBest(chi2, k="all")
    X_train = KBest.fit_transform(X_train, y_train)
    X_test = KBest.transform(X_test)

    # Train the model
    model = LogisticRegression(max_iter=1000, random_state=125)
    model.fit(X_train, y_train)
    
    print("Model training completed")
    return model, X_test


@task
def get_prediction(X_test, model: LogisticRegression):
    """
    Generate predictions using the trained model
    
    Args:
        X_test: Testing features
        model: Trained model
    
    Returns:
        Predictions array
    """
    print("Generating predictions...")
    predictions = model.predict(X_test)
    print(f"Generated {len(predictions)} predictions")
    return predictions


@task
def evaluate_model(y_test, prediction: pd.DataFrame):
    """
    Evaluate model performance using accuracy and F1 score
    
    Args:
        y_test: True labels
        prediction: Predicted labels
    """
    print("Evaluating model...")
    accuracy = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction, average="macro")

    print("=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {round(accuracy * 100, 2)}%")
    print(f"F1 Score: {round(f1, 2)}")
    print("=" * 50)


@task
def save_model(model: LogisticRegression):
    """
    Save the trained model to disk using skops
    
    Args:
        model: Trained model to save
    """
    print("Saving model...")
    sio.dump(model, "bank_model.skops")
    print("Model saved as 'bank_model.skops'")


@flow(log_prints=True)
def ml_workflow(filename: str = "data/train.csv"):
    """
    Main ML workflow that orchestrates all tasks
    
    This flow performs the following steps:
    1. Load data from CSV file
    2. Preprocess the data (imputation, encoding, scaling)
    3. Split data into train/test sets
    4. Train a Logistic Regression model
    5. Generate predictions
    6. Evaluate model performance
    7. Save the trained model
    
    Args:
        filename: Path to the training data CSV file
    """
    print("Starting ML Workflow...")
    print("=" * 50)
    
    # Execute tasks in sequence
    data = load_data(filename)
    prep_data = preprocessing(data)
    X_train, X_test, y_train, y_test = data_split(prep_data)
    model, X_test_transformed = train_model(X_train, X_test, y_train)
    predictions = get_prediction(X_test_transformed, model)
    evaluate_model(y_test, predictions)
    save_model(model)
    
    print("=" * 50)
    print("ML Workflow completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    # Run the workflow when script is executed directly
    ml_workflow()