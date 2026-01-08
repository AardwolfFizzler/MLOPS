"""
Model Definition
Defines the machine learning model for weather prediction.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def get_model(model_type='random_forest', n_estimators=100, max_depth=None, random_state=42):
    """
    Get a machine learning model for binary classification.
    
    Parameters:
    -----------
    model_type : str
        Type of model ('random_forest', 'logistic', 'decision_tree')
    n_estimators : int
        Number of trees for Random Forest
    max_depth : int or None
        Maximum depth of trees
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    model : sklearn classifier
    """
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        print(f" Random Forest with {n_estimators} trees, max_depth={max_depth}")
        
    elif model_type == 'logistic':
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )
        print(f" Logistic Regression")
        
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )
        print(f" Decision Tree with max_depth={max_depth}")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def get_model_info(model):
    """Get information about the trained model."""
    info = {
        'model_type': type(model).__name__,
        'parameters': model.get_params()
    }
    
    # For tree-based models, get feature importance
    if hasattr(model, 'feature_importances_'):
        info['has_feature_importance'] = True
    
    return info