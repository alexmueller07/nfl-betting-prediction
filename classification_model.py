import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import DMatrix, train

def train_classification_model(teams_scores_classification):
    """
    Train XGBoost classification model for predicting home team wins.
    """
    print("=== Training Classification Model ===")
    
    # Prepare data
    y = teams_scores_classification['home_team_wins']
    X = teams_scores_classification.drop(columns=['home_team_wins'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 3,
        'eta': 0.01,
        'subsample': 0.75,
        'colsample_bytree': 1.0,
        'random_state': 42,
        'scale_pos_weight': scale_pos_weight
    }

    # Train with early stopping
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evallist,
        early_stopping_rounds=10,
        verbose_eval=True
    )

    # Predict and evaluate
    y_pred_prob_test = model.predict(dtest)
    y_pred_test = np.where(y_pred_prob_test > 0.5, 1, 0)

    y_pred_prob_train = model.predict(dtrain)
    y_pred_train = np.where(y_pred_prob_train > 0.5, 1, 0)

    # Print accuracy on both
    print("Training Accuracy:", round(accuracy_score(y_train, y_pred_train), 4))
    print("Test Accuracy:", round(accuracy_score(y_test, y_pred_test), 4))

    # Classification report (test set)
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test))

    # Feature importance
    xgb.plot_importance(model, max_num_features=20)
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.show()
    
    return model, X, y

def train_final_classification_model(teams_scores_classification):
    """
    Train final classification model on the full dataset without test split.
    """
    print("=== Training Final Classification Model on Full Dataset ===")
    
    # Retrain Classification Model on All Data
    X_full_class = teams_scores_classification.drop(columns=['home_team_wins'])
    y_full_class = teams_scores_classification['home_team_wins']

    # Handle class imbalance on full set
    scale_pos_weight_full = len(y_full_class[y_full_class == 0]) / len(y_full_class[y_full_class == 1])

    # DMatrix for full data
    dfull = DMatrix(X_full_class, label=y_full_class)

    # Same parameters as before
    params_full = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 3,
        'eta': 0.01,
        'subsample': 0.75,
        'colsample_bytree': 1.0,
        'random_state': 42,
        'scale_pos_weight': scale_pos_weight_full
    }

    # Train final classification model
    final_class_model = train(
        params=params_full,
        dtrain=dfull,
        num_boost_round=100
    )

    print("✅ Final classification model trained on full dataset.")
    
    return final_class_model 