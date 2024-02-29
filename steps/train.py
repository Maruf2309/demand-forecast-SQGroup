from zenml import step, client
from typing import Tuple, Annotated
import pandas as pd
import config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
import logging
from xgboost import XGBRegressor
import mlflow

# Configure Experiment Tracker
experiment_tracker = client.Client().active_stack.experiment_tracker

@step(name='Hyper-parameter Tuning Step',experiment_tracker=experiment_tracker.name,enable_artifact_metadata=True, enable_artifact_visualization=True, enable_step_logs=True)
def train_model(
        X_train: Annotated[pd.DataFrame, 'X_train'],
        y_train: Annotated[pd.DataFrame, 'y_train'],
        model_name: str) -> Annotated[BaseEstimator, 'model']:
    """
    Train a model on the training data.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.DataFrame): The training target.

    Returns:
        BaseEstimator: The trained model.
    """
    try:
        logging.info("Tuning model...")
        if model_name.lower() == 'xgb':
            # Randomized Grid Search for XGBoost hyperparameters
            grid = RandomizedSearchCV(
                XGBRegressor(),
                param_distributions=config.XGB_PARAMS_SPACE,
                scoring='neg_mean_squared_error',
                cv=5,
                verbose=1
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            mlflow.log_params(grid.best_params_)
            mlflow.xgboost.log_model(model, model_name)

        logging.info("Model Tuned successfully.")
        return model
    except Exception as e:
        logging.error(f"Error in model Tuning: {e}")
        raise e
    finally:
        logging.info("Model Tuning complete.")
