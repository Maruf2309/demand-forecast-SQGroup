from zenml import step, client
from typing import Tuple, Annotated
import pandas as pd
import config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
import logging
from xgboost import XGBRFRegressor
from lightgbm import LGBMRegressor
import mlflow
from steps.evaluation import evaluate_model

# Configure Experiment Tracker
experiment_tracker = client.Client().active_stack.experiment_tracker

@step(name='Hyper-parameter Tuning Step',experiment_tracker=experiment_tracker.name,enable_artifact_metadata=True, enable_artifact_visualization=True, enable_step_logs=True)
def train_models(
        X_train: Annotated[pd.DataFrame, 'X_train'],
        y_train: Annotated[pd.Series, 'y_train'],
        X_test: Annotated[pd.DataFrame, 'X_test'],
        y_test: Annotated[pd.Series, 'y_test'],
    ) -> Tuple[Annotated[BaseEstimator, 'XGBRFRegressor'], Annotated[BaseEstimator, 'LGBMRegressor']]:
    """
    Train a model on the training data.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.DataFrame): The training target.
        X_test (pd.DataFrame): The test features.
        y_test (pd.DataFrame): The test target.

    Returns:
        Tuple[BaseEstimator: The trained model, BaseEstimator: The trained model]
    """
    try:
        logging.info("Tuning model...")

        with mlflow.start_run(run_name="XGBoost Run", nested=True) as xgb_run:
            # Randomized Grid Search for XGBoost hyperparameters
            grid = RandomizedSearchCV(
                XGBRFRegressor(),
                param_distributions=config.XGB_PARAMS_SPACE,
                scoring='r2',
                cv=5,
                verbose=0
            )
            grid.fit(X_train[config.SELECTED_FEATURES].values, y_train)
            xgb = grid.best_estimator_
            mlflow.log_params(grid.best_params_)
            mlflow.log_param('components', len(config.SELECTED_FEATURES))
            mlflow.log_param('columns', config.SELECTED_FEATURES)
            ### log metrics
            r2 = evaluate_model(grid.best_estimator_, X_test[config.SELECTED_FEATURES].values, y_test)
            if r2 > config.R2_THRESHOLD:
                mlflow.xgboost.log_model(xgb, f'XGB-QTYM-{r2}')
        
        
        with mlflow.start_run(run_name="LGB Run", nested=True) as lgb_run:
            # Randomized Grid Search for XGBoost hyperparameters
            grid = RandomizedSearchCV(
                LGBMRegressor(),
                param_distributions=config.LGB_PARAMS_SPACE,
                scoring='r2',
                cv=5,
                verbose=0
            )
            grid.fit(X_train[config.SELECTED_FEATURES].values, y_train)
            lgb = grid.best_estimator_
            mlflow.log_params(grid.best_params_)
            mlflow.log_param('components', len(config.SELECTED_FEATURES))
            mlflow.log_param('columns', config.SELECTED_FEATURES)

            ### log metrics
            r2 = evaluate_model(grid.best_estimator_, X_test[config.SELECTED_FEATURES].values, y_test)
            if r2 > config.R2_THRESHOLD:
                mlflow.lightgbm.log_model(lgb, f'LGB-QTYM-{r2}')

        logging.info("Model Tuned successfully.")
        return xgb, lgb
    except Exception as e:
        logging.error(f"Error in model Tuning: {e}")
        raise e
    finally:
        logging.info("Model Tuning complete.")
