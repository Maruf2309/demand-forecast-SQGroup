from zenml import step, client
from typing import Tuple, Annotated, Union, Any
import pandas as pd
import config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
import logging
from xgboost import XGBRFRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, core
import mlflow
from steps.evaluation import evaluate_model
import shelve
from model.schema import signature

# Configure Experiment Tracker
experiment_tracker = client.Client().active_stack.experiment_tracker

@step(name='Hyper-parameter Tuning Step',experiment_tracker=experiment_tracker.name,enable_artifact_metadata=True, enable_artifact_visualization=True, enable_step_logs=True, enable_cache=False)
def train_models(
        X_train: Annotated[pd.DataFrame, 'X_train'],
        y_train: Annotated[pd.Series, 'y_train'],
        X_test: Annotated[pd.DataFrame, 'X_test'],
        y_test: Annotated[pd.Series, 'y_test'],
    ) -> Tuple[Annotated[Any, 'model'], Annotated[float, 're_score']]:
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
        logging.info(f'Track Experiments here: {client.Client().active_stack.experiment_tracker.get_tracking_uri()}')
        r2_best = 0.0
        best_model = None
        xgb = None
        lgb = None
        catboost = None
        model = None
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
            r2_xgb = evaluate_model(grid.best_estimator_, X_test[config.SELECTED_FEATURES].values, y_test)
            if r2_xgb > config.R2_THRESHOLD:
                if r2_xgb > r2_best:
                    r2_best = r2_xgb
                    best_model = 'xgb'
                mlflow.xgboost.log_model(xgb, config.MODEL_NAME, signature=signature)
        
        
        with mlflow.start_run(run_name="LGB Run", nested=True) as lgb_run:
            # Randomized Grid Search for LGBM hyperparameters
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
            r2_lgbm = evaluate_model(grid.best_estimator_, X_test[config.SELECTED_FEATURES].values, y_test)
            if r2_lgbm > config.R2_THRESHOLD:
                if r2_lgbm > r2_best:
                    r2_best = r2_lgbm
                    best_model = 'lgb'
                mlflow.lightgbm.log_model(lgb, config.MODEL_NAME, signature=signature)


        # with mlflow.start_run(run_name="CatBoost Run", nested=True) as catboost_run:
        #     # Randomized Grid Search for LGBM hyperparameters
        #     grid = RandomizedSearchCV(
        #         CatBoostRegressor(),
        #         param_distributions=config.CAT_PARAMS_SPACE,
        #         scoring='r2',
        #         cv=5,
        #         verbose=0
        #     )
        #     grid.fit(X_train[config.SELECTED_FEATURES].values, y_train)
        #     catboost = grid.best_estimator_
        #     mlflow.log_params(grid.best_params_)
        #     mlflow.log_param('components', len(config.SELECTED_FEATURES))
        #     mlflow.log_param('columns', config.SELECTED_FEATURES)

        #     ### log metrics
        #     r2_catboost= evaluate_model(grid.best_estimator_, X_test[config.SELECTED_FEATURES].values, y_test)
        #     if r2_catboost > config.R2_THRESHOLD:
        #         if r2_catboost > r2_best:
        #             r2_best = r2_catboost
        #             best_model = 'catboost'
        #         mlflow.catboost.log_model(catboost, config.MODEL_NAME)

        # Get the run ID
        run_id = mlflow.active_run().info.run_id
        
        # Register the model
        if best_model == 'xgb':
            mlflow.xgboost.log_model(xgb, config.MODEL_NAME, signature=signature)
            model = xgb

        if best_model == 'lgb':
            mlflow.lightgbm.log_model(lgb, config.MODEL_NAME, signature=signature) 
            model = lgb
        
        if best_model == 'catboost':
            mlflow.catboost.log_model(catboost, config.MODEL_NAME, signature=signature)
            model = catboost
        # Get latest best R2
        # Read R-squared score from shelf file
        with shelve.open(config.CACHE) as db:
            if  r2_best > db['r2_score']:
                db['r2_score'] = r2_best
                db['last_best_run_id'] = run_id
                db['last_best_model'] = best_model
                # mlflow.register_model(f"runs:/{run_id}/model", config.MODEL_NAME)
    except Exception as e:
        logging.error(f"Error in model Tuning: {e}")
        raise e
    finally:
        logging.info("Model Tuning complete.")
        return model, r2_best
