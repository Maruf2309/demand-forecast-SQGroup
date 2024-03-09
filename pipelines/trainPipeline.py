from zenml import pipeline, client
from typing_extensions import Annotated, Tuple
import logging
from steps.ingest import ingest_data
from steps.split import split_data
from steps.train import train_models
from steps.evaluation import evaluate_model
from steps.merge_all_feature import merge_all_features
from steps.encode import encode_and_aggregate_data
from steps.clean import clean_data
import config
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

@pipeline(name='SQ Group Model Training', enable_artifact_metadata=True, enable_step_logs=True)
def trainModelPipeline(data_source: Annotated[str, 'data_source']) -> Tuple[Annotated[BaseEstimator, 'XGBRFRegressor'], Annotated[BaseEstimator, 'LGBMRegressor']]:
    """
    Pipeline that runs the data ingestion and data cleaning steps.
    """
    try:
        tracker = client.Client().active_stack.experiment_tracker.get_tracking_uri()
        logging.info(f"Running Model Tuning Pipeline: {tracker}")
        # Ingestion
        df = ingest_data(data_source=config.DATA_SOURCE)
        df = clean_data(data=df)
        targets, static_features, application_group, uses, mkt = encode_and_aggregate_data(data=df)
        features, target, imputer = merge_all_features(targets, static_features, application_group, uses, mkt)
        X_train, X_test, y_train, y_test = split_data(features=features, target=target)
        
        # Tuning Step
        xgb, lgb = train_models(X_train, y_train, X_test, y_test)

        # Evaluate Step
        # evaluated = evaluate_model(model, X_test, y_test)
    except Exception as e:
        logging.error(e)
        raise e
    finally:
        logging.info("Model pipeline completed.")
        return xgb, lgb

