from zenml import pipeline, client
from typing_extensions import Annotated, Tuple, Dict, Any
import logging
from steps.ingest import ingest_data
from steps.split import split_data
from steps.train import train_models
import config, os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from zenml.steps.step_output import Output

@pipeline(name='SQ Group Continuous Training', enable_artifact_metadata=True, enable_step_logs=True, enable_cache=False)
def trainModelPipeline() -> Tuple[Annotated[Any, 'model'], Annotated[float, 're_score']]:
    """
    Pipeline that runs the data ingestion and data cleaning steps.
    """
    try:
        tracker = client.Client().active_stack.experiment_tracker.get_tracking_uri()
        logging.info(f"Running Model Tuning Pipeline: {tracker}")

        # Ingest Features and Target
        feature_path = os.path.join(config.FEATURE_STORE, 'features.parquet')
        data = ingest_data(data_source=feature_path)

        # Split
        X_train, X_test, y_train, y_test = split_data(data=data)
        
        # Tuning Step
        model, r2 = train_models(X_train, y_train, X_test, y_test)
    except Exception as e:
        logging.error(e)
        raise e
    finally:
        logging.info("Model pipeline completed.")
        return model, r2 

