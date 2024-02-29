from zenml import pipeline, client
from typing_extensions import Annotated
import logging
from steps.ingest import ingest_data
from steps.split import split_data
from steps.train import train_model
from steps.evaluation import evaluate_model
import config
import pandas as pd


@pipeline(name='SQ Group Model Training', enable_artifact_metadata=True, enable_step_logs=True)
def trainModelPipeline(data_source: Annotated[str, 'data_source']) -> Annotated[pd.DataFrame, 'features']:
    """
    Pipeline that runs the data ingestion and data cleaning steps.
    """
    global ingest_data, clean_data, encode_data, AddTemporalFeatures, AddLagFeatures, AddWindowFeatures, AddExpWindowFeatures, scale_data
    try:
        tracker = client.Client().active_stack.experiment_tracker.get_tracking_uri()
        logging.info(f"Running Model Tuning Pipeline: {tracker}")
        # Ingest Step
        data = ingest_data(data_source=data_source)

        # Split Step
        X_train, X_test, y_train, y_test = split_data(data=data)
        
        # Tuning Step
        model = train_model(X_train, y_train, model_name='xgb')

        # Evaluate Step
        evaluated = evaluate_model(model, X_test, y_test)

        
    except Exception as e:
        logging.error(e)
        raise e
    finally:
        logging.info("Model pipeline completed.")
        return data

