from zenml import pipeline
import logging
from steps.ingest import ingest_data
from steps.clean import clean_data
from steps.encode import encode_data
from steps.split import split_data
import config


@pipeline
def run_feature_pipeline() -> None:
    """
    Pipeline that runs the data ingestion and data cleaning steps.
    """
    global ingest_data, clean_data
    try:
        logging.info("Running Feature pipeline...")
        data = ingest_data(config.DATA_SOURCE)
        data = clean_data(data)
        data = encode_data(data)
        # train, valid, test = split_data(data)
        logging.info("Feature pipeline completed.")

    except Exception as e:
        logging.error(
            f"Error running data ingestion and cleaning pipeline: {e}")


if __name__ == "__main__":
    run_feature_pipeline()
