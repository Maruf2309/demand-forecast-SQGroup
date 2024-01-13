from zenml import pipeline
import logging
from steps.ingest import ingest_data
from steps.clean import clean_data
import config


@pipeline
def run_pipeline() -> None:
    """
    Pipeline that runs the data ingestion and data cleaning steps.
    """
    global ingest_data, clean_data
    try:
        logging.info("Running data ingestion and cleaning pipeline...")
        data = ingest_data(config.DATA_SOURCE)
        # data = clean_data(data)
        logging.info("Data ingestion and cleaning pipeline completed.")

    except Exception as e:
        logging.error(
            f"Error running data ingestion and cleaning pipeline: {e}")


if __name__ == "__main__":
    run_pipeline()
