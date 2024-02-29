import config
from pipelines.ETLPipeline import ETLFeaturePipeline
from pipelines.trainPipeline import trainModelPipeline


if __name__ == '__main__':
    features = ETLFeaturePipeline(data_source=config.DATA_SOURCE,)
    train = trainModelPipeline(data_source=config.FEATURE_STORE)
