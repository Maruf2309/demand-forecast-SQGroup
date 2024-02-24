from pipelines.ETLPipeline import ETLFeaturePipeline



if __name__ == '__main__':
    features = ETLFeaturePipeline(data_source='data/sales_BYA.csv')
