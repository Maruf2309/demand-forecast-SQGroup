DATA_SOURCE = 'data/sales_BYA.parquet'
ARTIFACTS_DIR = 'artifacts'
FEATURE_STORE = 'data/features.parquet'
MODEL_STORE = 'data/model.pkl'
TEST_SIZE = 0.2
RANDOM_STATE = 42
XGB_PARAMS_SPACE = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.3, 0.5],
    'n_estimators': [100, 200, 300],
    'subsample': [0.5, 0.7, 0.9],
    'colsample_bytree': [0.5, 0.7, 0.9],
    'gamma': [0, 0.1, 0.2]
}
