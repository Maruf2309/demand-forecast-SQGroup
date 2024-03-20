DATA_SOURCE = 'data/sales_bya.parquet'
ARTIFACTS_DIR = 'artifacts'
FEATURE_STORE = 'data'
ARTIFACTS_DIR = 'artifacts'
TEST_SIZE = 0.2
RANDOM_STATE = 42
R2_THRESHOLD = 0.7
MODEL_NAME = 'model'
XGB_PARAMS_SPACE = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.1, 0.3, 0.5],
    'n_estimators': [100, 200, 300],
    'subsample': [0.5, 0.7, 0.9],
    'colsample_bytree': [0.5, 0.7, 0.9],
    'gamma': [0, 0.1, 0.2, 0.01],
    'random_state': [42]  # Ensure reproducibility
}

LGB_PARAMS_SPACE = {
            'boosting_type': ['gbdt', 'dart', 'goss'],
            'num_leaves': [20, 30, 40, 50, 60],
            'max_depth': [5, 7, 9, 11, 15, 21],  # -1 means no limit
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'n_estimators': [50, 100, 200, 300],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0.0, 0.1, 0.5, 1.0, 0.01],
            'reg_lambda': [0.0, 0.1, 0.5, 1.0],
            'min_child_samples': [20, 30, 40, 50, 60],
            'random_state': [42]  # Ensure reproducibility
        }

CAT_PARAMS_SPACE = {
    'learning_rate': [0.01, 0.05, 0.1],  # Learning rate for gradient boosting
    'depth': [4, 6, 8, 10],              # Depth of the trees
    'l2_leaf_reg': [1, 3, 5, 7],          # L2 regularization coefficient
    'random_state': [42]  # Ensure reproducibility
}

SELECTED_FEATURES = [
    'General', 'Moderate', 'Rich', 'Industry', 'House Wiring',
    'Fan & Lighting Connection', 'Lift & Heavy Item', 'Earthing',
    'Industry, Machineries', 'Urban', 'Rural', 'Semi Urban', 'Others',
    'wire', 'rm', 'fy', 'grade', 'noc', 'dfc', 'area_km2', 'population',
    'literacy_rate_perc', 'pcx', 'excnts', 'exach', 'trc', 'tlcolt', 'tmtm',
    'ecoind', 'sf', 'sop', 'pminx', 'tms_cr', 'mas', 'kpi',
    'timestamp_month', 'timestamp_week', 'timestamp_day_of_week',
    'timestamp_day_of_month', 'timestamp_day_of_year', 'net_price_lag_3',
    'qtym_lag_3', 'net_price_lag_8', 'qtym_lag_8', 'net_price_lag_16',
    'qtym_lag_16', 'net_price_lag_24', 'qtym_lag_24',
    'net_price_window_24_mean', 'qtym_window_24_mean',
    'net_price_expanding_std', 'qtym_expanding_std'
]

# Cache
CACHE = 'data/cache.shlf'
# Configure Experiment Tracker
from zenml.client import Client
EXPERIMENT_TRACKER = Client().active_stack.experiment_tracker