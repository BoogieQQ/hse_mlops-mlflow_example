import os
import numpy as np


import pandas as pd
import mlflow
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from category_encoders import TargetEncoder
from constants import MY_SURNAME, MLFLOW_TRACKING_URL, DATASET_NAME, DATASET_PATH_PATTERN, TEST_SIZE, RANDOM_STATE
from utils import get_logger, load_params
from datetime import datetime

np.random.seed(RANDOM_STATE)

STAGE_NAME = 'process_data'

CAT_ENCODERS = {
    'ohe': OneHotEncoder(drop='first', sparse_output=False),
    'ordial': OrdinalEncoder(),
    'target': TargetEncoder()
}

def process_data():
    logger = get_logger(logger_name=STAGE_NAME)
    process_data_params = load_params(stage_name=STAGE_NAME)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    mlflow.set_experiment(f"homework_{MY_SURNAME}")

    if process_data_params['use_saved_data']:
        logger.info(f'Используются ранее собранные данные. {STAGE_NAME} стадия пропущена.')
        return
        
    logger.info('Начали скачивать данные')
    dataset = load_dataset(DATASET_NAME)
    logger.info('Успешно скачали данные!')

    logger.info('Делаем предобработку данных')

    num_features = process_data_params['columns']['num_features']
    cat_features = process_data_params['columns']['cat_features']
    ignore_features = process_data_params['columns'].get('ignore_features', []) or []
    
    target_column = process_data_params['columns']['target']
    train_columns = list(set(num_features) | set(cat_features) - set(ignore_features))

    df = dataset['train'].to_pandas()
    X, y = df[train_columns], df[target_column]
    logger.info(f'    Используемые фичи: {train_columns}')

    cat_encoder_name = process_data_params['preprocess']['cat_encoder']
    preprocessor = CAT_ENCODERS[cat_encoder_name]
    
    y_transformed = (y == '>50K').astype(int)

    X_cat_prep = preprocessor.fit_transform(X[cat_features]) if cat_encoder_name != 'target' else preprocessor.fit_transform(X=X[cat_features], y=y_transformed)

    X_transformed = np.hstack([X[num_features], X_cat_prep])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_transformed, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    n = len(X_train)
    train_size = process_data_params['subsample_train_size']
    if train_size < 1.:
        random_indices = np.random.choice(n, size=int(n*train_size), replace=False)

        X_train = X_train[random_indices]
        y_train = y_train[random_indices]
        logger.info(f'    Используем подвыборку train_size={int(n*train_size)}')
    
    logger.info(f'    Размер тренировочного датасета: {len(y_train)}')
    logger.info(f'    Размер тестового датасета: {len(y_test)}')

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    with mlflow.start_run(run_name=f"data_processing_{timestamp}"):
        mlflow.log_param("dataset_name", process_data_params['dataset'])
        mlflow.log_param("features", train_columns)
        mlflow.log_param("train_size_original", n)
        mlflow.log_param("train_subsample_size", len(y_train))
        mlflow.log_param("test_size", len(y_test))
        mlflow.log_param("categorical_features", cat_features)
        mlflow.log_param("numerical_features", num_features)
        mlflow.log_param("ignore_features", ignore_features)
        mlflow.log_param("cat_encoder", cat_encoder_name)
        mlflow.log_param("target_distribution_train", y_train.value_counts().to_dict())
        mlflow.log_param("target_distribution_test", y_test.value_counts().to_dict())
    
    logger.info('Начали сохранять датасеты')
    os.makedirs(os.path.dirname(DATASET_PATH_PATTERN), exist_ok=True)
    for split, split_name in zip(
        (X_train, X_test, y_train, y_test),
        ('X_train', 'X_test', 'y_train', 'y_test'),
    ):
        pd.DataFrame(split).to_csv(
            DATASET_PATH_PATTERN.format(split_name=split_name), index=False
        )
    logger.info('Успешно сохранили датасеты!')


if __name__ == '__main__':
    process_data()
