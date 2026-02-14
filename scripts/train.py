import mlflow
import mlflow.sklearn
import joblib

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE
from utils import get_logger, load_params

STAGE_NAME = 'train'

MODEL_CLASSES = {
    'logreg':       LogisticRegression,
    'randomforest': RandomForestClassifier,
    'boosting':     GradientBoostingClassifier
}

def train():
    logger = get_logger(logger_name=STAGE_NAME)
    train_params = load_params(stage_name=STAGE_NAME)

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')
    
    model_type   = train_params['model_type']
    model_params = train_params[model_type]
    model_params['random_state'] = RANDOM_STATE
    
    logger.info(f'Создаём модель типа: {model_type}')
    logger.info(f'    Параметры модели: {model_params}')
    
    model_class = MODEL_CLASSES[model_type]
    
    model = model_class(**model_params)

    mlflow.log_param("model_type", model_type)
    mlflow.log_param("random_state", RANDOM_STATE)

    if model_type == 'logreg':
        logger.info('Для логистической регрессии добавляем StandardScaler')
        
        pipeline_steps = [
            ('scaler', StandardScaler()),
            ('logreg', model_class(**model_params))
        ]
        model = Pipeline(pipeline_steps)
        
        mlflow.log_param("has_scaler", True)
        mlflow.log_param("scaler_type", "StandardScaler")
    else:
        model = model_class(**model_params)
        mlflow.log_param("has_scaler", False)
        
    for param_name, param_value in model_params.items():
        mlflow.log_param(f"model_{param_name}", param_value)
    
    logger.info('Обучаем модель')
    model.fit(X_train, y_train)
    
    logger.info('Логируем модель в MLflow')
    
    mlflow.sklearn.log_model(model, "model")
    
    logger.info('Сохраняем модель')
    joblib.dump(model, MODEL_FILEPATH)
    
    logger.info(f'Успешно!')

if __name__ == '__main__':
    train()
