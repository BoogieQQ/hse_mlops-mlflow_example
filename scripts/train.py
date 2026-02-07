import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, RANDOM_STATE, MY_SURNAME, MLFLOW_TRACKING_URL
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
    logger.info(train_params)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    mlflow.set_experiment(f"homework_{MY_SURNAME}")

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

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    with mlflow.start_run(run_name=f"{model_type}_training_{timestamp}") as run:
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("random_state", RANDOM_STATE)

        if model_type == 'logreg':
            logger.info('Для логистической регрессии добавляем StandardScaler')
            
            pipeline_steps = [
                ('scaler', StandardScaler()),
                ('classifier', model_class(**model_params))
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
        import joblib
        joblib.dump(model, MODEL_FILEPATH)
        
        logger.info(f'Успешно!')

if __name__ == '__main__':
    train()
