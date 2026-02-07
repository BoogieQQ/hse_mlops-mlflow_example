import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from joblib import load
from sklearn.metrics import (
    get_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    ConfusionMatrixDisplay
)

from datetime import datetime

from constants import DATASET_PATH_PATTERN, MODEL_FILEPATH, MY_SURNAME, MLFLOW_TRACKING_URL, ARTIFACTS_PATH_PATTERN
from utils import get_logger, load_params

STAGE_NAME = 'evaluate'


def evaluate():
    logger = get_logger(logger_name=STAGE_NAME)
    params = load_params(stage_name=STAGE_NAME)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URL)
    mlflow.set_experiment(f"homework_{MY_SURNAME}")

    model_type = load_params(stage_name='train')['model_type']

    logger.info('Начали считывать датасеты')
    splits = [None, None, None, None]
    for i, split_name in enumerate(['X_train', 'X_test', 'y_train', 'y_test']):
        splits[i] = pd.read_csv(DATASET_PATH_PATTERN.format(split_name=split_name))
    X_train, X_test, y_train, y_test = splits
    logger.info('Успешно считали датасеты!')
    
    logger.info('Загружаем обученную модель')
    if not os.path.exists(MODEL_FILEPATH):
        raise FileNotFoundError(
            'Не нашли файл с моделью. Убедитесь, что был запущен шаг с обучением'
        )
    model = load(MODEL_FILEPATH)

    logger.info('Делаем предсказания')
    y_pred = model.predict(X_test)
    
    y_proba = model.predict_proba(X_test)[:, 1]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    with mlflow.start_run(run_name=f"{model_type}_evaluation_{timestamp}") as run:
        logger.info('Начали считать метрики на тесте')
        metrics = {}
     
        for metric_name in params['metrics']:
            try:
                scorer = get_scorer(metric_name)
                score = scorer(model, X_test, y_test)
                metrics[metric_name] = score
            except Exception as e:
                logger.warning(f'Не удалось вычислить метрику {metric_name}: {e}')
        
        logger.info(f'Значения метрик - {metrics}')
        
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        logger.info('Создаем confusion matrix')
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(12, 8))
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(cmap=plt.cm.Blues)
        
        plt.title('Confusion Matrix')
        
        cm_img_path = ARTIFACTS_PATH_PATTERN.format(name="confusion_matrix.pdf")
        plt.savefig(cm_img_path)
        mlflow.log_artifact(cm_img_path, "evaluation_artifacts")
        
        logger.info('Проверяем feature importances')
        if model_type != 'logreg':
            feature_names = X_train.columns
            importances = model.feature_importances_
        else:
            feature_names = X_train.columns
            importances = np.abs(model.coef_[0])
            
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        feature_importance_path = ARTIFACTS_PATH_PATTERN.format(name="feature_importances.csv")
        feature_importance_df.to_csv(feature_importance_path, index=False)
        mlflow.log_artifact(feature_importance_path, "evaluation_artifacts")

       
        logger.info('Создаем PR-кривую')
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        
        plt.figure(figsize=(12, 8))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Rec')
        plt.ylabel('Pr')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        pr_curve_path = ARTIFACTS_PATH_PATTERN.format(name="pr_curve.pdf")
        plt.savefig(pr_curve_path)
        mlflow.log_artifact(pr_curve_path, "evaluation_artifacts")
            
        
        logger.info(f'Артефакты валидации успешно сохранены.')

if __name__ == '__main__':
    evaluate()