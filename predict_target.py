import pandas as pd
import numpy as np
from collections import Counter

import warnings
warnings.filterwarnings("ignore")


def get_calibrated_predictions(model, x_test, threshold, log=True):
    """
    Predict target and calibrate with threshold probability
    """
    predicts = model.predict(x_test)
    predicts_probas = model.predict_proba(x_test)[:, 1]
    calibrated_predicts = predicts[np.where(predicts_probas > threshold)] = 1
    if log:
        print(f'До калибровки: {Counter(predicts)}')
        print(f'После калибровки: {Counter(calibrated_predicts)}')
    return calibrated_predicts


def save_predictions(predicts, test, pred_path='predictions/', name='Predictions'):
    pred_is_churned = pd.DataFrame({'user_id': test['user_id'], 'is_churned': predicts},
                                   columns=['user_id', 'is_churned'])
    pred_is_churned.to_csv(f'{pred_path}{name}.csv', sep=',', index=None, encoding='utf-8')
