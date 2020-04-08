from make_dataset import *
from preprocessing import *
from model_training import *
from predict_target import *

import pandas as pd
import warnings
warnings.filterwarnings("ignore")


CHURNED_START_DATE = '2019-09-01'
CHURNED_END_DATE = '2019-10-01'

INTER_1 = (1, 7)
INTER_2 = (8, 14)
INTER_3 = (15, 21)
INTER_4 = (22, 28)
INTER_LIST = [INTER_1, INTER_2, INTER_3, INTER_4]

N_FEATURES = 20


build_dataset_raw(churned_start_date=CHURNED_START_DATE,
                  churned_end_date=CHURNED_END_DATE,
                  inter_list=INTER_LIST,
                  raw_data_path='train/',
                  dataset_path='dataset/',
                  mode='train')

build_dataset_raw(churned_start_date=CHURNED_START_DATE,
                  churned_end_date=CHURNED_END_DATE,
                  inter_list=INTER_LIST,
                  raw_data_path='test/',
                  dataset_path='dataset/',
                  mode='test')

prepare_dataset(dataset=pd.read_csv('dataset/dataset_raw_train.csv', sep=';'),
                dataset_type='train',
                inter_list=INTER_LIST)

prepare_dataset(dataset=pd.read_csv('dataset/dataset_raw_test.csv', sep=';'),
                dataset_type='test',
                inter_list=INTER_LIST)

train = pd.read_csv('dataset/dataset_train.csv', sep=';')
test = pd.read_csv('dataset/dataset_test.csv', sep=';')

x_train_bal, y_train_bal, x_test = get_data_for_prediction(train, test, get_best_feature(train, N_FEATURES))

model = get_fitted_model(x_train_bal, y_train_bal)
save_model(model, model_path='models/', name='model_xgb_final')

predicts = get_calibrated_predictions(model, x_test, threshold=0.3, log=True)
save_predictions(predicts, test, pred_path='predictions/', name='Predictions')
print('Predictions successfully created')
