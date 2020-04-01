from make_dataset import *
from processing_dataset import *
from model_training import *
from model_score import *
from predict_target import *
import pickle

CHURNED_START_DATE = '2019-09-01'
CHURNED_END_DATE = '2019-10-01'

INTER_1 = (1, 7)
INTER_2 = (8, 14)
INTER_3 = (15, 21)
INTER_4 = (22, 28)
INTER_LIST = [INTER_1, INTER_2, INTER_3, INTER_4]


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

prepare_dataset(dataset=pd.read_csv('dataset/dataset_raw_train.csv', sep=';'), dataset_type='train')
prepare_dataset(dataset=pd.read_csv('dataset/dataset_raw_test.csv', sep=';'), dataset_type='test')

train_new = pd.read_csv('dataset/dataset_train.csv', sep=';')
test_new = pd.read_csv('dataset/dataset_test.csv', sep=';')






model = xgb_fit_predict(X_train_balanced, y_train_balanced, X_test, y_test)
predict_test = model.predict(X_test)
predict_test_probas = model.predict_proba(X_test)[:, 1]

plot_confusion_matrix(y_test.values, predict_test, classes=['churn', 'active'])
plt.show()

plot_PR_curve(y_test.values, predict_test, predict_test_probas)
plt.show()

plot_ROC_curve(classifier=model,
               X=X_test,
               y=y_test.values,
               n_folds=3)

with open('models/baseline_xgb.pcl', 'wb') as f:
    pickle.dump(model, f)
