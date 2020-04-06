import time
from datetime import timedelta
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")


def time_format(sec):
    """
    Time formating
    """
    return str(timedelta(seconds=sec))


def prepare_dataset(dataset,
                    dataset_type='train',
                    dataset_path='dataset/',
                    inter_list=[(1, 7), (8, 14)]):
    print(dataset_type)
    start_t = time.time()
    print('Dealing with missing values, outliers, categorical features...')

    # Профили
    dataset['age'] = dataset['age'].fillna(dataset['age'].median())
    dataset['gender'] = dataset['gender'].fillna(dataset['gender'].mode()[0])
    dataset.loc[~dataset['gender'].isin(['M', 'F']), 'gender'] = dataset['gender'].mode()[0]
    dataset['gender'] = dataset['gender'].map({'M': 1., 'F': 0.})
    dataset.loc[(dataset['age'] > 80) | (dataset['age'] < 7), 'age'] = round(dataset['age'].median())
    dataset.loc[dataset['days_between_fl_df'] < -1, 'days_between_fl_df'] = -1
    # Пинги
    for period in range(1, len(inter_list) + 1):
        col = 'avg_min_ping_{}'.format(period)
        dataset.loc[(dataset[col] < 0) |
                    (dataset[col].isnull()), col] = dataset.loc[dataset[col] >= 0][col].median()
    # Сессии и прочее
    dataset.fillna(0, inplace=True)
    dataset.to_csv('{}dataset_{}.csv'.format(dataset_path, dataset_type), sep=';', index=False)

    print('Dataset is successfully prepared and saved to {}, run time (dealing with bad values): {}'. \
          format(dataset_path, time_format(time.time() - start_t)))


def get_best_feature(data, n_feature, random_state=21, sampling_strategy=0.4):
    """
    List of best feature for model
    """
    x = StandardScaler().fit_transform(data.drop(['user_id', 'is_churned'], axis=1))
    y = data['is_churned']

    x_bal, y_bal = ADASYN(random_state=random_state,
                          sampling_strategy=sampling_strategy).fit_sample(x, y)

    clf = xgb.XGBClassifier(n_estimators=100,
                            learning_rate=0.1,
                            random_state=random_state,
                            n_jobs=-1)
    clf.fit(x_bal, y_bal)
    best_feature = clf.feature_importances_
    return best_feature[0][:n_feature]


def get_data_for_prediction(train, test, best_feature, sampling_strategy=0.4, random_state=21):
    """
    Get scaled and balanced train (x, y) and test (y) data
    """
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train.drop(['user_id', 'is_churned'], axis=1)[best_feature])
    y_train = train['is_churned']

    x_train_bal, y_train_bal = ADASYN(random_state=random_state,
                                      sampling_strategy=sampling_strategy).fit_sample(x_train, y_train)

    x_test = scaler.transform(test.drop(['user_id'], axis=1)[best_feature])
    return x_train_bal, y_train_bal, x_test
