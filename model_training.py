import xgboost as xgb
import pickle

from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, roc_auc_score

import warnings
warnings.filterwarnings("ignore")


def evaluation(y_true, y_pred, y_prob):
    """
    All score models
    """
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    ll = log_loss(y_true=y_true, y_pred=y_prob)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_prob)
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1: {}'.format(f1))
    print('Log Loss: {}'.format(ll))
    print('ROC AUC: {}'.format(roc_auc))
    return precision, recall, f1, ll, roc_auc


def model_fit_predict(clf, x_train, y_train, x_test, y_test):
    """
    Fit model and predict target
    """
    clf.fit(x_train, y_train)
    predict_proba_test = clf.predict_proba(x_test)
    predict_test = clf.predict(x_test)
    precision_test, recall_test, f1_test, log_loss_test, roc_auc_test = \
        evaluation(y_test, predict_test, predict_proba_test[:, 1])
    return clf


def get_fitted_model(x_train, y_train, n_estimators=500, learning_rate=0.1, random_state=21, n_jobs=-1, max_depth=5):
    """
    Get fitted model XGBoost
    """
    clf = xgb.XGBClassifier(n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            random_state=random_state,
                            n_jobs=n_jobs,
                            max_depth=max_depth)
    return clf.fit(x_train, y_train)


def save_model(clf, model_path='models/', name='model_xgb_final'):
    with open(f'{model_path}{name}.pcl', 'wb') as f:
        pickle.dump(clf, f)
