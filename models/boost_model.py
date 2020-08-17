import xgboost as xgb
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def run_xgb(x, y) -> xgb.Booster:
    """
    Run xgboost for binary classification
    :param x:
    :param y:
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    param = {'max_depth': 8, 'eta': 0.1, 'objective': 'binary:logistic', 'gamma': '0.3'}
    param['nthread'] = 8
    param['eval_metric'] = ['auc']

    validation_list = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(param, dtrain, num_boost_round=200, evals=validation_list)

    xgb.plot_importance(model)
    # plt.show()
    return model


if __name__ == '__main__':
    pass
