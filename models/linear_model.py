from sklearn import linear_model
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
import numpy as np


def run_linear_model(x_train, y_train, x_test, y_test):
    """
    A simple Linear Ridge Regression Model
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    # print("x train shape: {}, x test shape: {}".format(x_train.shape, x_test.shape))
    reg_model = linear_model.Ridge(alpha=0.5)

    reg_model.fit(x_train, y_train)
    score = reg_model.score(x_test, y_test)
    coefs = reg_model.coef_
    # print("Accuracy: {}".format(score))
    # print("Coefficients: {}".format(coefs))
    return score, coefs


def run_ktimes(x, y, n=100):
    """
    Run the Linear Ridge Regression model k times to get the mean from all the runs
    :param x:
    :param y:
    :param n:
    :return:
    """
    scores = []
    coefs = []
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)
        score, coef = run_linear_model(x_train, y_train, x_test, y_test)
        scores.append(score)
        coefs.append(coef)
    print(np.mean(scores))
    print(np.mean(coefs, axis=0))


def run_kfold(x, y):
    """
    Run the model with stratified K fold
    :param x:
    :param y:
    :return:
    """
    pass


if __name__ == '__main__':
    pass
