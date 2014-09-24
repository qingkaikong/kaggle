#!/usr/bin/env python
#-*- coding:utf-8 -*-

""" Practice ground for trying examples of sklearn's Classification """
import numpy as np
from sklearn import svm
from sklearn import decomposition
from sklearn import cross_validation as cv
from sklearn import grid_search as gs
from sklearn import metrics

'''
This is a nice example I download from kaggle
'''


def load_data(filename):
    """ Load CSV into numpy array """
    return np.genfromtxt(open(filename,'rb'), delimiter=',')

def decomposition_pca(train, test):
    """ Linear dimensionality reduction """
    pca = decomposition.PCA(n_components=12, whiten=True)
    train_pca = pca.fit_transform(train)
    test_pca = pca.transform(test)
    return train_pca, test_pca

def split_data(X_data, y_data):
    """ Split the dataset in train and test """
    return cv.train_test_split(X_data, y_data, test_size=0.1, random_state=0)

def grid_search(y_data):
    c_range = 10.0 ** np.arange(6.5,7.5,.25)
    gamma_range = 10.0 ** np.arange(-1.5,0.5,.25)
    params = [{'kernel': ['rbf'], 'gamma': gamma_range, 'C': c_range}]

    cvk = cv.StratifiedKFold(y_data, k=5)
    return gs.GridSearchCV(svm.SVC(), params, cv=cvk)

def train(features, result):
    """ Use features and result to train Support Vector Machine"""
    clf = grid_search(result)
    clf.fit(features, result)
    return clf

def predict(clf, features):
    """ Predict labels from trained CLF """
    return clf.predict(features).astype(np.int)

def show_score(clf, X_test, y_test):
    """ Scores are computed on the test set """
    y_pred = predict(clf, X_test)
    print metrics.classification_report(y_test.astype(np.int), y_pred)

def write_data(filename, data):
    """ Write numpy array into CSV """
    np.savetxt(filename, data, fmt='%d')

def main():
    """ Analysis and predicts data """
    X_data = load_data('data/train.csv')
    y_data = load_data('data/trainLabels.csv')
    test_data = load_data('data/test.csv')

    X_data, test_data = decomposition_pca(X_data, test_data)

    X_train, X_test, y_train, y_test = split_data(X_data, y_data)
    clf = train(X_train, y_train)
    #show_score(clf, X_test, y_test)
    print clf.grid_scores_

    #submission = predict(clf, test_data)
    #write_data('submission.csv', submission)

if __name__ == "__main__":
    main()