# coding: utf-8
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

cancer = load_breast_cancer()

f = open('/home/jy/Desktop/계량기&ltype1 지운 형식.csv')  # 1
data = csv.reader(f)
header = next(data)

cancer.data = np.array([])
cancer.extraData = np.array([])
cancer.target = np.array([])

cancer.feature_names = np.array(header[6:])
cancer.feature_names = np.append(
    cancer.feature_names, header[1:4])
cancer.target_names = np.array([0, 1])

for row in data:  # 2
    # print(1)
    if len(cancer.data) == 0:
        cancer.data = np.append(cancer.data, np.array(row[6:]), axis=0)
        cancer.extraData = np.append(
            cancer.extraData, np.array(row[1:4]), axis=0)
    else:
        cancer.data = np.vstack([cancer.data, np.array(row[6:])])
        cancer.extraData = np.vstack([cancer.extraData, np.array(row[1:4])])
    if (row[4] == '0' or row[4] == '1'):
        cancer.target = np.append(cancer.target, np.array(0))
    if (row[4] == '2' or row[4] == '3' or row[4] == '4'):
        cancer.target = np.append(cancer.target, np.array(1))

cancer.data = np.concatenate([cancer.data, cancer.extraData], axis=1)

f.close

# 훈련/테스트 세트로 나누기
precision_micro = 0
precision_macro = 0
precision_weighted = 0
precision_binary = 0
recall_micro = 0
recall_macro = 0
recall_weighted = 0
recall_binary = 0
f1_micro = 0
f1_macro = 0
f1_weighted = 0
f1_binary = 0
accuracy_True = 0
accuracy_False = 0

params = {'max_depth': [5, 15, 30],
          'n_estimators': [100, 200, 300]}
accuracy_array = []
parameter_array = []
# 훈련/테스트 세트로 나누기 (random_state 바꿔서)
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=i)
    forest = RandomForestClassifier()
    gsforest = GridSearchCV(forest, params, cv=3,
                            scoring="accuracy", verbose=1, refit=True)
    gsforest.fit(X_train, y_train)
    # forest = RandomForestClassifier(
    #    max_depth=30, max_features=0.5, min_samples_leaf=18, n_estimators=300)
    # forest.fit(X_train, y_train)
    # y_hat = forest.predict(X_test)
    # matrix
    # matrix = metrics.confusion_matrix(y_test2, y_hat)
    parameter_array.append(gsforest.best_params_)
    estimator = gsforest.best_estimator_
    y_hat = estimator.predict(X_test)
    accuracy_array.append(round(accuracy_score(y_test, y_hat), 4))

    """
    precision_micro += precision_score(y_test, y_hat, average='micro')
    precision_macro += precision_score(y_test, y_hat, average='macro')
    precision_weighted += precision_score(y_test, y_hat, average='weighted')
    precision_binary += precision_score(y_test,
                                        y_hat, average='binary', pos_label=0)
    recall_micro += recall_score(y_test, y_hat, average='micro')
    recall_macro += recall_score(y_test, y_hat, average='macro')
    recall_weighted += recall_score(y_test, y_hat, average='weighted')
    recall_binary += recall_score(y_test, y_hat,
                                  average='binary', pos_label=0)
    f1_micro += f1_score(y_test, y_hat, average='micro')
    f1_macro += f1_score(y_test, y_hat, average='macro')
    f1_weighted += f1_score(y_test, y_hat, average='weighted')
    f1_binary += f1_score(y_test, y_hat, average='binary', pos_label=0)
    accuracy_True += accuracy_score(y_test, y_hat, normalize=True)
    accuracy_False += accuracy_score(y_test, y_hat, normalize=False)
    # value_sum += forest.score(X_test, y_test)
    # print(matrix)
    # 최고 parameter -> 해당 matrix 뽑기
    # print("훈련 세트 정확도 : {:.3f}".format(Logreg.score(X_train, y_train)))
    # print("테스트 세트 정확도 : {:.3f}".format(Logreg.score(X_test, y_test)))
print(precision_micro / 100)
print(precision_macro / 100)
print(precision_weighted / 100)
print(precision_binary / 100)
print(recall_micro / 100)
print(recall_macro / 100)
print(recall_weighted / 100)
print(recall_binary / 100)
print(f1_micro / 100)
print(f1_macro / 100)
print(f1_weighted / 100)
print(f1_binary / 100)
print(accuracy_True / 100)
print(accuracy_False / 100)
"""
print(accuracy_array)
print(parameter_array)
