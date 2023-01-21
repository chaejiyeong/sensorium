# coding: utf-8
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import sys

cancer = load_breast_cancer()

f = open('계량기&ltype1 지운 형식.csv')  # # train, test 둘 다
f2 = open('/home/jy/Desktop/계량기&ltype1 지운 형식.csv')  # 1
data = csv.reader(f)
data2 = csv.reader(f2)
header = next(data)
header2 = next(data2)

cancer.data = np.array([])
cancer.data2 = np.array([])
cancer.extraData = np.array([])
cancer.extraData2 = np.array([])
cancer.target = np.array([])
cancer.target2 = np.array([])

cancer.feature_names = np.array(header[6:])
cancer.feature_names2 = np.array(header2[6:])  # OK
cancer.feature_names = np.append(
    cancer.feature_names, header[1:4])
cancer.feature_names2 = np.append(
    cancer.feature_names2, header[1:4])
cancer.target_names = np.array([0, 1])
cancer.target_names2 = np.array([0, 1])  # OK

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

for row in data2:  # 2
    # print(1)
    if len(cancer.data2) == 0:
        cancer.data2 = np.append(cancer.data2, np.array(row[6:]), axis=0)
        cancer.extraData2 = np.append(
            cancer.extraData2, np.array(row[1:4]), axis=0)
    else:
        cancer.data2 = np.vstack([cancer.data2, np.array(row[6:])])
        cancer.extraData2 = np.vstack([cancer.extraData2, np.array(row[1:4])])
    if (row[4] == '0' or row[4] == '1'):
        cancer.target2 = np.append(cancer.target2, np.array(0))
    if (row[4] == '2' or row[4] == '3' or row[4] == '4'):
        cancer.target2 = np.append(cancer.target2, np.array(1))

cancer.data = np.concatenate([cancer.data, cancer.extraData], axis=1)
cancer.data2 = np.concatenate([cancer.data2, cancer.extraData2], axis=1)

f.close
f2.close

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

params = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(-3, 3, 7),
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
}
acc = []
# 훈련/테스트 세트로 나누기 (random_state 바꿔서)
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=i)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        cancer.data2, cancer.target2, random_state=i, test_size=149.25/410)
    logreg = LogisticRegression()
    gslogreg = GridSearchCV(logreg, params,
                            scoring="f1_macro", verbose=1, cv=3, refit=True)
    gslogreg.fit(X_train, y_train)
    # Logreg = LogisticRegression(C=0.01, penalty='l1',
    #                            solver='liblinear')
    # Logreg.fit(X_train, y_train)
    # y_hat = Logreg.predict(X_test2)
    # matrix = metrics.confusion_matrix(y_test, y_hat)
    scores_df = pd.DataFrame(gslogreg.cv_results_)
    scores_df = scores_df[["params", "mean_test_score", "rank_test_score"]]
    # pd.set_option('display.max_colwidth', -1)
    acc.append(i)
    acc.append(scores_df)
"""
    parameter_array.append(gslogreg.best_params_)
    estimator = gslogreg.best_estimator_
    y_hat = estimator.predict(X_test2)
    accuracy_array.append(round(accuracy_score(y_test2, y_hat), 4))

    
    precision_micro += precision_score(y_test2, y_hat, average='micro')
    precision_macro += precision_score(y_test2, y_hat, average='macro')
    precision_weighted += precision_score(y_test2, y_hat, average='weighted')
    precision_binary += precision_score(y_test2,
                                        y_hat, average='binary', pos_label=0)
    recall_micro += recall_score(y_test2, y_hat, average='micro')
    recall_macro += recall_score(y_test2, y_hat, average='macro')
    recall_weighted += recall_score(y_test2, y_hat, average='weighted')
    recall_binary += recall_score(y_test2, y_hat,
                                  average='binary', pos_label=0)
    f1_micro += f1_score(y_test2, y_hat, average='micro')
    f1_macro += f1_score(y_test2, y_hat, average='macro')
    f1_weighted += f1_score(y_test2, y_hat, average='weighted')
    f1_binary += f1_score(y_test2, y_hat, average='binary', pos_label=0)
    accuracy_True += accuracy_score(y_test2, y_hat, normalize=True)
    accuracy_False += accuracy_score(y_test2, y_hat, normalize=False)
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

# y_hat = forest.predict(X_test)
# matrix = metrics.confusion_matrix(y_test, y_hat)
# print(matrix)
# 얘로 최고 parameter 뽑고 나서 -> 해당 matrix 뽑기
scores_df = pd.DataFrame(gslogreg.cv_results_)
scores_df = scores_df[["params", "mean_test_score",
                       "rank_test_score"]]
print(scores_df)
print("최적의 하이퍼 파라미터: ", gslogreg.best_params_)
print("최고의 예측 정확도: ", gslogreg.best_score_)

"""
sys.stdout = open('LG_2_result.txt', 'w')
print(acc)
sys.stdout.close
