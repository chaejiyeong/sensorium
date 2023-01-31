# coding: utf-8
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

cancer = load_breast_cancer()
f = open('/home/jy/Desktop/엑셀 형식.csv')
data = csv.reader(f)
header = next(data)
cancer.data = np.array([])
cancer.target = np.array([])

cancer.feature_names = np.array(header[7:])  # OK
cancer.target_names = np.array([0, 1, 2, 3, 4])  # OK

for row in data:
    if len(cancer.data) == 0:
        manufac = list(map(float, row[7:]))
        cancer.data = np.append(cancer.data, np.array(manufac[0:]), axis=0)
    else:
        manufac = list(map(float, row[7:]))
        cancer.data = np.vstack([cancer.data, np.array(manufac[0:])])  # OK
    cancer.target = np.append(cancer.target, np.array(row[4]))  # OK
f.close

# 훈련/테스트 세트로 나누기
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)
"""
params = {'n_neighbors': [5, 15, 30, 100], 'weights':  ['uniform', 'distance'],
          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
knn = KNeighborsClassifier()
gsknn = GridSearchCV(knn, params, cv=2,
                     scoring="accuracy", verbose=1)
gsknn.fit(X_train, y_train)
# y_hat = forest.predict(X_test)
# matrix = metrics.confusion_matrix(y_test, y_hat)
# print(matrix)
# 얘로 최고 parameter 뽑고 나서 -> 해당 matrix 뽑기
scores_df = pd.DataFrame(gsknn.cv_results_)
scores_df = scores_df[["params", "mean_test_score",
                       "rank_test_score"]]
pd.set_option('display.max_colwidth', -1)
print(scores_df)
print("최적의 하이퍼 파라미터: ", gsknn.best_params_)
print("최고의 예측 정확도: ", gsknn.best_score_)
# print("훈련 세트 정확도 : {:.3f}".format(forest.score(X_train, y_train)))
# print("테스트 세트 정확도 : {:.3f}".format(forest.score(X_test, y_test)))


"""

knn = KNeighborsClassifier(
    algorithm='auto', n_neighbors=100, weights='uniform')
knn.fit(X_train, y_train)
y_hat = knn.predict(X_test)
matrix = metrics.confusion_matrix(y_test, y_hat)
print(matrix)
# 최고 parameter -> 해당 matrix 뽑기
print("훈련 세트 정확도 : {:.3f}".format(knn.score(X_train, y_train)))
print("테스트 세트 정확도 : {:.3f}".format(knn.score(X_test, y_test)))
