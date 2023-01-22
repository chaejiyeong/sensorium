# coding: utf-8
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import pandas as pd

# cancer = load_breast_cancer()
# cancer.data = []
# cancer.target = []
# cancer.feature_names = []
# cancer.target_names = []

# f = open('/home/jy/Desktop/엑셀 형식.csv')
# data = csv.reader(f)
# header = next(data)
# cancer.feature_names.append(header[6:])
# cancer.target_names.append([0, 1, 2, 3, 4])
# for row in data:
#    cancer.data.append(row[6:])
#    cancer.target.append(row[4])
# f.close
""" # 파라미터 많이
cancer = load_breast_cancer()

f = open('/home/jy/Desktop/엑셀 형식.csv')
data = csv.reader(f)
header = next(data)
cancer.data = np.array([])
cancer.target = np.array([])

cancer.feature_names = np.array(header[6:])  # OK
cancer.target_names = np.array([0, 1, 2, 3, 4])  # OK

for row in data:
    if len(cancer.data) == 0:
        cancer.data = np.append(cancer.data, np.array(row[6:]), axis=0)
    else:
        cancer.data = np.vstack([cancer.data, np.array(row[6:])])  # OK
    cancer.target = np.append(cancer.target, np.array(row[4]))  # OK
f.close

# print(cancer.data)
# print(cancer.target)
# print(type(cancer))

# 훈련/테스트 세트로 나누기
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
# print(X_train, X_test, y_train, y_test)

params = {'activation': ['identity', 'logistic', 'relu', 'tanh'],
          'learning_rate_init':  [0.001, 0.1], 'max_iter': [200, 500]}
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

mlp = MLPClassifier(random_state=42)
gsMLP = GridSearchCV(mlp, params, cv=2,
                     scoring="accuracy", verbose=1)
gsMLP.fit(X_train, y_train)
# y_hat = forest.predict(X_test)
# matrix = metrics.confusion_matrix(y_test, y_hat)
# print(matrix)
# 얘로 최고 parameter 뽑고 나서 -> 해당 matrix 뽑기
scores_df = pd.DataFrame(gsMLP.cv_results_)
scores_df = scores_df[["params", "mean_test_score",
                       "rank_test_score"]]
pd.set_option('display.max_colwidth', -1)
print(scores_df)
print("최적의 하이퍼 파라미터: ", gsMLP.best_params_)
print("최고의 예측 정확도: ", gsMLP.best_score_)
# print("훈련 세트 정확도 : {:.3f}".format(forest.score(X_train, y_train)))
# print("테스트 세트 정확도 : {:.3f}".format(forest.score(X_test, y_test)))

# mlp = MLPClassifier(random_state=42)
# mlp.fit(X_train, y_train)
# print("기본_훈련 세트 정확도 : {:.2f}".format(mlp.score(X_train, y_train)))  # 0.92
# print("기본_테스트 세트 정확도 : {:.2f}".format(mlp.score(X_test, y_test)))  # 0.38



######################################################################
# 데이터 스케일 맞춘 후 + 0hz일때 제외
# coding: utf-8
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

# print(cancer.data.shape)  # (2385, 513) 나와야함

# 훈련/테스트 세트로 나누기
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)  # 각각 2차원 배열
# print(X_train, X_test, y_train, y_test)


mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

# 훈련 세트 각 특성의 평균

mean_on_train = X_train.mean(axis=0)

# 훈련 세트 각 특성의 표준 편차

std_on_train = X_train.std(axis=0)


# 표준 정규분포

X_train_scaled = (X_train-mean_on_train)/std_on_train


X_test_scaled = (X_test-mean_on_train)/std_on_train

mlp = MLPClassifier(random_state=0)

mlp.fit(X_train_scaled, y_train)


print("표준화_훈련 세트 정확도 : {:.3f}".format(mlp.score(X_train_scaled, y_train)))

print("표준화_테스트 세트 정확도 : {:.3f}".format(mlp.score(X_test_scaled, y_test)))



# max_iter 설정 ############################################################################ 42.5% 성능 보임

# coding: utf-8


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

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)


# 훈련 세트 각 특성의 평균

mean_on_train = X_train.mean(axis=0)

# 훈련 세트 각 특성의 표준 편차

std_on_train = X_train.std(axis=0)


# 표준 정규분포

X_train_scaled = (X_train-mean_on_train)/std_on_train

X_test_scaled = (X_test-mean_on_train)/std_on_train


# 이 경고는 모델을 학습시키는 adam 알고리즘에 관련한 것으로 반복 횟수를 늘려주어야 한다.

params = {'activation': ['identity', 'logistic', 'relu', 'tanh'],
          'learning_rate_init':  [0.001, 0.1], 'max_iter': [200, 500]}
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

mlp = MLPClassifier(random_state=42)
gsMLP = GridSearchCV(mlp, params, cv=2,
                     scoring="accuracy", verbose=1)
gsMLP.fit(X_train_scaled, y_train)
# y_hat = forest.predict(X_test)
# matrix = metrics.confusion_matrix(y_test, y_hat)
# print(matrix)
# 얘로 최고 parameter 뽑고 나서 -> 해당 matrix 뽑기
scores_df = pd.DataFrame(gsMLP.cv_results_)
scores_df = scores_df[["params", "mean_test_score",
                       "rank_test_score"]]
print(scores_df)
print("최적의 하이퍼 파라미터: ", gsMLP.best_params_)
print("최고의 예측 정확도: ", gsMLP.best_score_)












# mlp.fit(X_train_scaled, y_train)
# y_hat = mlp.predict(X_test)
# matrix = metrics.confusion_matrix(y_test, y_hat)
# print(matrix)

# print("훈련 세트 정확도 : {:.3f}".format(mlp.score(X_train_scaled, y_train)))

# print("테스트 세트 정확도 : {:.3f}".format(mlp.score(X_test_scaled, y_test)))
"""

# MATRIX 뽑기 (1)
cancer = load_breast_cancer()

f = open('/home/jy/Desktop/엑셀 형식.csv')
data = csv.reader(f)
header = next(data)
cancer.data = np.array([])
cancer.target = np.array([])

cancer.feature_names = np.array(header[6:])  # OK
cancer.target_names = np.array([0, 1, 2, 3, 4])  # OK

for row in data:
    if len(cancer.data) == 0:
        cancer.data = np.append(cancer.data, np.array(row[6:]), axis=0)
    else:
        cancer.data = np.vstack([cancer.data, np.array(row[6:])])  # OK
    cancer.target = np.append(cancer.target, np.array(row[4]))  # OK
f.close

# print(cancer.data)
# print(cancer.target)
# print(type(cancer))

# 훈련/테스트 세트로 나누기
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
# print(X_train, X_test, y_train, y_test)

mlp = MLPClassifier(random_state=42, activation='relu',
                    learning_rate_init=0.1, max_iter=200)
mlp.fit(X_train, y_train)
y_hat = mlp.predict(X_test)
matrix = metrics.confusion_matrix(y_test, y_hat)
print(matrix)
# 최고 parameter -> 해당 matrix 뽑기
print("훈련 세트 정확도 : {:.3f}".format(mlp.score(X_train, y_train)))
print("테스트 세트 정확도 : {:.3f}".format(mlp.score(X_test, y_test)))
