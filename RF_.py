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
    cancer.data, cancer.target, random_state=10)

"""
params = {'max_depth': [5, 15, 30], 'max_features':  [0.1, 0.5, 0.8], 'min_samples_leaf': [8, 18],
          'n_estimators': [100, 200, 300]}
forest = RandomForestClassifier()
gsforest = GridSearchCV(forest, params, cv=2,
                        scoring="accuracy", verbose=1)
gsforest.fit(X_train, y_train)
# y_hat = forest.predict(X_test)
# matrix = metrics.confusion_matrix(y_test, y_hat)
# print(matrix)
# 얘로 최고 parameter 뽑고 나서 -> 해당 matrix 뽑기
scores_df = pd.DataFrame(gsforest.cv_results_)
scores_df = scores_df[["params", "mean_test_score",
                       "rank_test_score"]]
pd.set_option('display.max_colwidth', -1)
print(scores_df)
print("최적의 하이퍼 파라미터: ", gsforest.best_params_)
print("최고의 예측 정확도: ", gsforest.best_score_)
# print("훈련 세트 정확도 : {:.3f}".format(forest.score(X_train, y_train)))
# print("테스트 세트 정확도 : {:.3f}".format(forest.score(X_test, y_test)))
"""
forest = RandomForestClassifier(
    max_depth=5, max_features=0.8, min_samples_leaf=8, n_estimators=300)
forest.fit(X_train, y_train)
y_hat = forest.predict(X_test)
matrix = metrics.confusion_matrix(y_test, y_hat)
print(matrix)
# 최고 parameter -> 해당 matrix 뽑기
print("훈련 세트 정확도 : {:.3f}".format(forest.score(X_train, y_train)))
print("테스트 세트 정확도 : {:.3f}".format(forest.score(X_test, y_test)))


"""
# 특성 중요도
print("특성 중요도 : \n{}".format(forest.feature_importances_))

# 특성 중요도 시각화 하기


def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("attr importances")
    plt.ylabel("attr")
    plt.ylim(-1, n_features)


plt.show()


plot_feature_importances_cancer(forest)

"""
