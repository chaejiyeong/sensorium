
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import pandas as pd

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

# 훈련/테스트 세트로 나누기
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)


"""
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("훈련 세트 정확도 : {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 정확도 : {:.3f}".format(gbrt.score(X_test, y_test)))
# 훈련 세트 정확도 : 1.000
# 테스트 세트 정확도 : 0.958

# 훈련 세트의 정확도가 100%이므로 과대적합되었다.
# 과대적합을 막기위해 사전 가지치기를 한다.
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)
print("훈련 세트 정확도 : {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 정확도 : {:.3f}".format(gbrt.score(X_test, y_test)))
# 훈련 세트 정확도 : 0.991
# 테스트 세트 정확도 : 0.972


# 과대적합을 막기위해 학습률을 낮춘다
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gbrt.fit(X_train, y_train)
print("훈련 세트 정확도 : {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 정확도 : {:.3f}".format(gbrt.score(X_test, y_test)))
# 훈련 세트 정확도 : 0.988
# 테스트 세트 정확도 : 0.965
"""

"""
params = {
    "learning_rate": [0.01, 0.1, 0.5],
    "max_depth": [3, 5, 8],
    "max_features": ["log2", "sqrt"],
    "criterion": ["friedman_mse",  "squared_error"]
}
gbrt = GradientBoostingClassifier()
gsgbrt = GridSearchCV(gbrt, params,
                      scoring="accuracy", verbose=1, cv=2)
gsgbrt.fit(X_train, y_train)
# y_hat = forest.predict(X_test)
# matrix = metrics.confusion_matrix(y_test, y_hat)
# print(matrix)
# 얘로 최고 parameter 뽑고 나서 -> 해당 matrix 뽑기
scores_df = pd.DataFrame(gsgbrt.cv_results_)
scores_df = scores_df[["params", "mean_test_score",
                       "rank_test_score"]]
pd.set_option('display.max_colwidth', -1)
print(scores_df)
print("최적의 하이퍼 파라미터: ", gsgbrt.best_params_)
print("최고의 예측 정확도: ", gsgbrt.best_score_)
"""


gbrt = GradientBoostingClassifier(
    criterion='friedman_mse', learning_rate=0.01, max_depth=8, max_features='log2')
gbrt.fit(X_train, y_train)
y_hat = gbrt.predict(X_test)
matrix = metrics.confusion_matrix(y_test, y_hat)
print(matrix)
# 최고 parameter -> 해당 matrix 뽑기
print("훈련 세트 정확도 : {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 정확도 : {:.3f}".format(gbrt.score(X_test, y_test)))


""" (아직 안고침)
# 특성 중요도
# coding: utf-8


cancer = load_breast_cancer()


# 훈련/테스트 세트로 나누기

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)


# 훈련 세트의 정확도가 100%이므로 과대적합되었다.

# 과대적합을 막기위해 사전 가지치기를 한다.

gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)

gbrt.fit(X_train, y_train)


print("훈련 세트 정확도 : {:.3f}".format(gbrt.score(X_train, y_train)))

print("테스트 세트 정확도 : {:.3f}".format(gbrt.score(X_test, y_test)))

# 훈련 세트 정확도 : 0.991

# 테스트 세트 정확도 : 0.972


# 특성 중요도

print("특성 중요도 : \n{}".format(gbrt.feature_importances_))


# 특성 중요도 시각화 하기

def plot_feature_importances_cancer(model):

    n_features = cancer.data.shape[1]

    plt.barh(range(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), cancer.feature_names)

    plt.xlabel("attr importances")

    plt.ylabel("attr")

    plt.ylim(-1, n_features)


plt.show()


plot_feature_importances_cancer(gbrt)
"""
