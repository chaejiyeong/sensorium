# 각기 다른 C값
# coding: utf-8
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
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
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

""" parameter
params = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(-3, 3, 7),
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
}
logreg = LogisticRegression()
gslogreg = GridSearchCV(logreg, params,
                        scoring="accuracy", verbose=1, cv=2)
gslogreg.fit(X_train, y_train)
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
Logreg = LogisticRegression(C=0.001, penalty='l1',
                            solver='liblinear')
Logreg.fit(X_train, y_train)
y_hat = Logreg.predict(X_test)
matrix = metrics.confusion_matrix(y_test, y_hat)
print(matrix)
# 최고 parameter -> 해당 matrix 뽑기
print("훈련 세트 정확도 : {:.3f}".format(Logreg.score(X_train, y_train)))
print("테스트 세트 정확도 : {:.3f}".format(Logreg.score(X_test, y_test)))


"""





logreg = LogisticRegression().fit(X_train, y_train)
print("훈련 세트 점수: {:.3f}".format(logreg.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(logreg.score(X_test, y_test)))

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("훈련 세트 점수: {:.3f}".format(logreg100.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("훈련 세트 점수: {:.3f}".format(logreg001.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(logreg001.score(X_test, y_test)))
"""


"""
#############################################################################각기 다른 C값을 사용하여 만든 로지스틱 회귀 계수값의 그래프
(아직 안고침)
# coding: utf-8
cancer = load_breast_cancer()



# 훈련/테스트 세트로 나누기

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target, random_state=42)



logreg = LogisticRegression().fit(X_train, y_train)

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)



# 유방암 데이터셋에 각기 다른 C값을 사용하여 만든 로지스텍 회의 계수값의 그래프

plt.plot(logreg.coef_.T,'o',label="C=1")

plt.plot(logreg100.coef_.T,'^',label="C=100")

plt.plot(logreg001.coef_.T,'v',label="C=0.01")

plt.xticks(range(cancer.data.shape[1]),cancer.feature_names, rotation=90)

plt.hlines(0,0,cancer.data.shape[1])

plt.ylim(-5,5)

plt.xlabel("attribute")

plt.ylabel("w size")

plt.legend()

########################################################################L1규제를 사용하여 각기 다른 C값을 적용한 로지스틱 회귀 모델의 계수
# coding: utf-8

from sklearn.datasets import load_breast_cancer

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression



cancer = load_breast_cancer()



# 훈련/테스트 세트로 나누기

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target, random_state=42)



# 유방함 데이터와 L1규제를 사용하여 각기 다른 C값을 적용한 로지스틱 회귀 모델의 계수

for C, marker in zip([0.001,1,100],['o','^','v']):

    lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train,y_train)

    print("C={:.3f}인  L1 로지스틱 회귀의 훈련 정확도: {:.2f}".format(C,lr_l1.score(X_train, y_train)))

    print("C={:.3f}인  L1 로지스틱 회귀의  테스트 정확도: {:.2f}".format(C,lr_l1.score(X_test, y_test)))

    plt.plot(lr_l1.coef_.T,marker,label="C={:.3f}".format(C))



plt.xticks(range(cancer.data.shape[1]),cancer.feature_names, rotation=90)

plt.hlines(0,0,cancer.data.shape[1])

plt.ylim(-5,5)

plt.xlabel("attribute")

plt.ylabel("w size")

plt.legend(loc=3)



# C=0.001인  L1 로지스틱 회귀의 훈련 정확도: 0.91

# C=0.001인  L1 로지스틱 회귀의  테스트 정확도: 0.92

# C=1.000인  L1 로지스틱 회귀의 훈련 정확도: 0.96

# C=1.000인  L1 로지스틱 회귀의  테스트 정확도: 0.96

# C=100.000인  L1 로지스틱 회귀의 훈련 정확도: 0.99

# C=100.000인  L1 로지스틱 회귀의  테스트 정확도: 0.98
"""
