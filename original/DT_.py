# coding: utf-8
from sklearn.tree import export_graphviz  # DT는 덜함 !
import graphviz
from IPython import display
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt
import numpy as np
import math

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

tree = DecisionTreeClassifier(random_state=0)

# 훈련 데이터로 학습 시키기
tree.fit(X_train, y_train)
print("훈련 세트 점수: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(tree.score(X_test, y_test)))
# 훈련 세트 점수: 1.000
# 테스트 세트 점수: 0.937

# 사전 가지치기 추가
# coding: utf-8
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(max_depth=4, random_state=0)

# 훈련 데이터로 학습 시키기
tree.fit(X_train, y_train)
print("훈련 세트 점수: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(tree.score(X_test, y_test)))
# 훈련 세트 점수: 0.988
# 테스트 세트 점수: 0.951

# 트리 모듈의 export_graphviz 함수를 이용해 트리를 시각화 할 수 있다.
# 이 함수는 그래프 저장용 텍스트 파일 포맷인 .dot 파일을 만든다.
# 각 노드에서 다수인 클래스를 색으로 나타내기 위해 옵션(filled=True)을 주고 적절히 레이블 되도록 클래스 이름과 특성 이름을 매개변수로 전달한다.
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
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
# 훈련 데이터로 학습 시키기
tree.fit(X_train, y_train)
export_graphviz(tree, out_file="tree.dot", feature_names=cancer.feature_names,
                class_names=["cancer", "not cancer"], filled=True, impurity=False)

with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))

# ################################################################################################3특성 중요도 확인
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

# 훈련/테스트 세트로 나누기
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(max_depth=4, random_state=0)

# 훈련 데이터로 학습 시키기
tree.fit(X_train, y_train)

# 특성 중요도
print("특성 중요도 : \n{}".format(tree.feature_importances_))

"""
(아직 안고침)
################################################################################특성 중요도 수평 막대 그래프

# coding: utf-8

from sklearn.datasets import load_breast_cancer

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

import numpy as np



cancer = load_breast_cancer()



# 훈련/테스트 세트로 나누기

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target, random_state=42)



tree = DecisionTreeClassifier(max_depth=4,random_state=0)

# 훈련 데이터로 학습 시키기

tree.fit(X_train,y_train)



# 특성 중요도

print("특성 중요도 : \n{}".format(tree.feature_importances_))



# 특성 중요도 시각화 하기

def plot_feature_importances_cancer(model):

    n_features = cancer.data.shape[1]

    plt.barh(range(n_features),model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features),cancer.feature_names)

    plt.xlabel("attr importances")

    plt.ylabel("attr")

    plt.ylim(-1,n_features)

plot_feature_importances_cancer(tree)

plt.show()

################################################################특성과 클래스(0 1 2 3 4)과의 관계

# coding: utf-8

from sklearn.datasets import load_breast_cancer

from sklearn.tree import DecisionTreeClassifier

import mglearn

from IPython.core.display import display



cancer = load_breast_cancer()



# 훈련/테스트 세트로 나누기

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)



tree = DecisionTreeClassifier(max_depth=4, random_state=0)

# 훈련 데이터로 학습 시키기

tree.fit(X_train, y_train)



tree = mglearn.plots.plot_tree_not_monotone()

display(tree)
"""
