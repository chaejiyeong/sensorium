from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from sklearn.datasets import load_iris
import csv
from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ANN 모델 => DNN 모델


class Model(nn.Module):
    # in_features : (feature 10Hz ~ 5120Hz) or (feature 10Hz ~ 5120Hz and ptype, pdtype, stype)
    # out_features : (0, 1, 2, 3, 4) or (0, 1)
    def __init__(self, in_features=515, h1=100, h2=50, h3=5, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        # self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h3, out_features)
        # self.batch_norm1 = nn.BatchNorm1d(100)
        # self.batch_norm2 = nn.BatchNorm1d(50)
        # self.batch_norm3 = nn.BatchNorm1d(5)

# 입력 -> 은닉층 1 -> 은닉층 2 -> 은닉층 3 -> 출력
# 순전파
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.batch_norm1(x)
        x = F.relu(self.fc2(x))
        # x = self.batch_norm2(x)
        x = F.relu(self.fc3(x))
        # x = self.batch_norm3(x)
        # x = F.relu(self.fc4(x))
        x = self.out(x)
        return x  # model 객체 생성


torch.manual_seed(1)
model = Model()

# 데이터
cancer = load_breast_cancer()

f = open('/home/jy/Desktop/엑셀 형식 - 계량기만.csv')  # 1
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

cancer.feature_names = np.array(header[7:])
cancer.feature_names2 = np.array(header2[7:])  # OK
cancer.feature_names = np.append(
    cancer.feature_names, header[1:4])
cancer.feature_names2 = np.append(
    cancer.feature_names2, header[1:4])
cancer.target_names = np.array([0, 1])
cancer.target_names2 = np.array([0, 1])  # OK

for row in data:  # 2
    # print(1)
    if len(cancer.data) == 0:
        cancer.data = np.append(cancer.data, np.array(row[7:]), axis=0)
        cancer.extraData = np.append(
            cancer.extraData, np.array(row[1:4]), axis=0)
    else:
        cancer.data = np.vstack([cancer.data, np.array(row[7:])])
        cancer.extraData = np.vstack([cancer.extraData, np.array(row[1:4])])
    if (row[4] == '0' or row[4] == '1'):
        cancer.target = np.append(cancer.target, np.array(0))
    if (row[4] == '2' or row[4] == '3' or row[4] == '4'):
        cancer.target = np.append(cancer.target, np.array(1))

for row in data2:  # 2
    # print(1)
    if len(cancer.data2) == 0:
        cancer.data2 = np.append(cancer.data2, np.array(row[7:]), axis=0)
        cancer.extraData2 = np.append(
            cancer.extraData2, np.array(row[1:4]), axis=0)
    else:
        cancer.data2 = np.vstack([cancer.data2, np.array(row[7:])])
        cancer.extraData2 = np.vstack([cancer.extraData2, np.array(row[1:4])])
    if (row[4] == '0' or row[4] == '1'):
        cancer.target2 = np.append(cancer.target2, np.array(0))
    if (row[4] == '2' or row[4] == '3' or row[4] == '4'):
        cancer.target2 = np.append(cancer.target2, np.array(1))

cancer.data = np.concatenate([cancer.data, cancer.extraData], axis=1)
cancer.data2 = np.concatenate([cancer.data2, cancer.extraData2], axis=1)


f.close
f2.close

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
for j in range(100):
    model = Model()
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, random_state=j)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        cancer.data2, cancer.target2, random_state=j, test_size=149.25/410)

    X_train = np.vstack(X_train).astype(np.float)
    X_test = np.vstack(X_test).astype(np.float)
    X_train2 = np.vstack(X_train2).astype(np.float)
    X_test2 = np.vstack(X_test2).astype(np.float)

    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)
    X_train2 = torch.from_numpy(X_train2)
    X_test2 = torch.from_numpy(X_test2)
    y_train2 = torch.from_numpy(y_train2)
    y_test2 = torch.from_numpy(y_test2)

    # 위에서 설명한 데이터 텐서화
    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)
    X_train2 = torch.Tensor(X_train2)
    X_test2 = torch.Tensor(X_test2)
    y_train2 = torch.Tensor(y_train2)
    y_test2 = torch.Tensor(y_test2)

    X_train = X_train.float()
    X_test = X_test.float()
    y_train = y_train.long()
    y_test = y_test.float()
    X_train2 = X_train2.float()
    X_test2 = X_test2.float()
    y_train2 = y_train2.long()
    y_test2 = y_test2.float()

    # 손실함수 정의
    criterion = torch.nn.CrossEntropyLoss()

    # 최적화 함수 정의
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 1000  # 훈련 횟수 100번
    losses = []  # loss를 담을 리스트, 시각화 하기 위함

    for i in range(epochs):
        model.train()
        y_pred = model(X_train)

        loss = criterion(y_pred, y_train)
        losses.append(loss)

        # if i % 10 == 0:
        #    print(f'epoch {i}, loss is {loss}')

        # 역전파 수행
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    correct = 0
    Y_test_list = []
    Y_hat_list = []
    # test는 test2로
    with torch.no_grad():
        for i, data in enumerate(X_test2):
            y_hat = model.forward(data)
            Y_hat_list.append(y_hat.argmax().item())
            Y_test_list.append(y_test2[i])
            # print(f'{i+1}.) {str(y_hat.argmax().item())} {y_test[i]}')
            if y_hat.argmax().item() == y_test2[i]:
                correct += 1
    # matrix = metrics.confusion_matrix(Y_test_list, Y_hat_list)
    # print(matrix)
    precision_micro += precision_score(Y_test_list,
                                       Y_hat_list, average='micro')
    precision_macro += precision_score(Y_test_list,
                                       Y_hat_list, average='macro')
    precision_weighted += precision_score(Y_test_list,
                                          Y_hat_list, average='weighted')
    precision_binary += precision_score(Y_test_list,
                                        Y_hat_list, average='binary', pos_label=0)
    recall_micro += recall_score(Y_test_list, Y_hat_list, average='micro')
    recall_macro += recall_score(Y_test_list, Y_hat_list, average='macro')
    recall_weighted += recall_score(Y_test_list,
                                    Y_hat_list, average='weighted')
    recall_binary += recall_score(Y_test_list,
                                  Y_hat_list, average='binary', pos_label=0)
    f1_micro += f1_score(Y_test_list, Y_hat_list, average='micro')
    f1_macro += f1_score(Y_test_list, Y_hat_list, average='macro')
    f1_weighted += f1_score(Y_test_list, Y_hat_list, average='weighted')
    f1_binary += f1_score(Y_test_list, Y_hat_list,
                          average='binary', pos_label=0)
    accuracy_True += accuracy_score(Y_test_list, Y_hat_list, normalize=True)
    accuracy_False += accuracy_score(Y_test_list, Y_hat_list, normalize=False)

    # accuracy = correct / 597  # 엑셀 형식인 경우
    # value_sum += accuracy
# print(value_sum / 100)
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

# DNN은 하이퍼파라미터 보팅 안하고 하나씩 해보기
