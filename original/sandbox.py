from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from sklearn.datasets import load_iris
import csv
from sklearn.datasets import load_breast_cancer

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
f = open('/home/jy/Desktop/계량기&ltype1 지운 형식.csv')  # 1
data = csv.reader(f)
header = next(data)

cancer.data = np.array([], dtype=np.float16)
cancer.extraData = np.array([], dtype=np.float16)
cancer.target = np.array([], dtype=np.float16)
cancer.feature_names = np.array(header[7:])  # OK
cancer.feature_names = np.append(
    cancer.feature_names, header[1:4])
cancer.target_names = np.array([0, 1])   # OK

for row in data:
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

cancer.data = np.concatenate([cancer.data, cancer.extraData], axis=1)

f.close


X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=100)

X_train = np.vstack(X_train).astype(np.float)
X_test = np.vstack(X_test).astype(np.float)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

# 위에서 설명한 데이터 텐서화
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)

X_train = X_train.float()
X_test = X_test.float()
y_train = y_train.long()
y_test = y_test.float()

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

with torch.no_grad():

    for i, data in enumerate(X_test):
        # data = data.unsqueeze(0)
        y_val = model.forward(data)
        # print(f'{i+1}.) {str(y_val.argmax().item())} {y_test[i]}')
        if y_val.argmax().item() == y_test[i]:
            correct += 1
print(correct)
# df = pd.DataFrame(value_list)
# df.to_csv('/home/jy/Desktop/3_DNN.csv', index=False)
# print(df)
