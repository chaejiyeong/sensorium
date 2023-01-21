import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn import datasets, model_selection
# 현재 이미지들은 계량기를 빼지 않은 상태임

dirs = ['누수', '비누수']

data = []
label = []  # 0, 1

for i, d in enumerate(dirs):
    files = os.listdir('/home/jy/data/ltype0/')
    for i in range(len(files)):
        img = Image.open('/home/jy/data/ltype0/ltype0_' + str(i+1) + '.png')
        resize_img = img.resize((128, 128))
        # 이미지를 RGB 컬러로 각각 쪼갠다.
        # https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.split 참조
        a, b, c, d = resize_img.split()
        # 각 쪼갠 이미지를 255로 나눠서 0~1 사이의 값이 나오도록 정규화 한다.
        r_resize_img = np.asarray(np.float32(a) / 255.0)
        b_resize_img = np.asarray(np.float32(b) / 255.0)
        g_resize_img = np.asarray(np.float32(c) / 255.0)
        d_resize_img = np.asarray(np.float32(d) / 255.0)

        abcd_resize_img = np.asarray(
            [r_resize_img, b_resize_img, g_resize_img, d_resize_img])
        data.append(abcd_resize_img)
        label.append(i)

data = np.array(data, dtype='float32')
label = np.array(label, dtype='int64')

train_X, test_X, train_Y, test_Y = model_selection.train_test_split(
    data, label, test_size=0.1)

train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()

test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()

train = TensorDataset(train_X, train_Y)
train_loader = DataLoader(train, batch_size=32, shuffle=True)

# 신경망 구성


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 합성곱층
        self.conv1 = nn.Conv2d(4, 10, 5)  # 입력 채널 수, 출력 채널 수, 필터 크기
        self.conv2 = nn.Conv2d(10, 20, 5)

        # 전결합층
        self.fc1 = nn.Linear(20 * 29 * 29, 50)  # 29=(((((128-5)+1)/2)-5)+1)/2
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        # 풀링층
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 풀링 영역 크기
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 20 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


# 인스턴스 생성
model = Net()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(500):
    total_loss = 0
    for train_x, train_y in train_loader:
        train_x, train_y = Variable(train_x), Variable(train_y)
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item()
    if (epoch+1) % 50 == 0:
        print(epoch+1, total_loss)

test_x, test_y = Variable(test_X), Variable(test_Y)
result = torch.max(model(test_x).data, 1)[1]
accuracy = sum(test_y.data.numpy() == result.numpy()) / \
    len(test_y.data.numpy())
print(accuracy)
