
import result
import param_set
import os

# name: 저장하고 싶은 파일 이름 (이름 여러개 X)
# data_list: 돌리고 싶은 데이터 이름 (여러개 가능)
# model_list: 돌리고 싶은 방법론 (여러개 가능)
# random_seed: 돌리는 횟수 (int)

# ==================== Moderate Here! ================================
names = ["0131"]
data_list = ["SH01"]
model_list = ["KNN", "MLP", "DNN", "GB", "LR", "RF", "SVM"]
random_seed = 10

for name in names:
    result.createFolder('log/' + name)
    result.createFolder('result/' + name)
param_set.param_0131(names, data_list, model_list, random_seed)
result.average(model_list, names, data_list, random_seed)
# ====================================================================