
import result
import param_set
import os
import preprocess

# name: 저장하고 싶은 파일 이름 (이름 여러개 X)
# data_list: 돌리고 싶은 데이터 이름 (여러개 가능)
# model_list: 돌리고 싶은 방법론 (여러개 가능)
# random_seed: 돌리는 횟수 (int)

# ==================== Moderate Here! ================================
names = ["0206"]
data_list = ["SH1"]
model_list = ["KNN", "MLP", "DNN", "GB", "LR", "RF", "SVM"]
# model_list = ["SVM"]
random_seed = 1
for name in names:
    result.createFolder('log/' + name)
    result.createFolder('result/' + name)

# original: 원본 코드
# for data in data_list:
#     preprocess.pre_process(data)
# param_set.param(names, data_list, model_list, random_seed)
# result.average(model_list, names, data_list, random_seed)

# augmentation: 데이터 증대용 코드
for data in data_list:
    preprocess.pre_process(data)
    for i in range(random_seed):
        x_train, x_test, y_train, y_test  = preprocess.pre_process_aug(data, i)
        param_set.param_aug(names, data_list, model_list, i, x_train, x_test, y_train, y_test)
    result.average(model_list, names, data_list, random_seed)
# ====================================================================