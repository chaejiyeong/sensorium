
import result
import param_set
import os
import preprocess

# name: 저장하고 싶은 파일 이름 (이름 여러개 X)
# data_list: 돌리고 싶은 데이터 이름 (여러개 가능)
# model_list: 돌리고 싶은 방법론 (여러개 가능)
# random_seed: 돌리는 횟수 (int)

# 코드 수정하는 법
# 1. pre_process에서 하던대로, 원하는 데이터로 바꾸기
# 2. augmentation 적용할꺼면 aug(data, i ,1)로 설정 후 인덱스 맞게 바꾸기
# 3. 적용안할꺼면 aug(data, i ,0)으로 설정 후 진행 
# 4. 돌릴때 data 이름을 구분 가능한 이름으로 잘 설정
# 5. random seed를 상황에 맞게 설정

# ==================== Moderate Here! ================================
names = ["0206"]
data_list = ["SH2"]
model_list = ["KNN", "MLP", "DNN", "GB", "LR", "RF", "SVM"]
# model_list = ["KNN"]
random_seed = 1
# ====================================================================

for name in names:
    result.createFolder('log/' + name)
    result.createFolder('result/' + name)

# 1. original: 원본 코드
# for data in data_list:
#     preprocess.pre_process(data)
# param_set.param(names, data_list, model_list, random_seed)
# result.average(model_list, names, data_list, random_seed)
# result.result(names, data_list, random_seed)

# 2. augmentation: 데이터 증대용 코드
for data in data_list:
    preprocess.pre_process(data)
    for i in range(random_seed):
        x_train, x_test, y_train, y_test  = preprocess.pre_process_aug(data, i, 0)
        param_set.param_aug(names, data_list, model_list, i, x_train, x_test, y_train, y_test)
    result.average(model_list, names, data_list, random_seed)
    result.result(names, data_list, random_seed)


# 3. validation: 확인용 코드
# for data in data_list:
#     preprocess.pre_process(data)
#     for i in range(random_seed):
#         x_train, x_test, y_train, y_test  = preprocess.pre_process_aug(data, i)
#         param_set.param_test(names, data_list, model_list, i, x_train, x_test, y_train, y_test)
#     result.average(model_list, names, data_list, random_seed)
