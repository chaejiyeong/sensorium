
import result
import param_set

# name: 저장하고 싶은 파일 이름 (이름 여러개 X)
# data_list: 돌리고 싶은 데이터 이름 (여러개 가능)
# model_list: 돌리고 싶은 방법론 (여러개 가능)
# random_seed: 돌리는 횟수 (int)

# ==================== Moderate Here! ================================
name = ["birth"]
data_list = ["17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27"]
model_list = ["KNN", "MLP", "DNN", "GB", "LR", "RF"]
# "SVM " error!
random_seed = 10

param_set.param_0128(name, data_list, model_list, random_seed)
result.average(model_list, name, data_list, random_seed)
# ====================================================================