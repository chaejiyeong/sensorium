# csv file read
from cmath import nan
import csv
import pandas as pd
import numpy as np

def average(models, names, data, seeds):
    for model in models:
        for datum in data:
            result = dict()
            print(model, "with ", datum)
            result_csv = pd.DataFrame()
            for name in names:
                for seed in range(seeds):
                    try:
                        file_name = 'log/' + model + '_' + name + '_' + datum + '_' + str(seed) + '.csv'
                        file = pd.read_csv(file_name)
                        file = file[['params', 'mean_test_score']]
                        for idx, row in file.iterrows():
                            # dict_tmp = {row['params'][1:len(row['params'])-1]}
                            # print(row['params'][1:len(row['params']) - 1])
                            if (row['mean_test_score'] > 0):
                                if row['params'] in result:
                                    result[row['params']].append(float(row['mean_test_score']))
                                else:
                                    result[row['params']] = [float(row['mean_test_score'])]
                        
                    except FileNotFoundError:
                        pass
                    
            
            for key, value in result.items():
                result[key] = np.mean(value)
            
            first = 0
            for key, value in result.items():
                if first == 0:
                    idx_list = ['score']
                    idx_list.extend(eval(key).keys())
                    result_csv = pd.DataFrame(columns = idx_list)
                    first += 1
                value_dict = eval(key)
                value_dict['score'] = value
                result_csv = result_csv.append(value_dict, ignore_index= True)
                
            result_name = 'result/' + model + '_' + name + '_' + datum + '.csv'
            result_csv.to_csv(result_name)
                
                
models = ["KNN", "MLP", "DNN", "GB", "LR", "RF", "SVM"]
# models = ["SVM"]
names = ["0126"]
data = ["02"]
seeds = 10

average(models, names, data, seeds)