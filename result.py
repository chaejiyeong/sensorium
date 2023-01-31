# csv file read
from cmath import nan
import csv
import pandas as pd
import numpy as np
import os

def average(models, names, data, seeds):
    np.set_printoptions(precision = 4)
    for model in models:
        for datum in data:
            result = dict()
            high_result = dict()
            result_csv = pd.DataFrame()
            for name in names:
                for seed in range(seeds):
                    try:
                        file_name = 'log/' + name + '/' + model +  '_' + datum + '_' + str(seed) + '.csv'
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
                high_result[key] = np.max(value)
            
            first = 0
            for key, value in result.items():
                if first == 0:
                    idx_list = ['score']
                    idx_list.extend(eval(key).keys())
                    result_csv = pd.DataFrame(columns = idx_list)
                    first += 1
                value_dict = eval(key)
                value_dict['highest_score'] = high_result[key]
                value_dict['score'] = value
                result_csv = result_csv.append(value_dict, ignore_index= True)
            
            if first == 1:
                result_name = 'result/' + name + '/' + model + '_' + name + '_' + datum + '.csv'
                result_csv.to_csv(result_name)
                
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)