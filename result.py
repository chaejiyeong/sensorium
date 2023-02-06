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
                
def result(names, data, seeds):
    np.set_printoptions(precision = 4)
    for datum in data:
        result = dict()
        high_result = dict()
        result_csv = pd.DataFrame()
        for name in names:
            for seed in range(seeds):
                try:
                    file_name = 'log/' + name + '/' + datum + '_' + str(seed) + '_' + datum + '.csv'
                    file = pd.read_csv(file_name)
        
                    # file = file[['model', 'macro precision scores', 'micro precision scores',
                    #             'macro recall scores',
                    #             'micro recall scores',
                    #             'macro f1 scores',
                    #             'micro f1 scores']]
                    
                    for idx, row in file.iterrows():
                        if seed == 0:
                            result[row['model']] = dict()
                            result[row['model']]['model'] = row['model']
                            result[row['model']]['macro precision scores'] = [float(row['macro precision scores'])]
                            result[row['model']]['macro recall scores'] = [float(row['macro recall scores'])]
                            result[row['model']]['macro f1 scores'] = [float(row['macro f1 scores'])]
                            
                            result[row['model']]['micro precision scores'] = [float(row['micro precision scores'])]
                            result[row['model']]['micro recall scores'] = [float(row['micro recall scores'])]
                            result[row['model']]['micro f1 scores'] = [float(row['micro f1 scores'])]
                        else:
                            result[row['model']]['macro precision scores'].append(float(row['macro precision scores']))
                            result[row['model']]['macro recall scores'].append(float(row['macro recall scores']))
                            result[row['model']]['macro f1 scores'].append(float(row['macro f1 scores']))
                            
                            result[row['model']]['micro precision scores'] = [float(row['micro precision scores'])]
                            result[row['model']]['micro recall scores'] = [float(row['micro recall scores'])]
                            result[row['model']]['micro f1 scores'] = [float(row['micro f1 scores'])]
                        
                        if seed == seeds - 1:
                            result[row['model']]['macro precision high'] = np.max(result[row['model']]['macro precision scores'])
                            result[row['model']]['macro precision mean'] = np.mean(result[row['model']]['macro precision scores'])
                            result[row['model']]['macro recall high'] = np.max(result[row['model']]['macro recall scores'])
                            result[row['model']]['macro recall mean'] = np.mean(result[row['model']]['macro recall scores'])
                            result[row['model']]['macro f1 high'] = np.max(result[row['model']]['macro f1 scores'])
                            result[row['model']]['macro f1 mean'] = np.mean(result[row['model']]['macro f1 scores'])
                            
                            result[row['model']]['micro precision high'] = np.max(result[row['model']]['micro precision scores'])
                            result[row['model']]['micro precision mean'] = np.mean(result[row['model']]['micro precision scores'])
                            result[row['model']]['micro recall high'] = np.max(result[row['model']]['micro recall scores'])
                            result[row['model']]['micro recall mean'] = np.mean(result[row['model']]['micro recall scores'])
                            result[row['model']]['micro f1 high'] = np.max(result[row['model']]['micro f1 scores'])
                            result[row['model']]['micro f1 mean'] = np.mean(result[row['model']]['micro f1 scores'])
                            
                except FileNotFoundError:
                    pass
        

    first = 0
    for key, value in result.items():
        if first == 0:
            idx_list = []
            idx_list.extend(result[key].keys())
            result_csv = pd.DataFrame(columns = idx_list)
            first += 1
        result_csv = result_csv.append(result[key], ignore_index= True)
    
    if first == 1:
        result_name = 'result/' + name + '/' + name + '_' + datum + '.csv'
        result_csv.to_csv(result_name)
                

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)