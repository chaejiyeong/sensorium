# coding: utf-8

# csv file read
import csv
from platform import architecture
import pandas as pd
import numpy as np

def pre_process_00(name):
    data = open('data/raw_data.csv')
    data = csv.reader(data)
    header = next(data)
    
    freq_data = np.array([])
    target_data = np.array([])
    
    for row in data:
        if len(freq_data) == 0:
            freq_data = np.append(freq_data, np.array(row[6:]), axis=0)
        else:
            freq_data = np.vstack([freq_data, np.array(row[6:])])
        if (row[4] == '0' or row[4] == '1'):
            target_data = np.append(target_data, np.array(0))
        if (row[4] == '2' or row[4] == '3' or row[4] == '4'):
            target_data = np.append(target_data, np.array(1))
        
    x_data = freq_data
    x_data = np.array(x_data, int)
    
    x_df = pd.DataFrame(x_data)
    x_name = 'data/x_data_' + name + '.csv' 
    x_df.to_csv(x_name, index=False)
    
    target_data = np.array(target_data, int)
    y_df = pd.DataFrame(target_data)
    y_name = 'data/y_data_' + name + '.csv' 
    y_df.to_csv(y_name, index=False)
    
def pre_process_01(name):
    data = open('data/raw_data.csv')
    data = csv.reader(data)
    header = next(data)
    
    freq_data = np.array([])
    pipe_data = np.array([])
    target_data = np.array([])
    
    for row in data:
        if len(freq_data) == 0:
            freq_data = np.append(freq_data, np.array(row[6:]), axis=0)
            pipe_data = np.append(
                pipe_data, np.array(row[1:4]), axis=0)
        else:
            freq_data = np.vstack([freq_data, np.array(row[6:])])
            pipe_data = np.vstack([pipe_data, np.array(row[1:4])])
        if (row[4] == '0' or row[4] == '1'):
            target_data = np.append(target_data, np.array(0))
        if (row[4] == '2' or row[4] == '3' or row[4] == '4'):
            target_data = np.append(target_data, np.array(1))
        
    x_data = np.concatenate([pipe_data, freq_data], axis=1)
    x_data = np.array(x_data, int)
    
    x_df = pd.DataFrame(x_data)
    x_name = 'data/x_data_' + name + '.csv' 
    x_df.to_csv(x_name, index=False)
    
    target_data = np.array(target_data, int)
    y_df = pd.DataFrame(target_data)
    y_name = 'data/y_data_' + name + '.csv' 
    y_df.to_csv(y_name, index=False)

def pre_process_02(name):
    data = open('data/raw_data.csv')
    data = csv.reader(data)
    header = next(data)
    
    freq_data = np.array([])
    pipe_data = np.array([])
    target_data = np.array([])
    
    for row in data:
        arr = []
        # pipe data: augmentation
        for i in np.array(row[1:4]):
                arr.extend([i for j in range(10)])
        if len(freq_data) == 0:
            freq_data = np.append(freq_data, np.array(row[6:]), axis=0)
            pipe_data = np.append(
                pipe_data, np.array(np.array(arr)), axis=0)
        else:
            freq_data = np.vstack([freq_data, np.array(row[6:])])
            pipe_data = np.vstack([pipe_data, np.array(arr)])
        if (row[4] == '0' or row[4] == '1'):
            target_data = np.append(target_data, np.array(0))
        if (row[4] == '2' or row[4] == '3' or row[4] == '4'):
            target_data = np.append(target_data, np.array(1))
        
    x_data = np.concatenate([pipe_data, freq_data], axis=1)
    x_data = np.array(x_data, int)
    
    x_df = pd.DataFrame(x_data)
    x_name = 'data/x_data_' + name + '.csv' 
    x_df.to_csv(x_name, index=False)
    
    target_data = np.array(target_data, int)
    y_df = pd.DataFrame(target_data)
    y_name = 'data/y_data_' + name + '.csv' 
    y_df.to_csv(y_name, index=False)


# Only Freqeuncy
# pre_process_00("00")

# Frequency + Pipe info (p pd s)
# pre_process_01("01")

# Frequency + Pipe info (p pd s) * 10
pre_process_02("02")
