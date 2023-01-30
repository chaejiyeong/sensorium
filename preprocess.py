# coding: utf-8

# csv file read
import csv
from math import nan
from platform import architecture
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


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

def pre_process_03(name):
    data = open('data/raw_data.csv')
    data = csv.reader(data)
    header = next(data)
    
    freq_data = np.array([])
    pipe_data = np.array([])
    target_data = np.array([])
    
    for row in data:
        if row[1] == '-1':
            continue
        
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

def pre_process_04(name):
    data = open('data/raw_data.csv')
    data = csv.reader(data)
    header = next(data)
    
    freq_data = np.array([])
    pipe_data = np.array([])
    target_data = np.array([])
    
    onehot_encoder = OneHotEncoder(sparse = False)
    scaler = StandardScaler()
    
    for row in data:
        
        # Remove -1
        if row[1] == '-1':
            continue
        
        arr = []
        # pipe data: augmentation
        # for i in np.array(row[1:4]):
        #         arr.extend([i for j in range(10)])
        if len(freq_data) == 0:
            freq_data = np.append(freq_data, np.array(row[6:]), axis=0)
            pipe_data = np.append(
                pipe_data, np.array(np.array(row[1:4])), axis=0)
        else:
            freq_data = np.vstack([freq_data, np.array(row[6:])])
            pipe_data = np.vstack([pipe_data, np.array(row[1:4])])
        if (row[4] == '0' or row[4] == '1'):
            target_data = np.append(target_data, np.array(0))
        if (row[4] == '2' or row[4] == '3' or row[4] == '4'):
            target_data = np.append(target_data, np.array(1))
    pipe_data = np.array(pipe_data, int)
    pipe_data_one_hot = onehot_encoder.fit_transform(pipe_data)
    pipe_data_one_hot = np.array(pipe_data_one_hot, float)
    pipe_data_one_hot = pipe_data_one_hot * 100
    
    freq_data = np.array(freq_data, int)
    freq_data =scaler.fit_transform(freq_data)
    
    x_data = np.concatenate([pipe_data_one_hot, freq_data], axis=1)
    x_data = np.array(x_data, float)
    # x_data = np.array(x_data, int)
    
    x_df = pd.DataFrame(x_data)
    x_name = 'data/x_data_' + name + '.csv' 
    x_df.to_csv(x_name, index=False)
    
    target_data = np.array(target_data, int)
    y_df = pd.DataFrame(target_data)
    y_name = 'data/y_data_' + name + '.csv' 
    y_df.to_csv(y_name, index=False)
    
def pre_process_05(name):
    data = open('data/raw_data.csv')
    data = csv.reader(data)
    header = next(data)
    
    freq_data = np.array([])
    pipe_data = np.array([])
    target_data = np.array([])
    
    onehot_encoder = OneHotEncoder(sparse = False)
    scaler = MinMaxScaler()
    
    for row in data:

        # Remove -1
        if row[1] == '-1' or row[1] == '0':
            continue
        
        arr = []
        # pipe data: augmentation
        # for i in np.array(row[1:4]):
        #         arr.extend([i for j in range(10)])
        pipes = row[2:4]
        if len(freq_data) == 0:
            freq_data = np.append(freq_data, np.array(row[6:]), axis=0)
            pipe_data = np.append(
                pipe_data, np.array(np.array(pipes)), axis=0)
        else:
            freq_data = np.vstack([freq_data, np.array(row[6:])])
            pipe_data = np.vstack([pipe_data, np.array(pipes)])
        if (row[4] == '0' or row[4] == '1'):
            target_data = np.append(target_data, np.array(0))
        if (row[4] == '2' or row[4] == '3' or row[4] == '4'):
            target_data = np.append(target_data, np.array(1))
    pipe_data = np.array(pipe_data, int)
    pipe_data_one_hot = onehot_encoder.fit_transform(pipe_data)
    pipe_data_one_hot = np.array(pipe_data_one_hot, float)
    pipe_data_one_hot = pipe_data_one_hot * 100
    
    freq_data = np.array(freq_data, int)
    freq_data =scaler.fit_transform(freq_data)
    
    x_data = np.concatenate([pipe_data_one_hot, freq_data], axis=1)
    x_data = np.array(x_data, float)
    # x_data = np.array(x_data, int)
    
    x_df = pd.DataFrame(x_data)
    x_name = 'data/x_data_' + name + '.csv' 
    x_df.to_csv(x_name, index=False)
    
    target_data = np.array(target_data, int)
    y_df = pd.DataFrame(target_data)
    y_name = 'data/y_data_' + name + '.csv' 
    y_df.to_csv(y_name, index=False)

def pre_process_06(name):
    data = open('data/raw_data2.csv')
    data = csv.reader(data)
    header = next(data)
    
    freq_data = np.array([])
    pipe_data = np.array([])
    high_freq_data = np.array([])
    high_level_data = np.array([])
    target_data = np.array([])
    
    onehot_encoder = OneHotEncoder(sparse = False)
    scaler = MinMaxScaler()
    
    for row in data:
        # Remove -1
        if row[0] == '-1':
            continue
        
        # Data rows
        pipes = row[0:3]
        ltype = row[3]        
        high = row[12:32]
        high_freq = []
        high_level = []
        for idx in range(20):
            if idx % 2 == 0:
                high_freq.append(high[idx])
            else:
                high_level.append(high[idx])
                
        freq = row[32:]
        
        empty = 0
        # Remove empty
        for i in pipes:
            if len(i) == 0:
                empty = 1
        if empty == 1:
            continue
        
        # remove 옥내누수
        # if (ltype == '0'):
        #     continue
        
        # # remove 옥외누수
        if (ltype == '1'):
            continue
        
        # pipe data: augmentation
        # for i in np.array(pipes):
        #         pipes.extend([i for j in range(10)])
        
        if len(freq_data) == 0:
            pipe_data = np.append(
                pipe_data, np.array(np.array(pipes)), axis=0)
            high_freq_data = np.append(high_freq_data, np.array(high_freq), axis = 0)
            high_level_data = np.append(high_level_data, np.array(high_level), axis = 0)
            freq_data = np.append(freq_data, np.array(freq), axis=0)
        else:
            pipe_data = np.vstack([pipe_data, np.array(pipes)])
            high_freq_data = np.vstack([high_freq_data, np.array(high_freq)])
            high_level_data = np.vstack([high_level_data, np.array(high_level)])
            freq_data = np.vstack([freq_data, np.array(freq)])
            
        if (ltype == '0' or ltype == '1'):
            target_data = np.append(target_data, np.array(0))
        if (ltype == '2' or ltype == '3' or ltype == '4'):
            target_data = np.append(target_data, np.array(1))

    # one hot encoding
    pipe_data = np.array(pipe_data, int)
    pipe_data_one_hot = onehot_encoder.fit_transform(pipe_data)
    pipe_data_one_hot = np.array(pipe_data_one_hot, float)
    # scailing
    pipe_data_one_hot = pipe_data_one_hot * 100
    
    # normalization
    freq_data = np.array(freq_data, int)
    freq_data = scaler.fit_transform(freq_data)
    high_freq_data = np.array(high_freq_data, int)
    high_freq_data =scaler.fit_transform(high_freq_data)
    high_level_data = np.array(high_level_data, int)
    high_level_data = scaler.fit_transform(high_level_data)
    
    x_data = np.concatenate([pipe_data_one_hot, high_freq_data, high_level_data, freq_data], axis=1)
    x_data = np.array(x_data, float)
    
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
# pre_process_02("02")

# Frequency + Pipe info (p pd s) * 10 (remove -1)
# pre_process_03("03")

# "04" : Frequency + Pipe info (p pd s) * 10 (remove -1) + one hot encoding 
# "05" : Frequency + Pipe info (p pd s) * 10 (remove -1) + one hot encoding  * 100
pre_process_06("27")