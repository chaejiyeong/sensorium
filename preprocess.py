# coding: utf-8

# csv file read
import csv
from math import nan
from platform import architecture
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def sin_wave(amp, freq, time):
    return amp * np.sin(2*np.pi*freq*time)

def analysis():
    data = open('data/raw_data2.csv')
    data = csv.reader(data)
    header = next(data)
    
    onehot_encoder = OneHotEncoder(sparse = False)
    scaler = MinMaxScaler()
    

    leak_dict = dict()
    no_leak_dict = dict()
    
    for row in data:
        # Remove -1
        if row[0] == '-1':
            continue
        
        # Data rows
        pipes = str(row[0:3])
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
        # if (ltype == '1'):
        #    continue
            
        if (ltype == '0' or ltype == '1'):
            if pipes in leak_dict:
                leak_dict[pipes].append(ltype)
            else:
                leak_dict[pipes] = [ltype]
                
        if (ltype == '2' or ltype == '3' or ltype == '4'):
            if pipes in no_leak_dict:
                no_leak_dict[pipes].append(ltype)
            else:
                no_leak_dict[pipes] = [ltype]
    
    print("Leakage data")
    for key, value in leak_dict.items():
        if key in no_leak_dict:
            print(key)
            print(len(value), ":" , len(no_leak_dict[key]))
        else:
            # print(key, ":", len(value))
            pass
        
    
    print("No Leakgage data")
    for key, value in no_leak_dict.items():
        if key in leak_dict:
            pass
        else:
            # print(key, ":", len(value))
            pass

def pre_process(name):
    data = open('data/raw_data2.csv')
    data = csv.reader(data)
    header = next(data)
    
    freq_data = np.array([])
    sin_data = np.array([])
    pipe_data = np.array([])
    high_freq_data = np.array([])
    high_level_data = np.array([])
    target_data = np.array([])
    
    onehot_encoder = OneHotEncoder(sparse = False)
    scaler = MinMaxScaler()
    
    # 사용할 관종 정보 선택: 밑에서 고르기
    # ['0', '0', '0'], ['0', '0', '2'], ['0', '1', '1'], ['0', '1', '0'], ['0', '3', '0'], ['0', '3', '1'], 
    # ['0', '3', '2'], ['0', '5', '2'], ['1', '6', '0'], ['1', '6', '2'], ['1', '7', '0'], ['1', '9', '2'], ['1', '10', '0']
    used = [['0', '3', '2']]
    leak_num = 0
    no_leak_num = 0
    for row in data:
        # Remove -1 (관종 정보 없는 것 제거)
        if row[0] == '-1':
            continue
        
        # 관종 정보
        pipes = row[0:3]
        
        # Use only specific type (특정 관종만 사용)
        fin = 0
        for types in used:
            if pipes == types:
                fin = 1
        if fin == 0:
            continue
        
        # 누수, 비누수 (타겟 정보)
        ltype = row[3]
        
        high = row[12:32]
        
        # 10번 측정의 최고 주파수
        high_freq = []
        
        # 10번 측정의 최고 주파수의 레벨
        high_level = []
        for idx in range(20):
            if idx % 2 == 0:
                high_freq.append(high[idx])
            else:
                high_level.append(high[idx])
        
        # 0Hz ~ 5120 Hz의 평균 주파수
        freq = row[33:]
        
        # Remove empty (공백 제거)
        empty = 0
        for i in pipes:
            if len(i) == 0:
                empty = 1
        if empty == 1:
            continue
        
        # random augmentation
        # for i in range(10):
        freq = np.array(freq, int)
        sin = np.zeros((200,), dtype = float)
        t = np.arange(0, 0.2, 0.001)
        hz = [i for i in range(0, 5120, 10)]
        for idx in range(len(freq)):
            # normal sin data
            # sin += sin_wave(freq[idx], hz[idx], t)
            
            # noise reduction
            # if hz[idx] < 300:
                # abs sin data
            sin += abs(sin_wave(freq[idx], hz[idx], t))
            
            # random augmentation
            # rand_pram = random.randint(0, 4) * 100
            # sin[rand_pram: rand_pram + 100] += sin_wave(freq[idx], hz[idx], t)
            
        # 옥내 누수 제거
        if (ltype == '0'):
            continue
        
        if (ltype == '3' or ltype == '4'):
            continue
        
        # 옥외 누수 제거
        #if (ltype == '1'):
            #continue
            
            # [pipes * 10] 관종 정보 10번 복제
            # pipe data: augmentation
            # for i in np.array(pipes):
            #         pipes.extend([i for j in range(10)])
            
        # numpy array 생성 (수정 안해도 됨)
        if len(freq_data) == 0:
            pipe_data = np.append(
                pipe_data, np.array(np.array(pipes)), axis=0)
            high_freq_data = np.append(high_freq_data, np.array(high_freq), axis = 0)
            high_level_data = np.append(high_level_data, np.array(high_level), axis = 0)
            freq_data = np.append(freq_data, np.array(freq), axis=0)
            sin_data = np.append(sin_data, np.array(sin), axis=0)
        else:
            pipe_data = np.vstack([pipe_data, np.array(pipes)])
            high_freq_data = np.vstack([high_freq_data, np.array(high_freq)])
            high_level_data = np.vstack([high_level_data, np.array(high_level)])
            freq_data = np.vstack([freq_data, np.array(freq)])
            sin_data = np.vstack([sin_data, np.array(sin)])
            
        if (ltype == '0' or ltype == '1'):
            target_data = np.append(target_data, np.array(1))
            leak_num += 1
        if (ltype == '2' or ltype == '3' or ltype == '4'):
            target_data = np.append(target_data, np.array(0))
            no_leak_num += 1
                
    print(leak_num, ":", no_leak_num)

    # one hot encoding: 관종 데이터
    # pipe_data = np.array(pipe_data, int)
    # pipe_data = onehot_encoder.fit_transform(pipe_data)
    # pipe_data = np.array(pipe_data, float)
    
    # scailing: 관종 데이터
    # pipe_data_one_hot = pipe_data_one_hot * 100
    
    # normalization: 0Hz ~ 5120 Hz의 평균 주파수
    freq_data = np.array(freq_data, int)
    # freq_data = scaler.fit_transform(freq_data)
    
    # normalization: 10번 측정의 최고 주파수
    # high_freq_data = np.array(high_freq_data, int)
    # high_freq_data =scaler.fit_transform(high_freq_data)
    sin_data = scaler.fit_transform(sin_data)
    
    # normalization: 10번 측정의 최고 주파수의 레벨
    high_level_data = np.array(high_level_data, int)
    # high_level_data = scaler.fit_transform(high_level_data)
    
    # Principal component analysis (PCA): 0Hz ~ 5120 Hz의 평균 주파수
    # pca = PCA(n_components = 100)
    # freq_data = pca.fit_transform(freq_data)
    
    # x_data concatenate : 원하는 데이터을 list에 넣으면 됨
    # 넣을 수 있는 데이터 : pipe_data, freq_data, sin_data, high_freq_data, high_level_data
    x_data = np.concatenate([freq_data], axis=1)
    x_data = np.array(x_data, float)
    
    # x_data 저장
    x_df = pd.DataFrame(x_data)
    x_name = 'data/x_data_' + name + '.csv' 
    x_df.to_csv(x_name, index=False)
    
    # y_data 저장
    target_data = np.array(target_data, int)
    y_df = pd.DataFrame(target_data)
    y_name = 'data/y_data_' + name + '.csv' 
    y_df.to_csv(y_name, index=False)

def split_data(x_data, y_data, rand_seed):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, 
            stratify = y_data, test_size = 0.2, random_state = rand_seed)
    return x_train, x_test, y_train, y_test

def pre_process_aug(name, rand_seed):    
    x_name = "data/x_data_" + name + ".csv"
    y_name = "data/y_data_" + name + ".csv"
    x_data = pd.read_csv(x_name).values
    y_data = pd.read_csv(y_name).values
    
    x_train, x_test, y_train, y_test = split_data(x_data, y_data, rand_seed)    
    
    new_x_data = np.array([])
    new_y_data = np.array([])
    freq_data = np.array([])
    sin_data = np.array([])
    pipe_data = np.array([])
    high_freq_data = np.array([])
    high_level_data = np.array([])
    
    onehot_encoder = OneHotEncoder(sparse = False)
    scaler = MinMaxScaler()
    
    leak_num = 0
    no_leak_num = 0
    
    # standard distribution 
    sigma = 0.1
    
    for idx in range(len(x_train)):
        row = x_train[idx]
        # 관종 정보
        pipes = row[:]
        
        # 누수, 비누수 (타겟 정보)
        ltype = y_train[idx]
        
        # high = row[]
        
        # 10번 측정의 최고 주파수
        high_freq = row[:]
        
        high_level = row[:]
        
        # 0Hz ~ 5120 Hz의 평균 주파수
        freq = row
        freq = np.array(freq, int)
        
        # augmentation n times
        for i in range(10):
            # # biased noise
            # noise = random.randrange(-20, 20)
            # for hz in range(len(freq)):
            #     if freq[hz] + noise > 0:
            #         freq[hz] = freq[hz] + noise
            #     else:
            #         pass
            # add Gaussian Noise
            for hz in range(len(freq)):
                # freq[hz] = np.random.normal(freq[hz], sigma, 1)
                freq[hz] = freq[hz] + np.random.normal(0, 0.2, 1)
            
            # time domain data
            sin = np.zeros((200,), dtype = float)
            t = np.arange(0, 0.2, 0.001)
            hz = [i for i in range(0, 5120, 10)]
            for idx in range(len(freq)):
                sin += abs(sin_wave(freq[idx], hz[idx], t))
                # normal sin data
                # sin += sin_wave(freq[idx], hz[idx], t)
                
                # noise reduction
                # if hz[idx] < 300:
                    # abs sin data
                    # sin += abs(sin_wave(freq[idx], hz[idx], t))
                
                # random augmentation
                # rand_pram = random.randint(0, 4) * 100
                # sin[rand_pram: rand_pram + 100] += sin_wave(freq[idx], hz[idx], t)
                
            # [pipes * 10] 관종 정보 10번 복제
            # pipe data: augmentation
            # for i in np.array(pipes):
            #         pipes.extend([i for j in range(10)])
            # numpy array 생성 (수정 안해도 됨)
            if len(freq_data) == 0:
                pipe_data = np.append(pipe_data, np.array(np.array(pipes)), axis=0)
                high_freq_data = np.append(high_freq_data, np.array(high_freq), axis = 0)
                high_level_data = np.append(high_level_data, np.array(high_level), axis = 0)
                freq_data = np.append(freq_data, np.array(freq), axis=0)
                sin_data = np.append(sin_data, np.array(sin), axis=0)
                new_y_data = np.append(new_y_data, np.array(ltype), axis = 0)
            else:
                pipe_data = np.vstack([pipe_data, np.array(pipes)])
                high_freq_data = np.vstack([high_freq_data, np.array(high_freq)])
                high_level_data = np.vstack([high_level_data, np.array(high_level)])
                freq_data = np.vstack([freq_data, np.array(freq)])
                sin_data = np.vstack([sin_data, np.array(sin)])
                new_y_data = np.vstack([new_y_data, np.array(ltype)])
            

    # one hot encoding: 관종 데이터
    # pipe_data = np.array(pipe_data, int)
    # pipe_data = onehot_encoder.fit_transform(pipe_data)
    # pipe_data = np.array(pipe_data, float)
    
    # scailing: 관종 데이터
    # pipe_data_one_hot = pipe_data_one_hot * 100
    
    # normalization: 0Hz ~ 5120 Hz의 평균 주파수
    freq_data = np.array(freq_data, float)
    # freq_data = scaler.fit_transform(freq_data)
    
    # normalization: 10번 측정의 최고 주파수
    # high_freq_data = np.array(high_freq_data, int)
    # high_freq_data =scaler.fit_transform(high_freq_data)
    sin_data = scaler.fit_transform(sin_data)
    
    # normalization: 10번 측정의 최고 주파수의 레벨
    high_level_data = np.array(high_level_data, int)
    # high_level_data = scaler.fit_transform(high_level_data)
    
    # Principal component analysis (PCA): 0Hz ~ 5120 Hz의 평균 주파수
    # pca = PCA(n_components = 100)
    # freq_data = pca.fit_transform(freq_data)
    
    # x_data concatenate : 원하는 데이터을 list에 넣으면 됨
    # 넣을 수 있는 데이터 : pipe_data, freq_data, sin_data, high_freq_data, high_level_data
    new_x_data = np.concatenate([freq_data], axis=1)
    new_x_data = np.array(new_x_data, float)
    
    x_train = new_x_data
    y_train = new_y_data    
    return x_train, x_test, y_train, y_test 
    
    
# analysis()
# pre_process("SH02")
