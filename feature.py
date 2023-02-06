from pickletools import int4
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import csv

def sin_wave(amp, freq, time):
    return amp * np.sin(2*np.pi*freq*time)

def feature_relation():
    data = pd.read_csv('data/x_data_SH01.csv')
    # data = data.iloc[:, 15:]
    corr = data.corr(method='spearman')
    # corr = data.corr(method='kendall')
    plt.figure()
    heatmap = sns.heatmap(corr, cmap = 'Greys').get_figure()
    heatmap.savefig("figure/heatmap_spearman.png")

def feature_selection():
    data = pd.read_csv('data/x_data_24.csv')
    data = data.iloc[:, 15:]
    corr = data.corr(method='pearson')
    corr30 = corr.nlargest(30, '35')
    corr30 = corr30[list(corr30.index)]
    print(corr30)

def inverse_fft():
    data = pd.read_csv('data/x_data_SH01.csv')
    hz = [i for i in range(0, 5120, 10)]
    hz = np.array(hz, int)
    for idx, row in data.iterrows():
        level = np.array(row, int)
        level = np.fft.ifft(level)
        plt.plot(hz, level.real, label = 'real')
        plt.show()
        input()
        
def example():
    
    data = pd.read_csv('data/x_data_SH06.csv')
    y_target = pd.read_csv('data/y_data_SH06.csv')
    all_t = np.arange(0, 0.3, 0.001)
    
    f = open('figure/result.csv', 'w', newline = '')
    wr = csv.writer(f)
    
    leak = 0
    noleak = 0
    leak_mean = []
    leak_var = []
    noleak_mean = []
    noleak_var = []
    leak_sin = np.zeros((300,), dtype = float)
    noleak_sin = np.zeros((300,), dtype = float)
    for idx, row in data.iterrows():
        # 10번 측정의 최고 주파수
        high_freq = row[:10]
        
        # 10번 측정의 최고 주파수의 레벨
        high_level = row[10:20]
        
        sin_sum = np.zeros((300,), dtype = float)
        t = np.arange(0, 0.3, 0.001)
        hz = [i for i in range(0, 5120, 10)]
        level = np.array(row[20:], int)
        for idx3 in range(len(level)):
            rand_pram = random.randint(0, 4) * 100
            if level[idx3] < 300:
                # sin_sum[rand_pram: rand_pram + 100] += sin_wave(level[idx], hz[idx], t)
                sin_sum += abs(sin_wave(level[idx3], hz[idx3], t))
        
        if y_target.iloc[idx][0] == 1:
            leak += 1
            leak_sin += sin_sum
        else:
            noleak += 1
            noleak_sin += sin_sum
        # plt.cla()
        # plt.title(y_target.iloc[idx][0])
        # plt.plot(all_t, sin_sum, label = 'origin')
        for i in range(10):
            if y_target.iloc[idx][0] == 1:
                tmp = sin_sum - abs(sin_wave(high_level[i], high_freq[i], t))
                leak_mean.append(np.mean(tmp))
                leak_var.append(np.var(tmp))
            else:
                tmp = sin_sum - abs(sin_wave(high_level[i], high_freq[i], t))
                noleak_mean.append(np.mean(tmp))
                noleak_var.append(np.var(tmp))
        # plt.legend(loc = 'lower left')
        # plt.savefig('figure/' + str(y_target.iloc[idx][0]) + '_' + str(idx) + '.png')
        # input()
    leak_sin = leak_sin / leak
    noleak_sin = noleak_sin / noleak
    
    plt.plot(all_t, leak_sin, label = 'leak')
    plt.plot(all_t, noleak_sin, label = 'no leak')
    plt.legend()
    plt.show()
        
    print("mean of mean in leak", np.mean(leak_mean))
    print("var of mean in leak", np.var(leak_mean))
    print("mean of var in leak", np.mean(leak_var))
    print("var of var in leak", np.var(leak_var))
    
    print("mean of mean in noleak", np.mean(noleak_mean))
    print("var of mean in noleak", np.var(noleak_mean))
    print("mean of var in noleak", np.mean(noleak_var))
    print("var of var in noleak", np.var(noleak_var))
# feature_relation()
# feature_selection()
# inverse_fft()
example()