import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def feature_relation():
    data = pd.read_csv('data/x_data_24.csv')
    data = data.iloc[:, 15:]
    corr = data.corr(method='spearman')
    # corr = data.corr(method='kendall')
    plt.figure()
    heatmap = sns.heatmap(corr, cmap = 'Greys').get_figure()
    heatmap.savefig("figure/heatmap24_spearman.png")
    
def feature_selection():
    data = pd.read_csv('data/x_data_24.csv')
    data = data.iloc[:, 15:]
    corr = data.corr(method='pearson')
    corr30 = corr.nlargest(30, '35')
    corr30 = corr30[list(corr30.index)]
    print(corr30)
    
# feature_relation()
feature_selection()