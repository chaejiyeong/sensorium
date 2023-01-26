# coding: utf-8
# make_data
from sklearn.model_selection import train_test_split
# Machine Learning Methods
from sklearn.model_selection import GridSearchCV, KFold
from sklearn import metrics
# MLP
from sklearn.neural_network import MLPClassifier
# KNN
from sklearn.neighbors import KNeighborsClassifier
# Gb
from sklearn.ensemble import GradientBoostingClassifier
# LR
from sklearn.linear_model import LogisticRegression
# RF
from sklearn.ensemble import RandomForestClassifier
# Svm
from sklearn.svm import SVC

# time record
import time

from pandas import read_csv
import pandas as pd
import random

class train():
    
    def __init__(self, name):
        x_name = "data/x_data_" + name + ".csv"
        y_name = "data/y_data_" + name + ".csv"
        self.x_data = read_csv(x_name).values
        self.y_data = read_csv(y_name).values
        
        self.result = pd.DataFrame({'random state': [],
                                    'parameters': [],
                                    'scores': []})
        return
    
    # ======================== Machine Learning Methods ========================
    
    def save_result(self, model_name, name, data):
        print(self.result.size)
        if self.result.size != 0:
            result_name = 'log/' + model_name + '_' + name +'_' + data + '.csv' 
            self.result.to_csv(result_name)
    
    # Find parameter
    def search_parameter(self, model_name, rand_seed, parameter, name, averaging = 0):
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(self.x_data, self.y_data, 
            stratify = self.y_data, test_size = 0.2, random_state = rand_seed)
    
        # K-Nearst Neighborhood
        if model_name == "KNN":
            model = KNeighborsClassifier()
        
        # Multi-Layer Perceptron
        if model_name == "MLP":
            model = MLPClassifier()
            
        # Deep-Neural Network
        if model_name == "DNN":
            model = MLPClassifier()
            
        # Gradient Boosting
        if model_name == "GB":
            model = GradientBoostingClassifier()
            
        # Logistic Regression
        if model_name == "LR":
            model = LogisticRegression()
            
        # Random Forest
        if model_name == "RF":
            model = RandomForestClassifier()
            
        # Support Vector Machine
        if model_name == "SVM":
            model = SVC()
            
        start = time.time()
        # try:
        # n_jobs = -1 : means using all processors
        gsmodel = GridSearchCV(model, param_grid = parameter, cv = 3,
                        scoring="f1_macro", verbose=0, refit = True, n_jobs = -1)
        gsmodel.fit(x_train, y_train.ravel())
        
        if averaging == 1:
            value = {'random state': rand_seed,
                        'parameters': gsmodel.best_params_,
                        'scores': gsmodel.best_score_}
            self.result = self.result.append(value, ignore_index=True)
        
        else:
            scores = pd.DataFrame(gsmodel.cv_results_)
            score_name = 'log/' + model_name + '_' + name + '.csv' 
            scores.to_csv(score_name)
            
        # Printing
        print("================================================")
        print("Time: ", round((time.time() - start) / 60, 3))
        print("Parameters: ", gsmodel.best_params_)
        print("Estimation Score: ", gsmodel.best_score_)
        
        y_hat = gsmodel.predict(x_test)
        matrix = metrics.confusion_matrix(y_test, y_hat)
        print("Confusion Matrix:")
        print(matrix)
        print()
        print("================================================")
            
        # except:
        # print("Something Wrong!")
