# coding: utf-8
# make_data
from sklearn.model_selection import train_test_split
# Machine Learning Methods
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import  f1_score, precision_recall_fscore_support
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
    
    def f(self, p, r):
        return 2*p*r/(p+r)
    
    def __init__(self, name):
        x_name = "data/x_data_" + name + ".csv"
        y_name = "data/y_data_" + name + ".csv"
        self.x_data = read_csv(x_name).values
        self.y_data = read_csv(y_name).values
        
        self.result = pd.DataFrame({'random state': [],
                                    'parameters': [],
                                    'macro precision scores': [],
                                    'micro precision scores': [],
                                    'macro recall scores': [],
                                    'micro recall scores': [],
                                    'macro f1 scores': [],
                                    'micro f1 scores': []})
        return
    
    # ======================== Machine Learning Methods ========================
    
    def save_result(self, folder_name, name, data):
        if self.result.size != 0:
            result_name =  'log/' + str(folder_name) + '/'+ str(name) +'_' + str(data) + '.csv' 
            self.result.to_csv(result_name)
    
    # Find parameter
    def grid_search_CV(self, model_name, rand_seed, parameter, folder_name, file_name, averaging = 0):
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
            score_name = 'log/' + folder_name + '/' + model_name + '_' + file_name + '.csv' 
            scores.to_csv(score_name)
            
        # Printing
        print("================================================")
        print("Time: ", round((time.time() - start) / 60, 3))
        print("Parameters: ", gsmodel.best_params_)
        print("Estimation Score: ", gsmodel.best_score_)
        
        gs_model = gsmodel.best_estimator_
        y_hat = gs_model.predict(x_test)
        matrix = metrics.confusion_matrix(y_test, y_hat)
        print("Confusion Matrix:")
        print(matrix)
        
        p_macro, r_macro, f_macro, support_macro = precision_recall_fscore_support(y_true=y_test, y_pred=y_hat, labels=[0, 1], average='macro')
        p_micro, r_micro, f_micro, support_micro = precision_recall_fscore_support(y_true=y_test, y_pred=y_hat, labels=[0, 1], average='micro')
        my_f_macro = self.f(p_macro, r_macro)
        my_f_micro = self.f(p_micro, r_micro)
        
        print('my f macro {}'.format(my_f_macro))
        print('my f micro {}'.format(my_f_micro))

        print('macro: p {}, r {}, f1 {}'.format(p_macro, r_macro, f_macro))

        print('micro: p {}, r {}, f1 {}'.format(p_micro, r_micro, f_micro))
        print()
        print("================================================")
        
        value = {'random state': rand_seed,
                'model': model_name,
                'parameters': gsmodel.best_params_,
                'macro precision scores': p_macro,
                'macro recall scores': r_macro,
                'macro f1 scores': f_macro,
                'micro precision scores': p_micro,
                'micro recall scores': r_micro,
                'micro f1 scores': f_micro
                }
        self.result = self.result.append(value, ignore_index=True)    
            
        # except:
            # print("Something Wrong!")
            
    # Find parameter
    def grid_search_CV_aug(self, model_name, rand_seed, parameter, folder_name, file_name, x_train, x_test, y_train, y_test):
    
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
        # accuracy precision f1_macro f1_micro, recall
        gsmodel = GridSearchCV(model, param_grid = parameter, cv = 3,
                        scoring="f1_macro", verbose=0, refit = True, n_jobs = -1)
        gsmodel.fit(x_train, y_train.ravel())
        
        scores = pd.DataFrame(gsmodel.cv_results_)
        score_name = 'log/' + folder_name + '/' + model_name + '_' + file_name + '.csv' 
        scores.to_csv(score_name)
            
        # Printing
        print("================================================")
        print("Time: ", round((time.time() - start) / 60, 3))
        print("Parameters: ", gsmodel.best_params_)
        print("Estimation Score: ", gsmodel.best_score_)
        
        gs_model = gsmodel.best_estimator_
        y_hat = gs_model.predict(x_test)
        matrix = metrics.confusion_matrix(y_test, y_hat)
        print("Confusion Matrix:")
        print(matrix)
        
        p_macro, r_macro, f_macro, support_macro = precision_recall_fscore_support(y_true=y_test, y_pred=y_hat, labels=[0, 1], average='macro')
        p_micro, r_micro, f_micro, support_micro = precision_recall_fscore_support(y_true=y_test, y_pred=y_hat, labels=[0, 1], average='micro')
        my_f_macro = self.f(p_macro, r_macro)
        my_f_micro = self.f(p_micro, r_micro)
        
        print('my f macro {}'.format(my_f_macro))
        print('my f micro {}'.format(my_f_micro))

        print('macro: p {}, r {}, f1 {}'.format(p_macro, r_macro, f_macro))

        print('micro: p {}, r {}, f1 {}'.format(p_micro, r_micro, f_micro))
        print()
        print("================================================")
        
        value = {'random state': rand_seed,
                'model': model_name,
                'parameters': gsmodel.best_params_,
                'macro precision scores': p_macro,
                'macro recall scores': r_macro,
                'macro f1 scores': f_macro,
                'micro precision scores': p_micro,
                'micro recall scores': r_micro,
                'micro f1 scores': f_micro
                }
        self.result = self.result.append(value, ignore_index=True)   
            
        # except:
            # print("Something Wrong!")