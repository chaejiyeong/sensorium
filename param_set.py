import train

def param(name, data_lst, model_lst, rand_seed):
    data_list = data_lst
    model_list = model_lst
    
    for data in data_list:
        trainer = train.train(data)
        for model in model_list:
            for seed in range(rand_seed):
                if model == "KNN":
                    params = {
                    'n_neighbors': [8, 9, 10],
                    'weights': ['distance'],
                    'algorithm' : ['ball_tree', 'kd_tree'],
                    'p' : [1],
                    }
                if model == "MLP":
                    params = {
                    'hidden_layer_sizes': [(100,), (150,), (200,)],
                    'activation': ['logistic'],
                    'solver': ['adam'],
                    'max_iter' : [400, 500],
                    }
                if model == "DNN":
                    params = {
                    'hidden_layer_sizes': [(100, 25), (150, 35), (200,50)],
                    'activation': ['logistic'],
                    'solver': ['adam'],
                    'learning_rate_init': [0.001, 0.01],
                    'max_iter' : [400],
                    }
                if model == "GB":
                    params = {
                    'loss': ['deviance'],
                    'learning_rate': [0.1],
                    'n_estimators': [200],
                    'subsample': [0.8],
                    'criterion': ['friedman_mse', 'squared_error'], 
                    'max_depth' : [3, 6],
                    'max_features' : ['auto'],
                    }
                if model == "LR":
                    params = {
                    'penalty': ['l2'],
                    'C': [1.0, 0.5, 0.2], 
                    'solver': ['lbfgs', 'saga'],
                    'max_iter': [200],
                    'multi_class': ['auto'] ,
                    }
                if model == "RF":
                    params = {
                    'n_estimators': [50, 100],
                    'criterion': ['gini'],
                    'max_features': [None],
                    'bootstrap': [True],
                    }
                if model == "SVM":
                    params = {
                    'C': [1.0, 0.5, 0.2], 
                    'kernel': ['poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto'],
                    'shrinking' : [False, True],
                    }
                
                trainer.grid_search_CV(model, seed, params, name[0], str(data) + "_" + str(seed), 0)
            # trainer.save_result(model, name[0], data)
            
def param_aug(name, data_lst, model_lst, rand_seed, x_train, x_test, y_train, y_test):
    data_list = data_lst
    model_list = model_lst
    
    for data in data_list:
        trainer = train.train(data)
        for model in model_list:
            if model == "KNN":
                params = {
                'n_neighbors': [8, 9, 10],
                'weights': ['distance'],
                'algorithm' : ['ball_tree', 'kd_tree'],
                'p' : [1],
                }
            if model == "MLP":
                params = {
                'hidden_layer_sizes': [(100,), (150,), (200,)],
                'activation': ['logistic'],
                'solver': ['adam'],
                'max_iter' : [400, 500],
                }
            if model == "DNN":
                params = {
                'hidden_layer_sizes': [(100, 25), (150, 35), (200,50)],
                'activation': ['logistic'],
                'solver': ['adam'],
                'learning_rate_init': [0.001, 0.01],
                'max_iter' : [400],
                }
            if model == "GB":
                params = {
                'loss': ['deviance'],
                'learning_rate': [0.1],
                'n_estimators': [200],
                'subsample': [0.8],
                'criterion': ['friedman_mse', 'squared_error'], 
                'max_depth' : [3, 6],
                'max_features' : ['auto'],
                }
            if model == "LR":
                params = {
                'penalty': ['l2'],
                'C': [1.0, 0.5, 0.2], 
                'solver': ['lbfgs', 'saga'],
                'max_iter': [200],
                'multi_class': ['auto'] ,
                }
            if model == "RF":
                params = {
                'n_estimators': [50, 100],
                'criterion': ['gini'],
                'max_features': [None],
                'bootstrap': [True],
                }
            if model == "SVM":
                params = {
                'C': [1.0, 0.5, 0.2], 
                'kernel': ['poly', 'rbf', 'sigmoid'],
                'gamma': ['scale', 'auto'],
                'shrinking' : [False, True],
                }
            folder_name = name[0]
            file_name = str(data) + "_" + str(rand_seed)
            trainer.grid_search_CV_aug(model, rand_seed, params, folder_name, file_name, x_train, x_test, y_train, y_test)
            # trainer.save_result(model, name[0], data)