import train

def param_0131(name, data_lst, model_lst, rand_seed):
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
                    'activation': ['logistic', 'relu', 'identity'],
                    'solver': ['adam'],
                    'max_iter' : [400, 500],
                    }
                if model == "DNN":
                    params = {
                    'hidden_layer_sizes': [(100, 25), (150, 35), (200,50)],
                    'activation': ['logistic'],
                    'solver': ['adam'],
                    'learning_rate_init': [0.0001, 0.001],
                    'max_iter' : [400, 500],
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
                    'max_iter': [100, 200],
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
                
                trainer.search_parameter(model, seed, params, name[0], str(data) + "_" + str(seed), 0)
            # trainer.save_result(model, name[0], data)