import train

def param_0122():
    data_list = ["00", "01"]
    model_list = ["KNN", "MLP", "DNN", "GB", "LR", "RF", "SVM"]
    # model_list = ["GB"]
    
    for data in data_list:
        trainer = train.train(data)
        for model in model_list:
            for seed in range(2):
                if model == "KNN":
                    params = {
                    'n_neighbors': [5, 10, 20, 40],
                    'weights': ['uniform', 'distance'],
                    'algorithm' : ['auto', 'ball_tree', 'kd_tree'],
                    'leaf_size' : [30, 60, 120],
                    'p' : [1, 2],
                    }
                if model == "MLP":
                    params = {
                    'hidden_layer_sizes': [(50,), (100,), (200,), (400)],
                    'activation': ['tanh', 'relu', 'logistic', 'identity'],
                    'solver': ['adam', 'bfgs', 'sgd'],
                    'alpha': [0.001, 0.01, 0.1],
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'learning_rate_init': [0.0001, 0.001, 0.01],
                    'max_iter' : [300, 400],
                    }
                if model == "DNN":
                    params = {
                    'hidden_layer_sizes': [(50, 10), (100, 25), (200,50), (400, 100)],
                    'activation': ['tanh', 'relu', 'logistic', 'identity'],
                    'solver': ['adam', 'bfgs', 'sgd'],
                    'alpha': [0.001, 0.01, 0.1],
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'learning_rate_init': [0.0001, 0.001, 0.01],
                    'max_iter' : [300, 400],
                    }
                if model == "GB":
                    params = {
                    'loss': ['log_loss', 'deviance', 'exponential'],
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.2, 0.4, 0.8, 1.0],
                    'criterion': ['friedman_mse', 'squared_error'], 
                    'max_depth' : [3, 6, 12, 24],
                    'max_features' : ['auto', 'sqrt', 'log2'],
                    }
                if model == "LR":
                    params = {
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'C': [1.0, 0.5, 0.2], 
                    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                    'max_iter': [100, 200, 400],
                    'multi_class': ['auto', 'ovr', 'multinomial'] ,
                    }
                if model == "RF":
                    params = {
                    'n_estimators': [10, 50, 100],
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False],
                    }
                if model == "SVM":
                    params = {
                    'C': [1.0, 0.5, 0.2], 
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                    'gamma': ['scale', 'auto'],
                    'shrinking' : [False, True],
                    }
                
                trainer.search_parameter(model, seed, params, "_0122_" + str(data) + "_" + str(seed))
                

def param_0124(name):
    data_list = ["01"]
    model_list = ["KNN", "MLP", "DNN", "GB", "LR", "RF"]
    # model_list = ["SVM"]
    
    for data in data_list:
        trainer = train.train(data)
        for model in model_list:
            for seed in range(50):
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
                    'loss': ['deviance', 'exponential'],
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [50, 100, 200],
                    'subsample': [0.2, 0.4, 0.8, 1.0],
                    'criterion': ['friedman_mse', 'squared_error'], 
                    'max_depth' : [3, 6, 12, 24],
                    'max_features' : ['auto', 'sqrt', 'log2'],
                    }
                if model == "LR":
                    params = {
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'C': [1.0, 0.5, 0.2], 
                    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                    'max_iter': [100, 200, 400],
                    'multi_class': ['auto', 'ovr', 'multinomial'] ,
                    }
                if model == "RF":
                    params = {
                    'n_estimators': [10, 50, 100],
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False],
                    }
                if model == "SVM":
                    params = {
                    'C': [1.0, 0.5, 0.2], 
                    'kernel': ['poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto'],
                    'shrinking' : [False, True],
                    }
                
                trainer.search_parameter(model, seed, params, name + "_" + str(data) + "_" + str(seed), 0)
            trainer.save_result(model, name, data)


def param_0126(name):
    data_list = ["02"]
    model_list = ["KNN", "MLP", "DNN", "GB", "LR", "RF", "SVM"]
    # model_list = ["SVM"]
    
    for data in data_list:
        trainer = train.train(data)
        for model in model_list:
            for seed in range(10):
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
                
                trainer.search_parameter(model, seed, params, name + "_" + str(data) + "_" + str(seed), 0)
            trainer.save_result(model, name, data)

# Param of 0127
def param_0127(name, data_lst, model_lst, rand_seed):
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
                    'hidden_layer_sizes': [(100,)],
                    'activation': ['logistic'],
                    'solver': ['adam'],
                    'max_iter' : [500],
                    }
                if model == "DNN":
                    params = {
                    'hidden_layer_sizes': [(100, 25), (150, 35)],
                    'activation': ['logistic'],
                    'solver': ['adam'],
                    'max_iter' : [500],
                    }
                if model == "GB":
                    params = {
                    'loss': ['deviance'],
                    'learning_rate': [0.1],
                    'n_estimators': [100, 200],
                    'subsample': [0.8, 1],
                    'criterion': ['friedman_mse', 'squared_error'], 
                    'max_depth' : [3, 6],
                    'max_features' : ['auto'],
                    }
                if model == "LR":
                    params = {
                    'penalty': ['l2', 'l1'],
                    'C': [0.2], 
                    'solver': ['lbfgs', 'liblinear'],
                    'max_iter': [100, 200],
                    'multi_class': ['auto'] ,
                    }
                if model == "RF":
                    params = {
                    'n_estimators': [100],
                    'criterion': ['gini', 'entropy'],
                    'max_features': [None],
                    'bootstrap': [True],
                    }
                if model == "SVM":
                    params = {
                    'C': [1.0], 
                    'kernel': ['poly'],
                    'gamma': ['auto'],
                    'shrinking' : [False],
                    }
                
                trainer.search_parameter(model, seed, params, name[0] + "_" + str(data) + "_" + str(seed), 0)
            trainer.save_result(model, name[0], data)
            
# Param of 0128
def param_0128(name, data_lst, model_lst, rand_seed):
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
                    'hidden_layer_sizes': [(100,), (150,)],
                    'activation': ['logistic'],
                    'solver': ['adam'],
                    'max_iter' : [400, 500],
                    }
                if model == "DNN":
                    params = {
                    'hidden_layer_sizes': [(100, 25), (200,50)],
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
                    'C': [1.0, 0.2], 
                    'kernel': ['poly'],
                    'gamma': ['scale', 'auto'],
                    'shrinking' : [False, True],
                    }
                
                trainer.search_parameter(model, seed, params, name[0] + "_" + str(data) + "_" + str(seed), 0)
            trainer.save_result(model, name[0], data)