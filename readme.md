# How to Use?
This is a project for the 2023 study of the Sensorium research project. \\
All copyrights belong to Sanghoon Lee, Ji-yeong Chae. \\
For questions, please email leesh2913@dgist.ac.kr.

# preprocess.py
process raw file to **int** csv file   
csv file is saved at **data/x_data_ .csv** and **data/y_data_ .csv**   
## pre_process_00
save only frequency (0 ~ 5120hz data)   
data name is **data/x_data_00.csv**, **data/y_data_00.csv**

## pre_process_01
frequency (0 ~ 5120hz data) + Pipe info (p pd s type data)
data name is **data/x_data_01.csv**, **data/y_data_01.csv**

# train.py
## 1. import train.py
```
import train
```
## 2. Initialize train class   
```
trainer = train.train(name)
``` 
name is the file name of **/data folder**
example: "00", "01"   

## 3. search_parameter(model_name, rand_seed, parameter, name)   
```
params = {'n_neighbors': [5, 15, 30, 100], 'weights':  ['uniform', 'distance'],
          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

trainer.search_parameter("KNN", 3, param, "First test")
```
- **model_name** is the name of ML methods   
    examples: "KNN", "MLP", "DNN", "GB", etc ...

- **rand_seed** is the seed of randome state
    examples: 0, 42, 1040, etc ...

- **parameter** is the dictionary of parameter set
    examples: 
    ```
    params = {
        'hidden_layer_sizes': [(50,), (100,), (150,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['adam'],
        'alpha': [0.001, 0.01, 0.1]
    }
    ```
    You can check the parameter settings in **parameter_list.txt**

- **name** is the file name that you want to save
    examples: 0234   
    If **model_name** is MLP and **name** is "All" then   
    the result of GridSearchCV is saved at **log/MLP_ALL.csv**

# Parameter_list.txt
Save the possible parameter list of any M.L methods.
