# MLFlow Introduction
MLflow aims to assist with some difficulties of the Machine Learning lifecycle. 
In this module the following capabilitites of MLflow is demonstrated:
1) Create and run experiments such as model tuning.
2) Log and track metrics, parameters, and artifacts. 
3) Log and register created Models.
4) Search MLflow to find experiment runs and/or models that fit certain criteria.
5) Load registered models and use them for predictions.

# Install MLflow & Run Tracking Server Locally
- Install/Upgrade MLflow on your local environment using pip
```sh
pip install -U mlflow
```
- cd into this folder 'MLflow Introduction'
```sh
cd '.\MLflow Fundamentals & Demonstrations\MLflow Introduction\'
```
- Run the mlflow tracking server from this folder 'MLflow Introduction' on port 5000 or any other preferred port.
```sh
mlflow ui --port 5000
```

# A- Model Tuning
The script <code>run_model_tuning.py</code> performs a grid search and cross validation on a decision tree classifier using the following configured parameters.

```json
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 15, 25, 35, 45],
    'min_samples_leaf': [1, 3],
    'max_leaf_nodes': [10, 20, 35, 50]
}
```
The script also utilizes a random_state to be able to reproduce results for documentation purposes.
```
random_state = 123
```
Run the script <code>run_model_tuning.py</code> by following the steps below:
- Inside the same folder 'MLflow Introduction'
- Examine the script <code>run_model_tuning.py</code>, and then run it as shown below in new terminal
```sh
python run_model_tuning.py
```
- Go to the MLflow Tracking Server http://127.0.0.1:5000 and notice the newly created experiment
- Enter the experiment to view its runs
- Change the random_state in the script <code>run_model_tuning.py</code> to another number of your choice
- Run the script above again and compare between the different models using their metrics

# B- Model Registery
The script <code>register_tuned_model.py</code> performs a search between all the tuning runs of the previous script and picks the optimal parameters to train the classifier then fits the model before logging and registering this model both on the tracking server and on local filesystem.

Run the script <code>register_tuned_model.py</code> by following the steps below:
- Inside the same folder 'MLflow Introduction'
- Examine both the custom model in <code>custom_model.py</code> and the code in <code>register_tuned_model.py</code>
- Run the script <code>register_tuned_model.py</code> as shown below in new terminal
```sh
python register_tuned_model.py
```
- Go to the MLflow Tracking Server http://127.0.0.1:5000 and notice the newly created model and its building experiment.
- Examine the model and its building experiment to view key metrics and information about this model
- Rerun the script <code>register_tuned_model.py</code> multiple time and notice how the model versioning is happening automatically in the MLflow UI and how it is increasing by 1 each time you run the script

# C- Search, Load & Predict using a Model
The script <code>predict_using_local_model.py</code> loads the model created and stored on local filesystem in the directory <code>local_models</code>.<br/>
The script <code>predict_using_latest_model.py</code> seach for latest version of the registered model in the trackign service, then loads it and make a prediction from it.

Run both the scripts by following the steps below:
- Examine the script <code>predict_using_local_model.py</code>, and then run it
```sh
python predict_using_local_model.py
```
- Examine the output of this script

- Examine the script <code>predict_using_latest_model.py</code>, and then run it
```sh
python predict_using_latest_model.py
```
- Examine the output of this script