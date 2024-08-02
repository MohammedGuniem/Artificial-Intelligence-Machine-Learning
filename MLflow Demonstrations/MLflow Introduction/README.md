# MLFlow Introduction
MLflow aims to assist with some difficulties of the Machine Learning lifecycle. 
In this module the following capabilitites of MLflow is demonstrated:
1) Create experiments.
2) Runs experiments.
3) Log and track metrics, parameters, and artifacts. 
4) Log Created Models.
4) Search MLflow programmatically to find experiment runs that fit certain criteria.

# Demonstration Run
- Install/Upgrade MLflow on your local environment using pip
```sh
    pip install -U mlflow
```
- cd into this folder 'MLflow Introduction'
- Run the mlflow tracking server from this folder 'MLflow Introduction' on port 5000 or any other preferred port.
```sh
    mlflow ui --port 5000
```

## A- register_model.py
- In another terminal and inside the same folder 'MLflow Introduction'
- Examine the script <code>register_model.py</code>, and then run it
```sh
    python register_model.py
```
- Examine the output of logging and registering the model and predicting from it.
- Run the script above a couple more times to notice how the model version is increasing
- Go to the MLflow Tracking Server http://127.0.0.1:5000 and examine the Ui and its content
- You may need to refresh the page in order to update the content of the tracking server.

### B- search_model.py
- Examine the script <code>search_model.py</code>, and then run it
```sh
    python search_model.py
```
- Examine the output of the search results

### C- load_model.py
- Examine the script <code>load_model.py</code>, and then run it
```sh
    python load_model.py
```
- Examine the output of loading the model and predicting from it.