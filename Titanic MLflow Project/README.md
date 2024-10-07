# MLFlow Introduction
MLflow aims to assist with some difficulties of the Machine Learning lifecycle. 
In this module the following capabilitites of MLflow is demonstrated:
- Create and run experiments such as model tuning.
- Log and track metrics, parameters, and artifacts. 
- Log and register created Models.
- Search MLflow to find experiment runs and/or models that fit certain criteria.
- Load registered models and use them for predictions.
- Use the capabilities of MLProject to create, run and manage machine learning experiments.

# Install MLflow & Run Tracking Server Locally
Follow the steps below to install and run the mlflow traching server locally:

1) Install/Upgrade the required packages and the MLflow package on your local environment using pip as shown
```sh
pip install -r requirements.txt
```
2) cd into this folder 'Titanic MLflow Project'
```sh
cd 'Titanic MLflow Project'
```
3) Run the mlflow tracking server from this folder 'Titanic MLflow Project' on port 5000 or any other preferred port.
```sh
mlflow ui --port 5000
```
* NB! Your mlflow tracking server need to be up and listening when you run the tuning and building scripts in this project. so keep this terminal window open during your experimenting

# A- Model Tuning
The script <code>run_model_tuning.py</code> performs a grid search and cross validation evaluation on the configured classifiers provided inside the file <code>tuning_config.json</code> in the format shown below with the classifier of logistic regression as an example:

```json
"Titanic Logistic Regression Model": {
    "classifier": "logistic_regression",
    "tunning_parameters": {
        "param_grid": {
            "penalty": ["l1", "l2"],
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "solver": ["liblinear", "saga"],
            "class_weight": [null, "balanced"],
            "tol": [1e-4, 1e-3, 1e-2]
        },
        "random_state": 123,
        "cross_validation_splits": 5,
        "scoring": "f1"
    }
}
```
- If you wish to configure a new classifier, you need to perform the following 2 steps below:
1) Add a configuration in the file <code>tuning_config.json</code> containing param_grid that includes the hyperparameters of the grid search, number of cross validation splits and scoring measure.
2) Use the name of your classifier stored in "classifier" to make a new elif statement under line 102 and create your classifier as shown below
<code>clf = LogisticRegression(random_state=random_state, max_iter=10000)</code>

* NB! Note that the tuning script also utilizes a random_state for 2 reasons:
1) Be able to use the same split of training and test subsets from the original datasett for each run and for each classifier. Which make comparing the results between classifiers and runs more accurate
2) Be able to reproduce results for documentation purposes.
```
"random_state": 123,
```
Run the script <code>run_model_tuning.py</code> by following the steps below:
- Open a new terminal and cd into the directory of this project
```sh
cd 'Titanic MLflow Project'
```
- Examine the script <code>run_model_tuning.py</code>, and then run it as shown below in new terminal
```sh
mlflow run . -e model_tuning --experiment-name 'Titanic Tuning'
```
- This will automatically create a dedicated conda environment and store the needed dependencies declated inside <code>conda.yaml</code>
- The script will take a significant amount of time but should not take more that 1 hour to finish, so please be patient since grid search need more time to find the optimal solution based on the given hyperparameters.

- Go to the MLflow Tracking Server http://127.0.0.1:5000 and notice the newly created experiment name 'Titanic Tuning'
- Enter the experiment to view its 5 runs, each configured classifier with its own run.
- Sort by using the metric <code>grid_search_best_f1_score</code> to see the tuned classifier with best performance results.

# B- Model Registery
The script <code>register_tuned_model.py</code> performs a search between all the tuning runs of the previous script and picks the optimal parameters from the classifier that had the best results in the grid search based on the parameter <code>grid_search_best_f1_score</code> then fits the model before logging and registering this model both on the tracking server and on local filesystem.

Run the script <code>register_tuned_model.py</code> by following the steps below:
- Open a new terminal and cd into the directory of this project
```sh
cd 'Titanic MLflow Project'
```
- Examine both the custom model in <code>custom_model.py</code> and the code in <code>register_tuned_model.py</code>
- Run the script <code>register_tuned_model.py</code> as shown below in new terminal
```sh
mlflow run . -e model_building --experiment-name 'Titanic Model Building'
```
- Go to the MLflow Tracking Server http://127.0.0.1:5000 and notice the newly created model and its building experiment.
- Examine the model and its building experiment to view key metrics and information about this model
- Rerun the script <code>register_tuned_model.py</code> multiple times and notice how the model versioning is happening automatically in the MLflow UI and how it is increasing by 1 each time you run build a new model.
- But don't worry older versions are also still available for retrieval and use for later.

# C- Search, Load & Predict using a Model
The script <code>predict_using_local_model.py</code> loads the model created and stored on local filesystem in the directory <code>local_models</code>.<br/>
The script <code>predict_using_latest_model.py</code> search for latest version of the registered model in the tracking service, then loads it and make a prediction from it.

Run both the scripts by following the steps below:
- Examine the script <code>predict_using_local_model.py</code>, and then run it
```sh
mlflow run . -e local_model_prediction --experiment-name 'Titanic Model Prediction'
```
- Examine the output of this script

- Examine the script <code>predict_using_latest_model.py</code>, and then run it
```sh
mlflow run . -e tracked_model_prediction --experiment-name 'Titanic Model Prediction'
```
- Examine the output of this script

* Note: You can also run the entire flow described above by using this one command:
```sh
mlflow run . --experiment-name 'Titanic MLProject'
```
* Also, if you need to run the scripts from inside the automatically created conda environment you can search for the name of this conda env using the command below:
```sh
conda env list
```
* Pick the conda env indexed with <code>mlflow-...</code>, activate the environment and then run the scripts using python:
```sh
conda activate mlflow-...
```
```sh
python run_model_tuning.py --tuning_config 'tuning_config.json'
```
```sh
python register_tuned_model.py
```
```sh
python predict_using_latest_model.py
```
```sh
python predict_using_local_model.py
```