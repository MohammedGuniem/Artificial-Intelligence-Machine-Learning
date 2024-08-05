from sklearn.tree import DecisionTreeClassifier
from custom_model import CustomSklearnModel
import pandas as pd
import mlflow
import shutil
import os


# Set Tracking Server
mlflow.set_tracking_uri("http://localhost:5000")

# This filter insure that we only use model with accuracy higher or equal to 0.70
accuracy_filter = "metrics.accuracy > .70"

# Search for experiment runs
runs = mlflow.search_runs(
    experiment_names=['Titanic Tunning'], 
    filter_string=accuracy_filter,
    order_by=["metrics.grid_search_best_f1_score DESC"],
    max_results=1
)

if len(runs) > 0:
    # Create or Set the experiment 
    mlflow.set_experiment(experiment_name="Titanic Model Building")

    # Tag new experiment
    mlflow.set_experiment_tag(key="experiment_version", value="1.0")
    mlflow.set_experiment_tag(key="learning_type", value="classification")

    # Enable auto logging
    mlflow.sklearn.autolog(disable=False)
    mlflow.autolog(disable=False)

    # Start an MLflow run explicitly
    with mlflow.start_run() as run:
        optimal_run = runs.iloc[0]
        grid_search_best_f1_score = optimal_run["metrics.grid_search_best_f1_score"]
        print(f"Best grid_search_best_f1_score: ", grid_search_best_f1_score)

        # Load dataset
        df = pd.read_csv('data/titanic.csv')

        # Create male column with 1 for male and 0 for female
        df['male'] = df['Sex'] == 'male'

        # Convert integer columns with missing values to float64
        integer_columns = ["Survived", "Pclass", "male", "Age", "Siblings/Spouses", "Parents/Children"]
        for int_col in integer_columns:
            df[int_col] = df[int_col].astype('float64')

        # Extract features
        X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']]
        # Extract corresponding targets
        y = df['Survived'].values

        # Create a Decision Tree classifier with optimal tunned parameters of best run
        # Also, using random_state to reproduce same model for analysis
        dt_clf = DecisionTreeClassifier(
            criterion=optimal_run["params.optimal_tunned_criterion"],
            max_depth=int(optimal_run["params.optimal_tunned_max_depth"]),
            min_samples_leaf=int(optimal_run["params.optimal_tunned_min_samples_leaf"]),
            max_leaf_nodes=int(optimal_run["params.optimal_tunned_max_leaf_nodes"]),
            random_state=int(optimal_run["params.random_state"])
        )

        # Fit the model
        dt_clf.fit(X, y)

        ## Register the model
        # Make Custom model
        custom_model = CustomSklearnModel(dt_clf)
        # Set model name
        model_name = "TitanicDecisionTreeModel"
        # Convert data to a DataFrame for logging
        input_example = pd.DataFrame(X[0:2])
        # Log the model
        mlflow.pyfunc.log_model(
            artifact_path="artifacts",
            python_model=custom_model,
            input_example=input_example,
            registered_model_name=model_name
        )
        
        # Save the model to local filesystem
        model_uri = f"local_models/{model_name}"

        # Check if the model directory exists on local filesystem and delete it if it does
        if os.path.exists(model_uri):
            shutil.rmtree(model_uri)
            print(f"Deleted existing model directory: {model_uri}")

        # Save a custom model to local filesystem
        
        mlflow.pyfunc.save_model(
            python_model=custom_model, 
            path=model_uri
        )
        mlflow.log_artifact(model_uri)
        
        ## Log the training code in artifacts
        # List of files to log as artifacts
        artifact_files = ["run_model_tunning.py", "register_tunned_model.py"]

        for artifact_file in artifact_files:
            mlflow.log_artifact(artifact_file)
        
else:
    print("Did not find any registered runs of tunning experiments! Please run tunning first.")
