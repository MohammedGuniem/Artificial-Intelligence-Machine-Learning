from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from custom_model import CustomSklearnModel
import pandas as pd
import mlflow
import shutil
import ast
import os


# This filter insure that we only use model with accuracy higher or equal to 0.70
accuracy_filter = "metrics.accuracy > .70"

# Search for experiment runs
runs = mlflow.search_runs(
    experiment_names=['Titanic Tuning'], 
    filter_string=accuracy_filter,
    order_by=["metrics.grid_search_best_f1_score DESC"],
    max_results=1
)

if len(runs) > 0:
    # Tag new experiment
    mlflow.set_experiment_tag(key="experiment_version", value="1.0")
    mlflow.set_experiment_tag(key="learning_type", value="classification")

    # Enable auto logging
    mlflow.sklearn.autolog(disable=False)
    mlflow.autolog(disable=False)

    # Start an MLflow run explicitly
    with mlflow.start_run() as run:
        
        # View the highest score of optimal grid search best f1-score
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

        # Create the optimal classifier with its tuned parameters
        # Also, using random_state to reproduce same model for analysis
        optimal_classifier = optimal_run["params.classifier"]
        if optimal_classifier == "logistic_regression":
            clf = LogisticRegression(
                C=float(optimal_run["params.optimal_tuned_C"]),
                class_weight=optimal_run["params.optimal_tuned_class_weight"],
                penalty=optimal_run["params.optimal_tuned_penalty"],
                solver=optimal_run["params.optimal_tuned_solver"],
                tol=float(optimal_run["params.optimal_tuned_tol"]),
                random_state=int(optimal_run["params.random_state"]),
                max_iter=10000,
            )
        elif optimal_classifier == "decision_tree":
            clf = DecisionTreeClassifier(
                criterion=optimal_run["params.optimal_tuned_criterion"],
                max_depth=int(optimal_run["params.optimal_tuned_max_depth"]),
                min_samples_leaf=int(optimal_run["params.optimal_tuned_min_samples_leaf"]),
                max_leaf_nodes=int(optimal_run["params.optimal_tuned_max_leaf_nodes"]),
                random_state=int(optimal_run["params.random_state"])
            )
        elif optimal_classifier == "random_forest":
            clf = RandomForestClassifier(
                criterion=optimal_run["params.optimal_tuned_criterion"],
                max_depth=int(optimal_run["params.optimal_tuned_max_depth"]),
                min_samples_leaf=int(optimal_run["params.optimal_tuned_min_samples_leaf"]),
                max_leaf_nodes=int(optimal_run["params.optimal_tuned_max_leaf_nodes"]),
                max_features=float(optimal_run["params.optimal_tuned_max_features"]) if optimal_run["params.optimal_tuned_max_features"] != "None" else None,
                n_estimators=int(optimal_run["params.optimal_tuned_n_estimators"]),
                random_state=int(optimal_run["params.random_state"])
            )
        elif optimal_classifier == "mlp_classifier":
            clf = MLPClassifier(
                hidden_layer_sizes=ast.literal_eval(optimal_run["params.optimal_tuned_hidden_layer_sizes"]),
                activation=optimal_run["params.optimal_tuned_activation"],
                solver=optimal_run["params.optimal_tuned_solver"],
                alpha=float(optimal_run["params.optimal_tuned_alpha"]),
                learning_rate=optimal_run["params.optimal_tuned_learning_rate"],
                random_state=int(optimal_run["params.random_state"]), 
                max_iter=10000
            )

        # Fit the model
        clf.fit(X, y)

        ## Register the model
        # Make Custom model
        custom_model = CustomSklearnModel(clf)
        # Set model name
        model_name = "Titanic Model"
        # Convert data to a DataFrame for logging
        input_example = pd.DataFrame(X[0:2])
        # Log and register the tuned model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=custom_model,
            input_example=input_example,
            registered_model_name=model_name
        )

        # Tag model build run with its classifier type
        mlflow.log_param("model_type", optimal_classifier)
        
        # Save the model to local filesystem
        model_uri = f"local_models/{model_name}"

        # Check if the model directory exists on local filesystem and delete it if it does
        if os.path.exists(model_uri):
            shutil.rmtree(model_uri)
            print(f"Deleted existing model directory: {model_uri} in order to update the model")

        # Save a custom model to local filesystem
        mlflow.pyfunc.save_model(
            python_model=custom_model, 
            path=model_uri
        )
        mlflow.log_artifact(model_uri)
        
        ## Log the training code in artifacts
        # List of files to log as artifacts
        artifact_files = [
            "conda.yaml",
            "custom_model.py",
            "MLproject",
            "predict_using_latest_model.py",
            "predict_using_local_model.py",
            "README.md",
            "register_tuned_model.py", 
            "run_model_tuning.py", 
            "script.py",
            "tuning_config.json",
        ]

        for artifact_file in artifact_files:
            mlflow.log_artifact(artifact_file)
        
else:
    print("Did not find any registered runs of tuning experiments! Please run tuning first.")
