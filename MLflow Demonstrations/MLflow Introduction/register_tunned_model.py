from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import mlflow

# Set Tracking Server
mlflow.set_tracking_uri("http://localhost:5000")

# Create or Set the experiment 
mlflow.set_experiment(experiment_name="Titanic Model Building")

# Enable auto logging
mlflow.sklearn.autolog(disable=False)

# Search for experiment runs
runs = mlflow.search_runs(
    experiment_names=['Titanic Tunning'], 
    order_by=["metrics.f1 ASC"],
    max_results=1
)

if len(runs) > 0:
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
    X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
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
    model_name = "TitanicDecisionTreeModel"
    # Convert data to a DataFrame for logging
    input_example = pd.DataFrame(X[0:2])
    # Log the model
    mlflow.sklearn.log_model(
        sk_model=dt_clf,
        artifact_path="artifacts",
        input_example=input_example,
        registered_model_name=model_name
    )
        
    ## Log the training code in artifacts
    # List of files to log as artifacts
    artifact_files = ["run_model_tunning.py", "register_tunned_model.py"]

    for artifact_file in artifact_files:
        mlflow.log_artifact(artifact_file)

else:
    print("Did not find any registered runs of tunning experiments")

