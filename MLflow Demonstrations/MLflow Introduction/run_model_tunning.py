from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold
from sklearn.tree import DecisionTreeClassifier
import mlflow.sklearn
import pandas as pd
import mlflow

# Set Tracking Server
mlflow.set_tracking_uri("http://localhost:5000")

# Create or Set the experiment 
mlflow.set_experiment(experiment_name="Titanic Tunning")

# Tag new experiment
mlflow.set_experiment_tag(key="experiment_version", value="1.0")
mlflow.set_experiment_tag(key="learning_type", value="classification")

# You can set auto logging for all metrics by using
#mlflow.autolog()

# You can also set auto logging for Scikit-learn flavor by using
#mlflow.sklearn.autolog()

# Disable autologging
mlflow.autolog(disable=True)

# Load dataset
df = pd.read_csv('data/titanic.csv')

# Create male column with 1 for male and 0 for female
df['male'] = df['Sex'] == 'male'

# Extract features
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
# Extract corresponding targets
y = df['Survived'].values

## Start a New MLflow Run
with mlflow.start_run() as run:

    # Log the distribution of target classes in dataset
    # Log the size of dataset
    dataset_size = len(df)
    print("dataset_size: ", dataset_size)
    mlflow.log_param("dataset_size", dataset_size)
    
    # Log unique target classes
    unique_target_classes = df['Survived'].unique().tolist()
    print("unique_target_classes: ", unique_target_classes)
    mlflow.log_param("unique_target_classes", unique_target_classes)

    # Log the number of unique target classes
    num_target_classes = df['Survived'].value_counts().tolist()
    print("num_target_classes: ", num_target_classes)
    mlflow.log_param("num_target_classes", num_target_classes)

    # Log the percentage of unique target classes
    percentage_target_classes = [(num_target_class / dataset_size) * 100 for num_target_class in num_target_classes]
    print("percentage_target_classes: ", percentage_target_classes)
    mlflow.log_param("percentage_target_classes", percentage_target_classes)

    # Define the tunning parameter grid
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 15, 25, 35, 45],
        'min_samples_leaf': [1, 3],
        'max_leaf_nodes': [10, 20, 35, 50]
    }

    # Set and log random_state to reproduce results
    random_state = 123
    print("random_state: ", random_state)
    mlflow.log_param("random_state", random_state)

    # Create a Decision Tree classifier with random_state
    clf = DecisionTreeClassifier(random_state=random_state)

    # Perform grid search with cross-validation
    cv = 5
    scoring='f1'
    # Set up GridSearchCV with random_state in the cv strategy
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=clf, 
        param_grid=param_grid,
        scoring=scoring,
        cv=cv_strategy
    )

    ## Fit the grid search to the data
    grid_search.fit(X, y)

    ## Log Parameters
    # Log the parameter cv that stores how many k-fold of data the tunning performed on
    print("cross_validation_splits: ", cv)
    mlflow.log_param("cross_validation_splits", cv)
    
    # Log the configured tunning parameters
    for parameter, value in param_grid.items():
        print(f"{parameter}: ", value)
    mlflow.log_params(param_grid)

    # Get the best tunning parameters from grid search
    best_params = grid_search.best_params_
    # Log the best tunning parameters with prefix "optimal_tunned_"
    prefix = "optimal_tunned"
    for parameter, value in best_params.items():
        print(f"{prefix}_{parameter}: ", value)
        mlflow.log_param(f"{prefix}_{parameter}", value)

    ## Log metrics
    # Log the best scoring of grid search
    gs_best_score = grid_search.best_score_
    print(f"grid_search_best_{scoring}_score", gs_best_score)
    mlflow.log_metric(f"grid_search_best_{scoring}_score", gs_best_score)
    
    # Get the best model
    best_model = grid_search.best_estimator_

    # Perform cross_val_predict with the same random_state in the cv strategy
    y_pred = cross_val_predict(best_model, X, y, cv=cv_strategy)
    
    metrics = {}

    # Calculate accuracy score
    accuracy = accuracy_score(y, y_pred)
    metrics["accuracy"] = accuracy

    # Calculate recall score
    recall = recall_score(y, y_pred, average='macro')
    metrics["recall"] = recall

    # Calculate precision score
    precision = precision_score(y, y_pred, average='macro')
    metrics["precision"] = precision

    # Calculate f1 score
    f1 = f1_score(y, y_pred, average='macro')
    metrics["f1"] = f1

    # Get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    metrics["cm_true_positives"] = tp
    metrics["cm_false_negatives"] = fn
    metrics["cm_false_positives"] = fp
    metrics["cm_true_negatives"] = tn

    # Print evaluation metrics
    for metric, value in metrics.items():
        print(f"{metric}: ", value)

    # Log evaluation metrics
    mlflow.log_metrics(metrics)

    ## Log the code using for tunning in artifacts
    mlflow.log_artifact("run_model_tunning.py")

    # Get and log the ID of this run as parameter
    run_id = run.info.run_id
    mlflow.log_param("run_id", run_id)
    print(f"run_id: {run_id}")