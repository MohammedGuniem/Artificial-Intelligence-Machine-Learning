from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow.sklearn
import pandas as pd
import mlflow

## Set Tracking Server
mlflow.set_tracking_uri("http://localhost:5000")

## Create or Set the experiment 
mlflow.set_experiment(experiment_name="Titanic Experiments")

## Tag new experiment
mlflow.set_experiment_tag(key="version", value="1.0")

## Set Auto logging for Scikit-learn flavor 
mlflow.sklearn.autolog()
# You can also log all metrics (Not just those for Scikit-learn flavor) by using
#mlflow.autolog()

## Load dataset
df = pd.read_csv('data/titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

## Start an MLflow run
with mlflow.start_run() as run:

    ## Randomly split data
    test_size = 0.2
    random_state = 99
    # Log the values of split metrics split_test_size, split_random_state
    mlflow.log_metric("split_test_size", test_size) 
    mlflow.log_metric("split_random_state", random_state)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    ## Define the parameter grid
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 15, 25, 35, 45],
        'min_samples_leaf': [1, 3],
        'max_leaf_nodes': [10, 20, 35, 50]
    }

    ## Create a Decision Tree classifier
    clf = DecisionTreeClassifier()

    ## Perform Grid Search with cross-validation
    cv = 5
    scoring='f1'
    grid_search = GridSearchCV(
        estimator=clf, 
        param_grid=param_grid, 
        cv=cv, 
        scoring=scoring
    )

    ## Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Print the number of unique hyperparameter combinations and the total runs
    total_combinations = len(grid_search.cv_results_['params'])
    total_runs = total_combinations * grid_search.cv
    print(f"Total unique hyperparameter combinations: {total_combinations}")
    print(f"Total number of models trained (considering CV folds): {total_runs}")

    ## Get the best parameters from grid search
    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")

    ## Get model with best model and log its evaluation metrics
    best_clf = grid_search.best_estimator_
    # Predict with best model
    y_pred = best_clf.predict(X_test)
    # Get Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    # Get Precision Score
    precision = precision_score(y_test, y_pred)
    # Get Recall Score
    recall = recall_score(y_test, y_pred)
    # Get F1 Score
    f1 = f1_score(y_test, y_pred)
    # Get Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    # Extract and log false negatives, true negatives, false positives, true positives
    tn, fp, fn, tp = cm.ravel()
    cm_metrics = {
        "evaluation_true_negatives": tn,
        "evaluation_false_positives": fp,
        "evaluation_false_negatives": fn,
        "evaluation_true_positives": tp
    }

    ## Log parameters
    # Log the parameter cv that stores how many k-fold of data the tunning performed on.
    mlflow.log_param("cv", cv)
    # Log the parameter scoring_measure that stores what scoring measure is used to find the optimal tunned model.
    mlflow.log_param("scoring_measure", scoring)
    # Log multiple best tunning parameters at once
    mlflow.log_params(best_params)

    # Log the value of the evaluation metrics from the best classifer
    mlflow.log_metric("evaluation_accuracy", accuracy)
    mlflow.log_metric("evaluation_precision", precision)
    mlflow.log_metric("evaluation_recall", recall)
    mlflow.log_metric("evaluation_f1_score", f1)
    # Log the values of the confusion matrix from best classifer all 4 at once
    mlflow.log_metrics(cm_metrics)

    # Get the run_id
    run_id = run.info.run_id
    print(f"RUN_ID: {run_id}")
    print(f"Best F1 Score: {f1}")