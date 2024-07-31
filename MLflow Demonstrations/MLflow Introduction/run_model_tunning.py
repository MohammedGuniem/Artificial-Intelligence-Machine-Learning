from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix
import mlflow.sklearn
import pandas as pd
import mlflow


## Load dataset
df = pd.read_csv('data/titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

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

## Start an MLflow run
with mlflow.start_run() as run:
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
    grid_search.fit(X, y)

    ## Get the best parameters from grid search
    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")

    ## Log parameters
    # Log the parameter random_state as "random_state"
    mlflow.log_param("cv", cv)
    # Log the parameter test_size as "test_split_size"
    mlflow.log_param("scoring", scoring)
    # Log multiple best tunning parameters at once
    mlflow.log_params(best_params)

    # Train model with tunned parameters
    # Get best model
    best_clf = grid_search.best_estimator_

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Make predictions
    y_pred = best_clf.predict(X_test)

    # Log metrics
    # Evaluate the model
    score_of_tunned_model = f1_score(y_test, y_pred)
    print(f'score_of_tunned_model: {score_of_tunned_model:.2f}')
    # Log the accuracy score metric with key "accuracy"
    mlflow.log_metric("score_of_tunned_model", score_of_tunned_model)

    # Get Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    # Extract and log false negatives, true negatives, false positives, true positives
    tn, fp, fn, tp = cm.ravel()
    cm_metrics = {
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp
    }
    mlflow.log_metrics(cm_metrics)

    # Get the run_id
    run_id = run.info.run_id
    print(f"RUN_ID: {run_id}")