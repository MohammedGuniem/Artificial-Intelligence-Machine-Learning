import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import mlflow.sklearn
import pandas as pd

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split the data
random_state = 45
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Set Tracking Server
mlflow.set_tracking_uri("http://localhost:5000")

# Create or Set the experiment 
mlflow.set_experiment(experiment_name="Diabetes Experiments")
# Tag new experiment
mlflow.set_experiment_tag(key="version", value="1.0")

# Set Auto logging for Scikit-learn flavor 
mlflow.sklearn.autolog()

# Start an MLflow run
with mlflow.start_run() as run:
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Log metrics
    score = model.score(X_test, y_test)
    # Log the accuracy score metric with key "accuracy"
    mlflow.log_metric("accuracy", score)

    # Log parameters
    # Log the parameter random_state as "random_state"
    mlflow.log_param("random_state", random_state)
    # Log the parameter test_size as "test_split_size"
    mlflow.log_param("test_split_size", test_size)

    # Get the run_id
    run_id = run.info.run_id
    print(f"RUN_ID: {run_id}")

    # Register the model
    model_name = "DiabetesLinearRegressionModel"
    # Convert data to a DataFrame for logging
    input_example = pd.DataFrame(X_train)
    # Log the model
    mlflow.sklearn.log_model(
        model, 
        "model",
        input_example=input_example,
        registered_model_name=model_name
    )

    # Log the training code in artifacts
    mlflow.log_artifact("register_model.py")