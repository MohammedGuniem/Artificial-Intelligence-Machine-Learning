import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import mlflow.sklearn

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split the data
random_state=45
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("diabetes-LR-experiment")

# Start an MLflow run
with mlflow.start_run() as run:
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Log metrics
    score = model.score(X_test, y_test)
    mlflow.log_metric("r2", score)

    # Optionally log parameters
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("test_size", 0.2)

    # Get the run_id
    run_id = run.info.run_id
    print(f"RUN_ID: {run_id}")

    # Register and Log the model
    model_name = "DiabetesLinearRegressionModel"
    mlflow.sklearn.log_model(
        model, 
        "model", 
        registered_model_name=model_name
    )
