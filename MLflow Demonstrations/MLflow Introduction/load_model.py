from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import mlflow.sklearn

# Set Tracking Server
mlflow.set_tracking_uri("http://localhost:5000")

# Create a filter string that picks up the last successful run
accuracy_filter_string = "status = 'FINISHED'"
# Search runs and get last run_id
search_results = mlflow.search_runs(
    experiment_names=["Diabetes Experiments"], 
    filter_string=accuracy_filter_string, 
    order_by=["metrics.accuracy DESC"]
)
run_id = search_results[['run_id']].values[-1][0]

# Load model
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# Load dataset
data = load_diabetes()
X = data.data
y = data.target

# Split the data
random_state=50
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Use the model for prediction
predictions = model.predict(X_test)
print(predictions)