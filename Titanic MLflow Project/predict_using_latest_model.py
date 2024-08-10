from mlflow.tracking import MlflowClient
import pandas as pd
import mlflow

# Set the tracking URI to your MLflow server or leave it default for local tracking
mlflow.set_tracking_uri("http://localhost:5000")

# Initialize the MLflow client
client = MlflowClient()

# Define the model name you want to search for
model_name = "Titanic Model"

# Try to get the registered model by name
try:
    # Check if the model is registered
    registered_model = client.get_registered_model(model_name)
    print(f"Found model with name: '{model_name}'")
    
    # List all versions of the model
    all_versions = client.search_model_versions(f"name='{model_name}'")
    
    # Get the latest version (by version number)
    latest_version = max(all_versions, key=lambda v: int(v.version)).version
    print(f"Latest version of the model '{model_name}': {latest_version}")
    
    # Load the model
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model loaded successfully from URI: {model_uri}")
    
    # Prepare some sample input data (example)
    # Replace this with your actual input data
    input_data = pd.DataFrame({
        "Pclass": [3.0, 3.0],
        "male": [1.0, 0.0],
        "Age": [22.0, 22.0],
        "Siblings/Spouses": [1.0, 1.0],
        "Parents/Children": [0.0, 1.0],
        "Fare": [7.25, 7.25]
    })
    print("Predicting for: ")
    print(input_data)

    # Make predictions
    predictions = model.predict(input_data)
    print(f"Predictions: {predictions}")

except Exception as e:
    print(e)

