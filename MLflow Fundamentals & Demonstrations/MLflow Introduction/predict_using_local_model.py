import os
import mlflow
import pandas as pd

# Define the model name you want to load from local file system
model_name = "Titanic Model"
model_uri = f"local_models/{model_name}"

# Check if the model directory exists on local filesystem
if os.path.exists(model_uri):
    # Load model from local filesystem
    model_uri = f"local_models/{model_name}"
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

    # Try to get the saved model
    try:
        print(f"Model loaded successfully from this path: {model_uri} on local filesystem")

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
        predictions = loaded_model.predict(input_data)
        print(f"Predictions: {predictions}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

else:
    print(f"Model directory {model_uri} was not found on local filesystem")