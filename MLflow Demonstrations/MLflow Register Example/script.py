from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlflow import MlflowClient
import mlflow.sklearn
import pandas as pd
import numpy as np
import mlflow

# Training data
df = pd.read_csv('insurance.csv')
df['gender'] = df['sex'] == 'male'
df['is_smoker'] = np.where(df['smoker'] == 'yes', 1, 0)
X = df[["age", "gender", "bmi", "children", "charges"]] #not taking into accound "region" since there need encoding and i want to keep things simple for demonstration purposes
y = df[["is_smoker"]]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set Auto logging for Scikit-learn flavor
mlflow.sklearn.autolog()

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Create an instance of MLflow Client Class named client
client = MlflowClient()

# Create new model
#client.create_registered_model("Insurance")

# Registering existing models
# mlflow.register_model("model_2022", "Insurance")

# Registering new models
#mlflow.sklearn.log_model(lr_model, "model", registered_model_name="Insurance")

# Insurance filter string
insurance_filter_string = "name LIKE 'Insurance%'"

# Search for Insurance models
print("Search results for 'Insurance models'")
print(client.search_registered_models(filter_string=insurance_filter_string))

# Not Insurance filter string
not_insurance_filter_string = "name != 'Insurance'"

# Search for non Insurance models
print("Search results for 'non Insurance models'")
print(client.search_registered_models(filter_string=not_insurance_filter_string))

# Transition version 2 of Insurance model to stable stage
#client.transition_model_version_stage(name="Insurance", version=1, stage="Production")

# Serve model using the CL command below
# mlflow models serve -m "models:/Insurance/Production"