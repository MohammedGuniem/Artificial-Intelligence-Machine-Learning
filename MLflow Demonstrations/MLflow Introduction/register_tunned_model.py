from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import mlflow

## Load dataset
df = pd.read_csv('data/titanic.csv')
df['male'] = df['Sex'] == 'male'

# Convert integer columns with missing values to float64
integer_columns = ["Survived", "Pclass", "male", "Age", "Siblings/Spouses", "Parents/Children"]
for int_col in integer_columns:
    df[int_col] = df[int_col].astype('float64')

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

# Create and fit the model with tunned parameters from run_model_tunning.py
model = DecisionTreeClassifier(
    criterion='gini', 
    max_depth=35, 
    max_leaf_nodes=35, 
    min_samples_leaf=1
)
model.fit(X, y)

## Set Tracking Server
mlflow.set_tracking_uri("http://localhost:5000")

## Register the model
model_name = "TitanicDecisionTreeModel"
# Convert data to a DataFrame for logging
input_example = pd.DataFrame(X[0:2])
# Log the model
mlflow.sklearn.log_model(
    model,
    "model",
    input_example=input_example,
    registered_model_name=model_name
)
    
## Log the training code in artifacts
# List of files to log as artifacts
artifact_files = ["run_model_tunning.py", "register_tunned_model.py"]

for artifact_file in artifact_files:
    mlflow.log_artifact(artifact_file)