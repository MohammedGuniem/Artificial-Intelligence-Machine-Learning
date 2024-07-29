import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression

# Training data
df = pd.read_csv('Salary_predict.csv')
X = df[["experience", "age", "interview_score"]]
y = df[["Salary"]]

# Set Auto logging for Scikit-learn flavor
mlflow.sklearn.autolog()

# Train model
lr = LinearRegression()
lr.fit(X, y)