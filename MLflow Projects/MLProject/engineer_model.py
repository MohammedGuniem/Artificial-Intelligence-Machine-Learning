import mlflow
import mlflow.sklearn
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Training data
df = pd.read_csv('Salary_predict.csv')
X = df[["experience", "age", "interview_score"]]
y = df[["Salary"]]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

# Set Auto logging for Scikit-learn flavor
mlflow.sklearn.autolog()
# Parameters
n_jobs = int(sys.argv[1])
fit_intercept = bool(sys.argv[2])

# Train model
lr = LinearRegression(n_jobs=n_jobs, fit_intercept=fit_intercept)
lr.fit(X_train, y_train)