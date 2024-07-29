import mlflow
import mlflow.sklearn
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Training data
df = pd.read_csv('Salary_predict.csv')
X = df[["experience", "age", "interview_score"]]
y = df[["Salary"]]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

# Set Auto logging for Scikit-learn flavor
mlflow.sklearn.autolog()
# Parameters
run_id = str(sys.argv[1])
print(run_id)

# Train model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)