from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import mlflow.sklearn

model_uri = "runs:/d9c325bc43a5432da9f324d62a325aa9/model"
# Load model
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