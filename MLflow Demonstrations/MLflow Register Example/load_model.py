from sklearn.model_selection import train_test_split
import mlflow.sklearn

# Load the Production stage of Insurance model using scikit-learn flavor
model = mlflow.sklearn.load_model("models:/Insurance/Production")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run prediction on our test data
model.predict(X_test)