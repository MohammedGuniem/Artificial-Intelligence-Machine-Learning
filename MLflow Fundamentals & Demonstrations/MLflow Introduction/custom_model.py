import mlflow.pyfunc

class CustomModel(mlflow.pyfunc.PythonModel):
    def __init__(self, sklearn_model):
        self.sklearn_model = sklearn_model

    def predict(self, context, model_input):
        # Example of using the sklearn model for prediction
        predictions = self.sklearn_model.predict(model_input)
        decoded_predictions = []  
        for prediction in predictions:
            if prediction == 0:
                decoded_predictions.append("Did Not Survived")
            else:
                decoded_predictions.append("Survived")
        return decoded_predictions

# Example of a custom model class for sklearn
class CustomSklearnModel(CustomModel):
    def __init__(self, sklearn_model):
        super().__init__(sklearn_model)
