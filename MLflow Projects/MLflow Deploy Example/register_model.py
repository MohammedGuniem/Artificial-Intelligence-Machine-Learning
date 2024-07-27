from mlflow.tracking import MlflowClient

client = MlflowClient()
run_id = "38c46d8c839c492d95fcede84ce3da2b"
model_name = "DiabetesLinearRegressionModel"
model_uri = f"runs:/{run_id}/model"

# the log_model registers the model if not it exist from before, 
# if the model is there it creates a new version
# So it automatically does what the code below does.
client.create_registered_model(model_name)
client.create_model_version(model_name, model_uri, run_id)
