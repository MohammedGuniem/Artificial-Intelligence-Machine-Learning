import mlflow

# Set the run function from the MLflow Projects module
mlflow.projects.run(
  	# Set the URI as the current working directory
    uri='./',
    # Set the entry point to main
    entry_point ='main',
    # Set the experiment name as Salary
    experiment_name='Salary',
    env_manager="local",
    synchronous=True,
)

# Step 1 - example: Model Engineering
# Set run method to model_engineering
model_engineering = mlflow.projects.run(
    uri='./',
    # Set entry point to model_engineering
    entry_point='model_engineering',
    experiment_name='Salary',
    # Set the parameters for n_jobs and fit_intercept
    parameters={
        'n_jobs': 2, 
        'fit_intercept': False
    },
    env_manager='local'
)

# Set Run ID of model training to be passed to Model Evaluation step
model_engineering_run_id = model_engineering.run_id
print(model_engineering_run_id)

# Step 2 - example: Model Evaluation
# Set the MLflow Projects run method
model_evaluation = mlflow.projects.run(
    uri="./",
    # Set the entry point to model_evaluation
    entry_point="model_evaluation",
  	# Set the parameter run_id to the run_id output of previous step
    parameters={
        "run_id": model_engineering_run_id,
    },
    env_manager="local"
)

print(model_evaluation.get_status())

# We can also run this project using command line using the following command
# mlflow run --entry-point main --experiment-name "Salary" --env-manager local -P n_jobs=3 -P fit_intercept=True ./
