import mlflow


# Activate the model tuning process using its entry point
model_tuning = mlflow.projects.run(
    # Set the URI as the current working directory
    uri='./',
    # Set the entry point to model_tuning as written in the MLProject configuration file
    entry_point ='model_tuning',
    # Set the experiment name as Titanic Tuning
    experiment_name='Titanic Tuning',
    # Specify an environment manager to create a new environment for the run
    env_manager="local",
    # Whether to block while waiting for a run to complete. Defaults to True.
    synchronous=True
)
print("Status(model_tuning): ", model_tuning.get_status())

# Activate the model building process using its entry point
model_building = mlflow.projects.run(
    uri='./',
    entry_point='model_building',
    experiment_name='Titanic Model Building',
    env_manager='local'
)
print("Status(model_building): ", model_building.get_status())