import mlflow

# Set Tracking Server
mlflow.set_tracking_uri("http://localhost:5000")

# Create a filter string for accuracy score smaller than or equal to 0.70
accuracy_filter_string = "metrics.accuracy <= .60 and status = 'FINISHED'"

# Search runs
search_results = mlflow.search_runs(
    experiment_names=["Diabetes Experiments", "More Diabetes Experiments"], 
    filter_string=accuracy_filter_string, 
    order_by=["metrics.accuracy DESC"]
)

print(search_results)

