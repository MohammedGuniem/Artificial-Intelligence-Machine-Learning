pip install mlflow

cd '.\MLProject Demonstration\'
mlflow ui
# Keep this window open

# In another terminal
cd '.\MLProject Demonstration\'

# If you want to create and use a dedicated conda env
mlflow run . --experiment-name 'Titanic MLProject'
conda activate mlflow-276c26d94868e1a2e2b62412a0d0e65a02e4d948
python .\predict_using_latest_model.py

# Or if you want to use the existing ml conda env
conda activate ml
python script.py
# But you will get the ml project run will not be logged to be viewed on the mlflow ui
python .\predict_using_latest_model.py
python .\predict_using_local_model.py 


mlflow run . -e model_tuning