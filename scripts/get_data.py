import wget
import mlflow
from mlflow.tracking import MlflowClient
 
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_data")

with mlflow.start_run():
   url = 'https://raw.githubusercontent.com/Kozhedu/SpamDetecting/main/traindata1.csv'
   wget.download(url, '/home/lnc/project/datasets/dataset.csv')
   mlflow.log_artifact(local_path="/home/lnc/project/scripts/get_data.py",
                        artifact_path="get_data code")
   mlflow.end_run()
