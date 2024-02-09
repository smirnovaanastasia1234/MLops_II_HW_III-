import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import mlflow
from mlflow.tracking import MlflowClient
 
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model")

df = pd.read_csv('/home/lnc/project/datasets/train.csv', sep = ';')
X = df.message
Y = df.label
tvec = TfidfVectorizer()
lr = LogisticRegression(solver = "lbfgs")
rfc = RFC(random_state=42)
model = Pipeline([('vectorizer',tvec),('classifier',rfc)])

with mlflow.start_run():
    mlflow.sklearn.log_model(model,
                             artifact_path="lr",
                             registered_model_name="lr")
    mlflow.log_artifact(local_path="/home/lnc/project/scripts/train_model.py",
                        artifact_path="train_model code")
    mlflow.end_run()
model.fit(X,Y)
with open('/home/lnc/project/models/data.pickle', 'wb') as f:
    pickle.dump(model, f)
