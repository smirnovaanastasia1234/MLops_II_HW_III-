import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


df = pd.read_csv('/home/lnc/project/datasets/test.csv', sep = ';')
X = df.message
Y = df.label
tvec = TfidfVectorizer()
lr = LogisticRegression(solver = "lbfgs")
rfc = RFC(random_state=42)
model = Pipeline([('vectorizer',tvec),('classifier',rfc)])
with open('/home/lnc/project/models/data.pickle', 'rb') as f:
    model = pickle.load(f)
score = model.score(X, Y)
print("score=", score)
