import pandas as pd 
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/lnc/project/datasets/cleandata.csv', sep = ';')
X = df.message
Y = df.label
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 225,stratify=Y)
train_df = pd.DataFrame()
test_df = pd.DataFrame()
train_df['message'] = X_train
train_df['label'] = Y_train
test_df['message'] = X_test
test_df['label'] = Y_test
test_df.to_csv('/home/lnc/project/datasets/test.csv', sep=';', index=None)
train_df.to_csv('/home/lnc/project/datasets/train.csv', sep=';', index=None)
