import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Hyperparams
reference_date = '1970-01-01'
eta = 0.001
est = 1000
depth = 10

# Preprocessing
train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

def preprocessing(raw_data):
    data = raw_data.dropna()
    data['date'] = pd.to_datetime(data['date'])
    data['date'] = (data['date'] - pd.to_datetime(reference_date)).dt.days

    data['dep_time'] = pd.to_datetime(data['dep_time'], format='%H:%M').dt.time.apply(lambda x : x.hour * 60 + x.minute)
    data['arr_time'] = pd.to_datetime(data['arr_time'], format='%H:%M').dt.time.apply(lambda x : x.hour * 60 + x.minute)
    data['time_taken'] = (data['arr_time'] - data['dep_time']).apply(lambda x : x + 1440 if x < 0 else x)

    drop_columns = ['id', 'ch_code', 'num_code']
    data = data.drop(columns = drop_columns)
    #data['stop'] = data['stop'].apply(lambda x : "1-stop" if x != "non-stop" and x!= "1-stop" else x)

    category_columns = ['airline', 'from', 'stop', 'to', 'class']
    #data = pd.get_dummies(data = data, columns = category_columns)
    data[category_columns] = data[category_columns].astype('category')

    return data

train = preprocessing(train_raw)#.sample(frac = 0.25, random_state = 0)
test = preprocessing(test_raw)

X = train.drop(columns = ['price'])
Y = np.array(train['price'])

# Split the data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state = 0)
X_test = test

# Boosted Tree
dtrain = xgb.DMatrix(X_train, label = Y_train, enable_categorical = True)
dval = xgb.DMatrix(X_val, label = Y_val, enable_categorical = True)
dtest = xgb.DMatrix(X_test, enable_categorical = True)

params = {"max_depth" : depth, "learning_rate" : eta, 
          "n_estimators" : est, "objective" : 'reg:squarederror'}
evals = [(dtrain, "train"), (dval, "validation")]

model = xgb.train(params, dtrain, num_boost_round = 8000, 
                  evals = evals, verbose_eval = 500)

predictions = model.predict(dtest)

test_dict = {'id' : test_raw['id'],
             'price' : predictions}
test_data = pd.DataFrame(data = test_dict)
test_data.to_csv('save.csv', index = False)
