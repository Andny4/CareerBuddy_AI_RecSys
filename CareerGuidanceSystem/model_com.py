import pandas as pd
models=pd.read_csv("career_com.csv")
models

import numpy as np
models = models.replace(np.nan,-1)
models = models.replace('y',1)
models = models.replace('n',0)
models

#define X=features and Y=target variables
models = models.replace('E-commerce/Marketing', 210)
models = models.replace('Journalism', 230)
models = models.replace('Insurance/Banking', 250)
models = models.replace('Law Programs', 270)
models = models.replace('CA', 290)
Y=models['target']
X=models[['mq1','mq2','mq3','mq4','mq5','jq1','jq2','jq3','jq4','jq5','bq1','bq2','bq3','bq4','bq5','lq1','lq2','lq3','lq4','lq5','caq1','caq2','caq3','caq4','caq5']]
models

Y.shape
X.shape

#Regressor
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.7,random_state=2)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape
from sklearn.ensemble import RandomForestRegressor
Rf=RandomForestRegressor()
#train or fit model
Rf.fit(X_train,Y_train)
#predict
y_predict=Rf.predict(X_test)
y_predict

#Split the data into train and test
from sklearn.model_selection import train_test_split
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)
X_train.shape, X_test.shape
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10,n_estimators=10, oob_score=True)
classifier_rf.fit(X_train, y_train)


import pickle
# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(classifier_rf, open('model_com.pkl','wb'))
#Loading model to compare the results
model1s = pickle.load(open('model_com.pkl','rb'))
#fop = model1.predict([[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])
fop = model1s.predict([[1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0]])
#fop = model1.predict([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1]])
fop
print()
print("Suggested Stream:")
if (fop == [210]):
    print('E')
if (fop == [230]):
  print('J')
if (fop == [250]):
  print('I')
if (fop == [270]):
  print('L')
if (fop == [290]):
  print('C')
print()
