import pandas as pd
model=pd.read_csv("career.csv")
model

import numpy as np
model = model.replace(np.nan,-1)
model = model.replace('y',1)
model = model.replace('n',0)
model

#define X=features and Y=target variables
model = model.replace('science', 100)
model = model.replace('comerce', 200)
model = model.replace('arts', 300)
Y=model['target']
X=model[['sq1', 'sq2', 'sq3', 'sq4', 'sq5', 'sq6', 'sq7', 'sq8', 'sq9', 'sq10',
       'cq1', 'cq2', 'cq3', 'cq4', 'cq5', 'cq6', 'cq7', 'cq8', 'cq9', 'cq10',
       'aq1', 'aq2', 'aq3', 'aq4', 'aq5', 'aq6', 'aq7', 'aq8', 'aq9','aq10']]
model

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

y_predict=Rf.predict(X_test)
y_predict
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
mean_absolute_error(Y_test, y_predict)
mean_absolute_percentage_error(Y_test, y_predict)

#Split the data into train and test
from sklearn.model_selection import train_test_split
# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)
X_train.shape, X_test.shape
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10,n_estimators=10, oob_score=True)
classifier_rf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = classifier_rf.predict(X_test)
print("A:",accuracy_score(Y_test,y_predict))

import pickle
# Saving model to current directory
pickle.dump(classifier_rf, open('model1.pkl','wb'))
#Loading model to compare the results
model1 = pickle.load(open('model1.pkl','rb'))
#fop = model1.predict([[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])
fop = model1.predict([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])
#fop = model1.predict([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1]])
fop
print()
print("Suggested Stream:")
if (fop == [100]):
    print('Science')
if (fop == [200]):
  print('Commerce')
if (fop == [300]):
  print('Arts')
print()
