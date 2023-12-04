import pandas as pd
models=pd.read_csv("career_sci.csv")
models

import numpy as np
#model = model.replace(r'^\s*$', np.nan, regex=True)
models = models.replace(np.nan,-1)
models = models.replace('y',1)
models = models.replace('n',0)
models

#define X=features and Y=target variables
models = models.replace('Medical', 110)
models = models.replace('Engineering', 130)
models = models.replace('Architecture', 150)
models = models.replace('Aeronautical', 170)
models = models.replace('Pharmacy', 190)
Y=models['target']
X=models[['mq1','mq2','mq3','mq4','mq5','eq1','eq2','eq3','eq4','eq5','arq1','arq2','arq3','arq4','arq5','aeq1','aeq2','aeq3','aeq4','aeq5','pq1','pq2','pq3','pq4','pq5']]
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
#error percentage
# from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
# mean_absolute_error(Y_test, y_predict)
# mean_absolute_percentage_error(Y_test, y_predict)

#Split the data into train and test
# Total : 168, Training data: 118, Testing Data: 50
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
pickle.dump(classifier_rf, open('model_sci.pkl','wb'))
#Loading model to compare the results
model1s = pickle.load(open('model_sci.pkl','rb'))
#fop = model1.predict([[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])
fop = model1s.predict([[1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0]])
#fop = model1.predict([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1]])
fop
print()
print("Suggested Stream:")
if (fop == [110]):
    print('Medical')
if (fop == [130]):
  print('Engineering')
if (fop == [150]):
  print('Architecture')
if (fop == [170]):
  print('Aeronautical')
if (fop == [190]):
  print('Pharmacy')
print()
