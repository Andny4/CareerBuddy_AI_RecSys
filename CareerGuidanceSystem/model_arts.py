import pandas as pd
modela=pd.read_csv("career_arts.csv")
modela

import numpy as np
modela = modela.replace(np.nan,-1)
modela = modela.replace('y',1)
modela = modela.replace('n',0)
modela

modela = modela.replace('artOne', 305)
modela = modela.replace('artTwo', 350)
modela = modela.replace('artThree', 395)
Y=modela['target']
X=modela[['sq1', 'sq2', 'sq3', 'sq4', 'sq5', 'sq6', 'sq7',
       'cq1', 'cq2', 'cq3', 'cq4', 'cq5', 'cq6', 'cq7', 
       'aq1', 'aq2', 'aq3', 'aq4', 'aq5', 'aq6', 'aq7']]
modela

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
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)
X_train.shape, X_test.shape
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10,n_estimators=10, oob_score=True)
classifier_rf.fit(X_train, y_train)


import pickle
# Saving model to current directory
pickle.dump(classifier_rf, open('model_art.pkl','wb'))
#Loading model to compare the results
model1a = pickle.load(open('model_art.pkl','rb'))
#fop = model1.predict([[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])
fop = model1a.predict([[1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]])
#fop = model1.predict([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1]])
fop
print()
print("Suggested Stream:")
if (fop == [305]):
    print('One')
if (fop == [350]):
  print('Two')
if (fop == [395]):
  print('Three')
print()
