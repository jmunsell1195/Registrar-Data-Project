import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
import xgboost
from dataProcessing import train,test,trainLab,testLab
from sklearn.metrics import classification_report,confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.model_selection import cross_val_score

train,val,trainLab,valLab = train_test_split(train,trainLab,test_size=0.25,random_state=42)

rf = RandomForestClassifier(n_estimators=5000)

train.drop(['Key','hsGradDate','course',
'academicPeriodDesc','collegeGPATerm','ChangedSchools','primaryProgram','hsPhysYr-Cat','hsMathYr','hsPhysYr'],axis=1,inplace=True)

# val.drop(['Key','hsGradDate','course',
# 'academicPeriodDesc','collegeGPATerm','primaryProgram','hsPhysYr-Cat','ChangedSchools','hsMathYr','hsPhysYr'],axis=1,inplace=True)

test.drop(['Key','hsGradDate','course',
'academicPeriodDesc','collegeGPATerm','primaryProgram','ChangedSchools','hsPhysYr-Cat','hsMathYr','hsPhysYr'],axis=1,inplace=True)

rf.fit(train,np.array(trainLab))
preds = rf.predict(test)

train = pd.concat([train,test])
trainLab = pd.concat([trainLab,testLab])


scores = cross_val_score(rf,train,trainLab,cv=5)
print(scores)

print(classification_report(testLab,preds))
print(confusion_matrix(testLab,preds))

print(scores.mean())
print(scores.std())







