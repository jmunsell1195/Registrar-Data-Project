import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from xgboost import XGBClassifier
from dataProcessing import train,test,trainLab,testLab
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


 ##!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!##
###!                                                         !###
 ##!                             ML.py                       !##
###!                                                         !###
 ##!      This program performs trains and validates         !##
###!      a machine learning model using the features        !###
 ##!      and labels generated in the previous files         !## 
###!         (Make_Data.py and dataProcessing.py)            !###  
 ##!                                                         !##   
###!      We do not have a designated testing set, so        !###
 ##!      our approach will be to take a random split        !##
###!      of the data for validation to select a model,      !###
 ##!      then we will combine train and test to see if      !##
###!               that result is reasonable                 !###
 ##!                                                         !##
###!                                                         !###
 ##!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!##



#Grab a split of the training data for validation
#train,val,trainLab,valLab = train_test_split(train,trainLab,test_size=0.25,random_state=42)

#Define random forest classifier object
rf = RandomForestClassifier(n_estimators=5000)
xgb = XGBClassifier(learning_rate=1e-1,n_estimators=5000)
vc = VotingClassifier(estimators=[('rf',rf),('xgb',xgb)])


#Make data frame for later analysis of classification results
classResults = pd.DataFrame(columns=['Key','finalGrade'])
classResults['Key'] = test['Key']
classResults['finalGrade'] = test['finalGrade']
classResults['label'] = testLab

#drop columns that are not needed
train.drop(['Key','hsGradDate','course',
'academicPeriodDesc','finalGrade','AP Phys Score','collegeGPATerm','ChangedSchools','primaryProgram','hsPhysYr-Cat','hsMathYr','hsPhysYr'],axis=1,inplace=True)

#val.drop(['Key','hsGradDate','course',
#'academicPeriodDesc','collegeGPATerm','primaryProgram','hsPhysYr-Cat','ChangedSchools','hsMathYr','hsPhysYr'],axis=1,inplace=True)

test.drop(['Key','hsGradDate','course',
'academicPeriodDesc','collegeGPATerm','finalGrade','AP Phys Score','primaryProgram','ChangedSchools','hsPhysYr-Cat','hsMathYr','hsPhysYr'],axis=1,inplace=True)

#cross validation to select the model and parameters
scores = cross_val_score(vc,train,trainLab,cv=5)
print(scores)
print(scores.mean())
print(scores.std())


#Fit the random forest classifier to training set
vc.fit(train,np.array(trainLab))

#Make predictions 
preds = vc.predict(test)

#make new column with the model predictions
classResults['preds'] = preds




#perform cross validation to see how the results of this prediction
#compare to other possible combinations of training/testing 

print(classification_report(testLab,preds))
print(confusion_matrix(testLab,preds))

#misclass10: predicted as 1 and actually 0
misclass10 = classResults[(classResults['preds'] == 1) & (classResults['label'] == 0)]['finalGrade']
misclass10 = dict(misclass10.value_counts())

#misclass01: predicted as 0 and actually 1
misclass01 = classResults[(classResults['preds'] == 0) & (classResults['label'] == 1)]['finalGrade']
misclass01 = dict(misclass01.value_counts())

#All classifications
totals = dict(classResults['finalGrade'].value_counts())

#replace raw error counts with normalized value
for gr in misclass10.keys():
    misclass10[gr] = misclass10[gr]/totals[gr]

for gr in misclass01.keys():
    misclass01[gr] = misclass01[gr]/totals[gr]








