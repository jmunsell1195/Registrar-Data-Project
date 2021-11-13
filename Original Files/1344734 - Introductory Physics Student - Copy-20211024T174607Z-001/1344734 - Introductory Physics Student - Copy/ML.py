from labels import df_train,df_test, X_train, X_test
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, roc_auc_score


y_train = df_train.iloc[:,5]
y_test = df_test.iloc[:,5]





ridge = linear_model.Ridge()
lasso = linear_model.Lasso()
linreg = linear_model.LinearRegression()
RFor = RandomForestClassifier()

# # cv = cross_validate(lasso,Data_Arr,Label_Arr,cv = 5)
# # print(type(cv['test_score']))

# # cv = cross_validate(ridge,Data_Arr,Label_Arr,cv = 5)
# # print(cv['test_score'])

# # cv = cross_validate(linreg,Data_Arr,Label_Arr,cv = 5)
# # print(cv['test_score'])




RFor.fit(X_train,y_train)
pred = RFor.predict(X_test)
print('RF')
print(accuracy_score(y_test,pred))
print(precision_score(y_test,pred))
print(roc_auc_score(y_test,pred))

def thresh(pred):
    if pred >= 0.25:
        return 1
    else:
        return 0
    
    
# # preds = np.vectorize(thresh)(pred)
    
    
# ridge.fit(X_train,y_train)
# pred = ridge.predict(X_test)
# preds = np.vectorize(thresh)(pred)
# print('Ridge')
# print(accuracy_score(y_test,preds))

# lasso.fit(X_train,y_train)
# pred = lasso.predict(X_test)
# preds = np.vectorize(thresh)(pred)
# print('lasso')
# print(accuracy_score(y_test,preds))

# y_predicted = km.fit_predict(Data_Exam)
# print('KMeans Exams')
# print(accuracy_score(Labs_Exam,y_predicted))

# pred = km.predict(Data_Mods)
# print('KMeans Mods')
# print(accuracy_score(Labs_Mods,pred))









