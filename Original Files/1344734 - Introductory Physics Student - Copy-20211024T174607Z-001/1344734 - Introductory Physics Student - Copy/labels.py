#Import libraries
import numpy as np
from sklearn.cluster import KMeans
from features import Data, PUID

drop_lst = [i for i in Data['Key'].to_list() if i not in PUID]
drop_lst = [Data['Key'].to_list().index(i) for i in drop_lst]
Data.drop(drop_lst,inplace = True)

Data.drop_duplicates(subset = 'Key',inplace = True)
Data.reset_index(drop = True,inplace = True)

df_train = Data.sample(n = int(0.05*Data.shape[0]),replace = False)
train_key = df_train['Key'].to_list()
df_test = Data[Data['Key'].apply(lambda x: x not in train_key)]

df_train.reset_index(inplace = True,drop = True)
df_test.reset_index(inplace = True,drop = True)


X_train = np.array(df_train.iloc[:,1:5])
X_test = np.array(df_test.iloc[:,1:5])

km = KMeans(n_clusters = 2,random_state = 20)
df_train['Label'] = km.fit_predict(X_train)
df_test['label'] = km.predict(X_test)







