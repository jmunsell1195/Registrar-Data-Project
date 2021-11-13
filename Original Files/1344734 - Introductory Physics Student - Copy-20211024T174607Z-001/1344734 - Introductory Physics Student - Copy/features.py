import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

'''This segment makes a dataframe which combines the math ACT/SAT
   scores, scaling the ACT scores to SAT values and Z-scaling all
                          of the scores'''

#Import data as pandas dataframes, set data type as string to avoid clipping
# PUID (i.e. int shaves off leading zeros)

actsat = pd.read_csv('ACT - SAT.csv',dtype = str)
acttosat = pd.read_csv('ACT2SAT.csv',dtype = str)
aptest = pd.read_csv('AP Tests.csv',dtype = str)
ptgpa = pd.read_csv('Prior Term GPA.csv',dtype =str)
course = ptgpa[['Key','COURSE_IDENTIFICATION']]
curr = pd.read_csv('Curricula.csv',dtype = str)
demo = pd.read_csv('Demographics.csv',dtype = str)
hs = pd.read_csv('Highschool.csv',dtype = str)
SP2020_220 = pd.read_csv('SP2020_220_Test/midterm.csv',dtype = str)

dfs = [actsat,acttosat,aptest,curr,demo,hs]
dfs[:] = [pd.merge(df,course,on = 'Key') for df in dfs]
dfs[:] = [df[df['COURSE_IDENTIFICATION']== 'PHYS22000'] for df in dfs]
actsat = dfs[0]
acttosat = dfs[1]
aptest = dfs[2]
curr = dfs[3]
demo = dfs[4]
hs = dfs[5]
    
# Make a list of PUID of Phys 220 Sp 2020 Students
PUID = SP2020_220['Student ID'].to_list()
PUID = [str(ID) for ID in PUID]
PUID = [ID[len(ID)-7:len(ID)-1] for ID in PUID]

#Combine 'act - sat' and 'acttosat' joining on PUID
Math = pd.merge(actsat,acttosat,on = 'Key')
# Keep only columns related to math score
Math = Math[['Key','ACT_Math','SAT_Mathematics','SATR Math']]
# Replace missing values with zero
Math.fillna(0,inplace = True)
#Convert act to sat score (int) 
Math['ACT_Math'] = Math['ACT_Math'].apply(int).apply(lambda x: x*(800/36)).apply(int)
#Convert sat score to int
Math['SAT_Mathematics'] = Math['SAT_Mathematics'].apply(int)
#convert satr score to ind
Math['SATR Math'] = Math['SATR Math'].apply(int)
# Combine act and sat score, keeping the highest value
Math['ACT/SAT Score'] = Math[['ACT_Math','SAT_Mathematics','SATR Math']].values.max(axis = 1)
# Tranform ACT/SAT scores to Z-value
score_mean = Math['ACT/SAT Score'].mean()
score_std = Math['ACT/SAT Score'].std()
Math['ACT/SAT Score'] = Math['ACT/SAT Score'].apply(lambda x: (x-score_mean)/score_std)
# Keep only the relevant columns
Math = Math[['Key','ACT/SAT Score']]

'''This segment makes a dataframe containing the number 
   of classes of AP math/physics that students took ''' 

# Place Null in for missing values
aptest.fillna('Null',inplace = True)
# Creates new column 'Math' the value is 1 if it is a math class, else 0
aptest['Math'] = aptest['TEST_DESC'].apply(lambda x: 1 if 'Calculus' in x else 0)
# Creates new column 'Physics' the value is 1 if it is a physics class, else 0
aptest['Physics'] = aptest['TEST_DESC'].apply(lambda x: 1 if ('Mechanics' in x or 'Elec' in x or 'Physics' in x) else 0)
# Keep only PUID, Math, and Physics columns
aptest = aptest[['Key','Math','Physics']]
# Z-scale Math and Physics columns
apmath_mean = aptest['Math'].mean()
apmath_std = aptest['Math'].std()
aptest['AP Math'] = aptest['Math'].apply(lambda x: (x-apmath_mean)/apmath_std)

apphys_mean = aptest['Physics'].mean()
apphys_std = aptest['Physics'].std()
aptest['AP Physics'] = aptest['Physics'].apply(lambda x: (x - apphys_mean)/apphys_std)
# Sum the columns for each PUID so the value represents the total number of AP classes in that subject
ap = aptest.groupby(['Key']).agg({'AP Math':'sum','AP Physics':'sum'})

'''This segment makes a dataframe containing information about the studen't GPA'''

# Make Dataframe of student's Highschool GPA. To handle missing values, a composite
# of all available data is used not including 0 values in the average

gpa = hs
# reset index since dropping 172 student data
gpa.reset_index(inplace = True,drop = True)
# keep only the relevant columns
gpa = gpa[['Key','MathGPA','PhysicsGPA','CoreGPA','SCHOOL_GPA']]
# fill NaN cells with 0.0 to avoid errors resulting from NaN values
gpa.fillna(0.0,inplace = True)
# Make list of columns containing GPA information
cols = [col for col in gpa.columns.to_list() if col != 'Key']
for col in cols:
    gpa[col] = gpa[col].apply(float)
# Function calculates average GPA from non-zero values given a list of
# all HS gpa values
def AvgNonZero(lst):
    lstNZ = [itm for itm in lst if itm != 0.0 ]
    if len(lstNZ) != 0:
        return np.average(lstNZ)
    else:
        return 0.0
# Make a list of indicies to use as an iterable for .iloc in the lambda function
gpa['ind'] = [i for i in range(gpa.shape[0])]
# add all of the non-zero elements to a list and set it as the 'gpa' column
gpa['gpa'] = gpa['ind'].apply(lambda x: gpa.iloc[x,1:5].to_list())
# use the function AvgNonZero to take average of all of the non-zero elements
gpa['gpa'] = gpa['gpa'].apply(AvgNonZero)
# Z-scale the gpa scores and name column 'Composite HS GPA'
gpa_mean = gpa['gpa'].mean()
gpa_std = gpa['gpa'].std()
gpa['Composite HS GPA'] = gpa['gpa'].apply(lambda x: (x-gpa_mean)/gpa_std)

# Keep the columns 'Key' and 'Composite HS GPA' only
gpa = gpa[['Key','Composite HS GPA']]
# Delete duplicates keeping the max Composite GPA
gpa = gpa.groupby(['Key']).agg({'Composite HS GPA':'max'})

#Put all numerical data in dataframe called Data (pd.merge method only takes 2 dataframes at at time)
Data = pd.merge(Math,ap,on = 'Key')
Data = pd.merge(Data,gpa,on = 'Key')

#Output Data dataframe into a .csv file
Data.to_csv('Data.csv', index = False)

'''This segment makes a one-hot encoded vector about the student's major'''
   
major = curr['MAJOR'].values.reshape((-1,1))
major_OH = OneHotEncoder().fit_transform(major).toarray()

'''This segment makes a one-hot encoded vector about the student's HS AP course history'''

hs['MathYrTaken'].fillna('Null',inplace = True)
hs['PhysicsYrTaken'].fillna('Null',inplace = True)
mathyr = hs['MathYrTaken'].values.reshape((-1,1))
physyr = hs['PhysicsYrTaken'].values.reshape((-1,1))
mathyr_OH = OneHotEncoder().fit_transform(mathyr)
physyr_OH = OneHotEncoder().fit_transform(physyr)

'''This segment makes a one-hot encoded vector about the student's demographic information'''

gender = demo['GENDER'].values.reshape((-1,1))
gender_OH = OneHotEncoder().fit_transform(gender)
race = demo['REPORTING_ETHNICITY'].values.reshape((-1,1))
race_OH = OneHotEncoder().fit_transform(race)
demo['ADMISSIONS_ATTRIBUTE'].fillna('Null',inplace = True)
attribute = demo['ADMISSIONS_ATTRIBUTE'].values.reshape((-1,1))
attribute_OH = OneHotEncoder().fit_transform(attribute)

#Function to find duplicate entries in a list. Returns a dictionary of 
#          dupl. items and their indicies in the list

def duplicate(lst):
    duplicates = {}
    dup_list = [(lst[i],i) for i in range(len(lst)) if lst.count(lst[i]) > 1]
    dup_item = [dup_list[i][0] for i in range(len(dup_list))]
    dup_item = list(set(dup_item))
    dup_item.sort()
    for item in dup_item:
        ind = []
        for i in range(len(dup_list)):
            if item == dup_list[i][0]:
                ind.append(dup_list[i][1])
        if item not in duplicates.keys():
            duplicates[item] = ind
    return duplicates

# pca = PCA(n_components = 2)
# X = Data.iloc[:,1:5]
# X_labs = Data.iloc[:,0].to_list()
# pc = pca.fit_transform(X)
# DFpc = pd.DataFrame(data = pc,columns = ['Principal Component 1','Principal Component 2'])
# DFpc['Key'] = X_labs
# DFpc = pd.merge(DFpc,Label,on = 'Key')


# x = DFpc['Principal Component 1'].to_list()
# y = DFpc['Principal Component 2'].to_list()
# label = DFpc['Knowledge Level']
# colors = ['red','blue']

# fig = plt.figure(figsize=(8,8))
# plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))

# cb = plt.colorbar()
# loc = np.arange(0,max(label),max(label)/float(len(colors)))
# cb.set_ticks(loc)
# cb.set_ticklabels(colors)






    
    




    
    


    













    
