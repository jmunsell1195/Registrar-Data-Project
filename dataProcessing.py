from numpy import dtype
import pandas as pd
from scipy.sparse.construct import random
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from Make_Data import Data 

Data.reset_index(inplace=True,drop=True)


Data_220 = Data[Data['course'] == 'PHYS22000']

 ##!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!##
###!                                                         !###
 ##!                    Data_Processing.py                   !##
###!                                                         !###
 ##!      This program processes the data output from        !##
###!      Make_Data.py preparing features for use in         !###
 ##!      machine learning algorithms by:                    !##
###!                                                         !###
 ##!                -- Splitting into train/test             !##
###!                -- Converting string -> int              !###
 ##!                -- Z-scaling (numerical)                 !##
###!                -- Mean Encoding (Categorical)           !###
 ##!                -- Creating new features                 !##
###!                                                         !### 
 ##!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!##

class Data_Processing:

    """
    Data_Processing takes a data frame and prepares the features for use in a 
    machine learning mode

    Attributes:
    ---------------------
    data: pd.DataFrame  -> features and target input
    size: Int           -> size of the testing set
    train: pd.DateFrame -> training split of data
    test: pd.DataFrame  -> testing split of size "size"

    Methods:
    ----------------------
    hs                  -> processes features from high school data
                           mainly numerical features

    categorical         -> processes the categorical features 

    makeLabels          -> converts final grade to binary target
                           1 >= B-  and  0 > C+

    output              -> applies other methods in order and outputs the 
                           result. train and test set/labels

    """

    # Load data
    def __init__(self,data,size,val):
        self.data = data
        self.data.dropna(subset=['finalGrade'],inplace=True)
        if val == 'split':
            self.train = self.data[self.data['academicPeriodDesc'] != 'Spring 2021']
            self.test = self.data[self.data['academicPeriodDesc'] == 'Spring 2021'] 
        else:
            self.train,self.test = train_test_split(self.data,test_size=size)
        self.train.drop_duplicates(subset=['Key'],keep='first',inplace=True)
        self.test.drop_duplicates(subset=['Key'],keep='first',inplace=True)
        self.trainLab,self.testLab = None,None
  

    # Processes HS values
    #   -impute missing values
    #   -perform Z-scaling
    #   -create new features
    def hs(self):
        # Drop data points with missing grad date a hsGPA values
        self.train.dropna(subset=['hsGradDate','hsGPA'],how='any',inplace=True)
        self.test.dropna(subset=['hsGradDate','hsGPA'],how='any',inplace=True)

        # Dictionary of replacement values:
        # First try to replace 'math/phys GPA' with hsGPA (vals)
        # Then try to replace with col average (vals2)
        valsTr = {'hsMathGPA':self.train['hsGPA'],'hsPhysGPA':self.train['hsGPA']}
        valsTe = {'hsMathGPA':self.test['hsGPA'],'hsPhysGPA':self.test['hsGPA']}
        vals2Tr = {'hsMathGPA':self.train['hsMathGPA'].astype(dtype=float).mean(),'hsPhysGPA':self.train['hsPhysGPA'].astype(dtype=float).mean()}
        vals2Te = {'hsMathGPA':self.test['hsMathGPA'].astype(dtype=float).mean(),'hsPhysGPA':self.test['hsPhysGPA'].astype(dtype=float).mean()}

        #fill nan with vals
        self.train.fillna(value=valsTr,inplace=True)
        self.test.fillna(value=valsTe,inplace=True)
        #fill remaining nan with vals2 
        self.train.fillna(value=vals2Tr,inplace=True)
        self.test.fillna(value=vals2Te,inplace=True)
        print(self.data['hsMathGPA'].isna().sum())

        #fill nan in 'math/phys Yr' cols
        vals = {'hsMathYr': '0','hsPhysYr': '0'}
        #fill nan with vals
        self.train.fillna(value=vals,inplace=True)
        self.test.fillna(value=vals,inplace=True)

        ### Standardize differences from the way diff analysts prepared the dat
        convDict = {'January': 1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,
                    'August':8,'September':9,'October':10,'November':11,'December':12}
        
        def hsDateReg(x):
            if '/' in x:
                return x
            else:
                x = x.replace(',','').split(' ')
                x[0] = str(convDict[x[0]])
                x[2] = x[2][2:4]
                return ('/').join(x)

        self.train['hsGradDate'] = self.train['hsGradDate'].map(lambda x: hsDateReg(x))
        self.test['hsGradDate'] = self.test['hsGradDate'].map(lambda x: hsDateReg(x))

        m = self.train['ACT/SAT Math (avg)'].mean()
        n = self.train['ACT/SAT Non-Math (avg)'].mean()

        val = {'ACT/SAT Math (avg)':m,'ACT/SAT Non-Math (avg)':n}
        self.train.fillna(value = val,inplace=True)
        self.test.fillna(value = val,inplace=True)

        ### Z-scaling ###
        for col in ['hsGPA','hsMathGPA','hsPhysGPA','ACT/SAT Math (avg)',
        'ACT/SAT Non-Math (avg)','AP Math Score','AP Phys Score','collegeGPA']:
            self.train[col] = self.train[col].astype(dtype=float)
            self.test[col] = self.test[col].astype(dtype=float)
            m = self.train[col].mean()
            s = self.train[col].std()
            self.train[col] = self.train[col].map(lambda x: (x - m)/s)
            self.test[col] = self.test[col].map(lambda x: (x-m)/s)


    #Processes categorical features (gender,ethnicity, etc...)
    def categorical(self):
        self.train.dropna(subset=['finalGrade','collegeGPA'],inplace=True)
        self.test.dropna(subset=['finalGrade','collegeGPA'],inplace=True)
        
        #placeholder in case  want to do regression later
        # gpa_dict = {'A+':4.0,'A':4.0,'A-':3.7,'B+':3.3,'B':3.0,'B-':2.7,'C+':2.3,
        #             'C':2.0,'C-':1.7,'D+':1.3,'D':1.0,'D-':0.0,'F':0.0}
        
        
        vals = {'firstGenCollege':'N','repeatInd':'N','college':stats.mode(self.train['college'])[0][0],
        'major':stats.mode(self.train['major'])[0][0],'studentClassification':stats.mode(self.train['studentClassification'])[0][0],
        'primaryProgram':stats.mode(self.train['primaryProgram'])[0][0]}
        
        #Fill missing values
        self.train.fillna(value=vals,inplace=True)
        self.test.fillna(value=vals,inplace=True)



        #################
        #               #
        # Mean Encoding #
        #               #
        #################

        # maj = self.train['major'].unique()[0:10]
        # self.train['major'] = self.train['major'].map(lambda x: x if x in maj else 'oth')
        # self.test['major'] = self.test['major'].map(lambda x: x if x in maj else 'oth')


        cols = ['hsMathYr-Cat','hsPhysYr-Cat','gender','ethnicity','underRepMin','firstGenCollege',
                'repeatInd','ChangedSchools','college','major','studentClassification','AP Math','AP Phys'] 
        
        grades1 = ['A+','A','A-','B+','B','B-','P']

        #Function standardizes classYrTaken
        # e.g. (mathYrTaken) by assigning 
        #          category
        def clsYrEnc(x):
            if x <= 1:
                return 0
            elif x > 1 and x <= 2:
                return 1
            elif x > 2 and x<= 3:
                return 2
            elif x > 3 and x<=4:
                return 3
            else:
                return 4

        # Function takes array of unique categories from column
        #    -Reduces the number of categories
        #    -Standardizes "unseen" values between data sets
        #     e.g. training/test
        def maxVal(xArr):
            v = len(xArr)
            if v<=10:
                return xArr
            else: 
                return xArr[0:10]


        # Standardize hsYr values for hsMathYr and hsPhysYr #

        self.train['hsMathYr-Cat'] = self.train['hsMathYr'].map(lambda x: clsYrEnc(float(x))) 
        self.train['hsPhysYr-Cat'] = self.train['hsPhysYr'].map(lambda x: clsYrEnc(float(x)))

        self.test['hsMathYr-Cat'] = self.test['hsMathYr'].map(lambda x: clsYrEnc(float(x))) 
        self.test['hsPhysYr-Cat'] = self.test['hsPhysYr'].map(lambda x: clsYrEnc(float(x)))


        self.train['finalGrade-temp'] = self.train['finalGrade'].map(lambda x: 1 if x in grades1 else 0)
        for col in cols:
            # Reduce number of categories and standardize between train and test
            cats = maxVal(list(self.train[col].unique()))
            
            # Replace minority categories with other and standardize
            self.train[col] = self.train[col].map(lambda x: x if x in cats else 'oth')
            self.test[col] = self.test[col].map(lambda x: x if x in cats else 'oth')
            
            #make empty mean-encoding dictionary
            meanENC = {}

            #build mean encoding dictionary
            #select each category and find
            #label mean value for that category
            for cat in self.train[col].unique():
                m = self.train[self.train[col] == cat]['finalGrade-temp'].mean()
                meanENC[cat] = m

            #Now we have encoding dict, perform transformation
            self.train[col] = self.train[col].map(lambda x: meanENC[x])
            try:
                self.test[col] = self.test[col].map(lambda x: meanENC[x])
            except:
                meanENC['oth'] = 0
                self.test[col] = self.test[col].map(lambda x: meanENC[x])
                


        self.train.drop(['finalGrade-temp'],axis=1,inplace=True)


    def APactSat(self):
        vals = {'ACT/SAT Math (avg)':stats.mode(self.train['ACT/SAT Math (avg)'])[0][0],'ACT/SAT Non-Math (avg)':stats.mode(self.train['ACT/SAT Non-Math (avg)'])[0][0],'AP Math':stats.mode(self.train['AP Math'])[0][0],
                'AP Math Score':0,'AP Phys':stats.mode(self.train['AP Phys'])[0][0],'AP Phys Score':0}
        self.train.fillna(value=vals,inplace=True)
        self.test.fillna(value=vals,inplace=True)


    def makeLabels(self):
        #class 1 if finalGrade in 'grades1' else class 0
        grades1 = ['A+','A','A-','B+','B','B-','P']
        self.trainLab = self.train['finalGrade'].map(lambda x: 1 if x in grades1 else 0)
        self.testLab = self.test['finalGrade'].map(lambda x: 1 if x in grades1 else 0)

    def output(self):
        self.APactSat()
        self.hs()
        self.categorical()
        self.makeLabels()
        self.train
        self.test
        self.trainLab,self.testLab
        return self.train,self.test,self.trainLab,self.testLab


x = Data_Processing(Data_220,0.2,'split')
train,test,trainLab,testLab = x.output()





