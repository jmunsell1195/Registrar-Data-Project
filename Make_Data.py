import pandas as pd
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None
import os


 ##!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!##
###!                                                         !###
 ##!                     Make Data.py                        !##
###!                                                         !###
 ##!   This program processes two sets of data obtained      !##
###!   obtained from the University registrar concerning     !###
 ##!   students' past academic behaviors. The data sets      !##
###!   were prepared by different analysts and thus have     !###
 ##!   different feature names and structure. This program   !##
###!   aggregates the data and standardizes the features     !###
 ##!               and performs minor processing             !##
###!                                                         !###
 ##!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!#!##


###################################
#                                 #
#  Read CSV data as pd dataframe  #
#  Data is stored as a dict of    #
#           dataframes            #
#                                 #
###################################

#######################################
#                                     #
#     Files:                          #
#        2020:                        #
#           -Highschool.csv           #
#           -Prior Term GPA.csv       #
#           -Demographics.csv         #
#           -Curricula.csv            #
#           -AP Tests.csv             #
#           -ACT-SAT.csv              #
#           -ACT2SAT.csv              #
#                                     #
#         2021:                       #
#           -HS.csv                   #
#           -Prior_Term_GPA.csv       #
#           -Prior_Overall_GPA.csv    #
#           -Student_Curricula.csv    #
#           -Demographics_Course.csv  #
#           -AP_Credit.csv            #
#           -ACT_SAT.csv              #
#                                     #
#######################################

class Make_Data_2020:

    def __init__(self):
        ## Imports 2020 data from folder ##
        files = [file for file in os.walk('./Aug2020_Data/')][0][2]
        data_2020 = {}
        print(files)

        ## Formats data as pd.DataFrame ##
        for file in files:
            data_2020[file.split('.')[0]] = pd.read_csv('./Aug2020_Data/' + file, dtype=str)

        self.data = data_2020


    def GPA(self):

        ################################################
        #                                              #
        #        Useful Features are Taken From        #
        #               Prior Term GPA                 #
        #                                              #
        ################################################

        d2020_cols = ('Key', 'ACADEMIC_PERIOD', 'FINAL_GRADE', 'REPEAT_COURSE_IND', 'Most_Recent_Prior_Term', 'COURSE_IDENTIFICATION',
                      'Most_Recent_Prior_Term_GPA')

        GPA = self.data['Prior Term GPA'].loc[:, d2020_cols]

        return GPA


    def ACT_SAT(self):

        ###################
        #                 #
        #     ACT/SAT     #
        #                 #
        ###################

        ####################################### ACT/SAT Math ###########################################################
        ## ACT/SAT in 2 different records, grab relevant math columns ##
        ACTSAT_MATH_1 = self.data['ACT - SAT'][['Key', 'ACT_Math', 'SAT_Mathematics']]
        ACTSAT_MATH_2 = self.data['ACT2SAT'][['Key', 'SATR Math', 'ACT Math']]

        ## Combine math cols from 2 diff records ##
        ACTSAT_MATH = pd.merge(ACTSAT_MATH_1, ACTSAT_MATH_2, how='inner', on='Key')

        ## put ACT (/36) and SAT (/800) math on common scale [0,1] for 2020 data ##
        ACTSAT_2020_Cols = [col for col in ACTSAT_MATH.columns if col != '']
        for col in ACTSAT_2020_Cols:
            if col != 'Key':
                m = ACTSAT_MATH[col].apply(float).max()
                ACTSAT_MATH.loc[:, (col)] = ACTSAT_MATH[col].apply(float).apply(lambda x: x / m)

        # Take the average of non-NAN ACT/SAT values for given person ##
        ACTSAT_MATH_AVG = ACTSAT_MATH.loc[:, tuple(ACTSAT_2020_Cols)].mean(axis=1)
        ACTSAT_MATH['ACT/SAT Math (avg)'] = ACTSAT_MATH_AVG

        ###################################### ACT/SAT Non-Math ########################################################

        dnm_cols = ('Key', 'ACT_English', 'ACT_Reading', 'ACT_Composite','ACT_Combined English_Writing',
                    'SAT_Critical_Reading','SAT_Writing')
        dnm_cols2 = ('Key', 'SATR EBRW', 'SATR Total', 'ACT Composite', 'ACT English', 'ACT Combined English/Writing')

        ACTSAT_NONMATH_1 = self.data['ACT - SAT'].loc[:, dnm_cols]
        ACTSAT_NONMATH_2 = self.data['ACT2SAT'].loc[:,dnm_cols2]
        ACTSAT_NONMATH = pd.merge(ACTSAT_NONMATH_1, ACTSAT_NONMATH_2)

        ## ACT/SAT Non-Math Columns scaled for 2020 ##
        ACTSAT_2020_NMCols = [col for col in ACTSAT_NONMATH if col != 'Key']
        for col in ACTSAT_2020_NMCols:
            if col != 'Key':
                m = ACTSAT_NONMATH.loc[:, (col)].apply(float).max()
                ACTSAT_NONMATH[col] = ACTSAT_NONMATH[col].apply(float).apply(lambda x: x / m)

        ## ACT/SAT Non-Math Averaged for 2020 ##
        ACTSAT_NONMATH_AVG = ACTSAT_NONMATH[ACTSAT_2020_NMCols].mean(axis=1)
        ACTSAT_NONMATH['ACT/SAT Non-Math (avg)'] = ACTSAT_NONMATH_AVG

        ACTSAT = pd.merge(ACTSAT_MATH,ACTSAT_NONMATH,on='Key',how='inner')

        return ACTSAT[['Key','ACT/SAT Math (avg)','ACT/SAT Non-Math (avg)']]

    def dem(self):

        ##########################
        #                        #
        #   Take Useful Feats    #
        #    from Demographic    #
        #                        #
        ##########################

        d2020_cols = ('Key', 'GENDER_DESC', 'REPORTING_ETHNICITY', 'UNDERREPRESENTED_MINORITY_IND',
                      'ADMISSIONS_ATTRIBUTE_DESC')

        return self.data['Demographics'].loc[:, d2020_cols]

    def curr(self):

        ##########################
        #                        #
        #   Take Useful Feats    #
        #    from Curricula      #
        #                        #
        ##########################

        d2020_cols = ('Key', 'ACADEMIC_PERIOD','ACADEMIC_PERIOD_DESC', 'COLLEGE', 'MAJOR', 'STUDENT_CLASSIFICATION_BOAP', 'STUDENT_STATUS','PRIMARY_PROGRAM_IND' )

        return self.data['Curricula'].loc[:, d2020_cols]

    def AP(self):

        ##########################
        #                        #
        #   Take Useful Feats    #
        #    from AP Classes     #
        #                        #
        ##########################

        ## AP Dict quantifies the level of each AP class (e.g. Calc 2 > Calc 1) ##

        ap_dict = {'Math': {'AP Calculus AB': 2, 'Calculus': 1, 'AP Calculus AB Subscore': 3, 'AP Calculus BC': 4,
                            'Anlytc Geomtry&Calc I': 2, 'Analytc Geom & Calc II': 4,
                            'Calculus AB': 2, 'Algebra And Trig I': 1, 'Calculus BC': 4},
                   'Physics': {'AP Physics C - Mechanics': 3, 'AP Physics 1': 1, 'AP Physics 2': 2,
                               'AP Physics C - Elec & Magn': 5, 'AP Physics B': 3,
                               'Physics I': 1, 'Physics C (Mechanics)': 3, 'Physics II': 2, 'E&M Interactions': 3,
                               'Modern Mechanics': 2, 'Physics C (E & M)': 5, 'Gen Physics': 1,
                               'AP: Physics C Mechanics': 3}}

        ## Broadly classify AP classes as either math or phys ##
        def math_phys(txt):
            try:
                if 'Calc' in txt:
                    return 'Math'
                elif 'Phys' in txt:
                    return 'Physics'
                else:
                    return 'None'
            except:
                return 'None'

        ## Make new pd.DataFrame of just AP Math Scores ##
        AP = self.data['AP Tests']
        AP['Category'] = AP['TEST_DESC'].apply(math_phys)
        AP_Math = AP[AP['Category'] == 'Math']

        ## Map AP class to level value ##
        AP_Math.loc[:, ('TEST_DESC')] = AP_Math['TEST_DESC'].apply(lambda x: ap_dict['Math'][x])
        AP_Math['TEST_SCORE'] = AP_Math['TEST_SCORE'].apply(float)

        ## AP Math => ['key','AP Math Level','AP Math Score'] => AP Math Level= Amount and level of AP Classes
        ##                                           AP Math Score = Average AP Grade

        ## Make dataframes for AP Math Level => [.groupby().sum()] and AP Math Score [.groupby().sum()]
        Math_AP_Classes = AP_Math.groupby('Key', as_index=False).sum()
        Math_AP_Classes[['Key', 'AP Math Level']] = Math_AP_Classes[['Key', 'TEST_DESC']]
        Math_AP_Classes_Perf = AP_Math.groupby('Key', as_index=False).mean()
        Math_AP_Classes_Perf[['Key', 'AP Math Score']] = Math_AP_Classes_Perf[['Key', 'TEST_SCORE']]

        ## Merge into Math_AP dataframe
        AP_Math = pd.merge(Math_AP_Classes, Math_AP_Classes_Perf, on='Key', how='inner')

        AP_Math = AP_Math.loc[:, ['Key', 'AP Math Level', 'AP Math Score']]

        ## AP Phys => ['key','AP Phys Level','AP Phys Score'] => AP Phys Level = Amount and level of AP Physics Classes
        ##                                                       AP Phys Score = Average AP Phys Grade

        AP_Phys = AP[AP['Category'] == 'Physics']
        AP_Phys.loc[:, ('TEST_DESC')] = AP_Phys['TEST_DESC'].apply(lambda x: ap_dict['Physics'][x])
        AP_Phys.loc[:, ('TEST_SCORE')] = AP_Phys['TEST_SCORE'].apply(float)
        Physics_AP_Classes = AP_Phys.groupby(by="Key", as_index=False).sum()

        Physics_AP_Classes[['Key', 'AP Phys Level']] = Physics_AP_Classes[['Key', 'TEST_DESC']]
        Physics_AP_Classes_Perf = AP_Phys.groupby('Key', as_index=False).mean()
        Physics_AP_Classes_Perf.loc[:, ('Key', 'AP Phys Score')] = Physics_AP_Classes_Perf[['Key', 'TEST_SCORE']]
        Physics_AP_Classes = Physics_AP_Classes.loc[:, ['Key', 'AP Phys Level']]

        AP_Phys = pd.merge(Physics_AP_Classes, Physics_AP_Classes_Perf, on='Key', how='outer')
        AP_Phys = AP_Phys[['Key', 'AP Phys Level', 'AP Phys Score']]

        ## Combine AP Metrics Math/Phys into single dataframe for 2020 ##

        AP_Final = pd.DataFrame()
        keys = list(self.data['AP Tests']['Key'].unique())
        AP_Final['Key'] = keys


        AP_Final = pd.merge(AP_Final, AP_Math, on='Key', how='outer')
        AP_Final = pd.merge(AP_Final, AP_Phys, on='Key', how='outer')
        AP_Final.fillna(0,inplace=True)
        print('ap')
        print(AP_Final['AP Math Level'].isna().sum())

        return AP_Final

    def HS(self):

        ##########################
        #                        #
        #   Take Useful Feats    #
        #    from Demographic    #
        #                        #
        ##########################

        hs_cols = ('Key', 'MathGPA', 'MathYrTaken', 'PhysicsGPA',
         'PhysicsYrTaken', 'CoreGPA', 'SCHOOL_GPA',
         'SECONDARY_SCHOOL_GRAD_DATE')

        HS = self.data['Highschool'].loc[:, hs_cols]
        HS['Changed_Schools'] = HS['Key'].apply(lambda x: HS['Key'].to_list().count(x))
        HS.drop_duplicates(keep='last',inplace=True)
        return HS

    def constructor(self):

        ##########################
        #                        #
        #   Combine the records  #
        #   Export a single df   #
        #                        #
        ##########################

        df = pd.merge(self.HS(),self.ACT_SAT(),on="Key",how="inner")
        df = pd.merge(df, self.AP(), on="Key", how="inner")
        df = pd.merge(df, self.curr(), on="Key", how="inner")
        df = pd.merge(df, self.dem(), on="Key", how="inner")
        df = pd.merge(df, self.GPA(), on="Key", how="inner")

        return df


class Make_Data_2021:

    def __init__(self):
        ## Imports 2021 data from folder ##
        files = [file for file in os.walk('./Spring2021_Data/')][0][2]
        data_2021 = {}
        print(files)

        ## Formats data as pd.DataFrame ##
        for file in files:
            data_2021[file.split('.')[0]] = pd.read_csv('./Spring2021_Data/'+file,dtype=str)

        self.data = data_2021

    def GPA(self):

        ################################################
        #                                              #
        #        Useful Features are Taken From        #
        #               Prior Term GPA                 #
        #                                              #
        ################################################

        d_cols = ('PUID_SIX','ACADEMIC_PERIOD','ACADEMIC_PERIOD_DESC','Prior_Overall_GPA_Term','Prior_Term_Overall_GPA')
        GPA = self.data['Prior_Overall_GPA'].loc[:, d_cols]

        return GPA

    def ACT_SAT(self):

        ################################################
        #                                              #
        #           Useful Features Taken From         #
        #                   ACT/SAT.                   #
        #                                              #
        ################################################

        ##Keep only useful columns from data_2020 and data_2021 ##

        ACTSAT_MATH = self.data['ACT_SAT'][['PUID_SIX','SATR Math','SAT Mathematics_OLD','ACT Math']]

        ###################
        #                 #
        #  ACT/SAT Math   #
        #                 #
        ###################

        ## put ACT (/36) and SAT (/800) math on common scale [0,1] ##
        ACTSAT_Cols = [col for col in ACTSAT_MATH.columns if col != '']
        for col in ACTSAT_Cols:
            if col != 'PUID_SIX':
                m = ACTSAT_MATH[col].apply(float).max()
                ACTSAT_MATH.loc[:,(col)] = ACTSAT_MATH[col].apply(float).apply(lambda x: x/m)

        #Take the average of non-NAN ACT/SAT values for given person ##
        ACTSAT_MATH_AVG = ACTSAT_MATH[ACTSAT_Cols].mean(axis=1)
        ACTSAT_MATH['ACT/SAT Math (avg)'] = ACTSAT_MATH_AVG

        ## put ACT (/36) and SAT (/800) on common scale [0,1] for 2021 data ##
        ACTSAT_Cols = [col for col in ACTSAT_MATH.columns if col != '']
        for col in ACTSAT_Cols:
            if col != 'PUID_SIX':
                m = ACTSAT_MATH[col].apply(float).max()
                ACTSAT_MATH.loc[:, col] = ACTSAT_MATH[col].apply(float).apply(lambda x: x/m)

        #Take the average of non-NAN ACT/SAT values for given person ##
        ACTSAT_MATH_AVG = ACTSAT_MATH.loc[:,tuple(ACTSAT_Cols)].mean(axis=1)
        ACTSAT_MATH['ACT/SAT Math (avg)'] = ACTSAT_MATH_AVG

        ## ACT/SAT Non-Math Columns selected ##
        ACTSAT_NONMATH = self.data['ACT_SAT'].loc[:,['PUID_SIX', 'SATR EBRW', 'SATR Total',
               'SAT_Critical_Reading_OLD', 'ACT English',
               'ACT Reading', 'ACT Science Reasoning', 'ACT Composite',
               'ACT Combined English/Writing']]

        ACTSAT_NMCols = [col for col in ACTSAT_NONMATH.columns if col != 'Key']
        for col in ACTSAT_NMCols:
            if col != 'PUID_SIX':
                m = ACTSAT_NONMATH[col].apply(float).max()
                ACTSAT_NONMATH.loc[:,(col)] = ACTSAT_NONMATH[col].apply(float).apply(lambda x: x/m)

        ACTSAT_NONMATH_AVG = ACTSAT_NONMATH[ACTSAT_NMCols].mean(axis=1)
        ACTSAT_NONMATH['ACT/SAT Non-Math (avg)'] = ACTSAT_NONMATH_AVG
        ACTSAT_NONMATH = ACTSAT_NONMATH

        ACTSAT =  pd.merge(ACTSAT_NONMATH,ACTSAT_MATH,on='PUID_SIX',how='inner')

        ACTSAT = ACTSAT[['PUID_SIX', 'ACT/SAT Math (avg)', 'ACT/SAT Non-Math (avg)']]

        return ACTSAT

    def AP(self):

        ##########################
        #                        #
        #   Take Useful Feats    #
        #    from AP Classes     #
        #                        #
        ##########################

        ## AP Dict quantifies the level of each AP class (e.g. Calc 2 > Calc 1) ##
        ap_dict = {'Math': {'AP Calculus AB': 2, 'Calculus': 1, 'AP Calculus AB Subscore': 3, 'AP Calculus BC': 4,
                            'Anlytc Geomtry&Calc I': 2, 'Analytc Geom & Calc II': 4,
                            'Calculus AB': 2, 'Algebra And Trig I': 1, 'Calculus BC': 4},
                   'Physics': {'AP Physics C - Mechanics': 3, 'AP Physics 1': 1, 'AP Physics 2': 2,
                               'AP Physics C - Elec & Magn': 5, 'AP Physics B': 3,
                               'Physics I': 1, 'Physics C (Mechanics)': 3, 'Physics II': 2, 'E&M Interactions': 3,
                               'Modern Mechanics': 2, 'Physics C (E & M)': 5, 'Gen Physics': 1,
                               'AP: Physics C Mechanics': 3}}

        ## Broadly classify AP classes as either math or phys ##
        def math_phys(txt):
            try:
                if 'Calc' in txt:
                    return 'Math'
                elif 'Phys' in txt:
                    return 'Physics'
                else:
                    return 'None'
            except:
                return 'None'

        ## Processes AP Data for 2021 as outlined above ##
        AP = self.data['AP_Credit']
        AP['Category'] = AP['COURSE_TITLE_SHORT'].apply(math_phys)
        AP_Math = AP[AP['Category'] == 'Math']
        AP_Math.loc[:, ('COURSE_TITLE_SHORT')] = AP_Math['COURSE_TITLE_SHORT'].apply(lambda x: ap_dict['Math'][x])
        AP_Math.loc[:, ('CREDITS_EARNED')] = AP_Math['CREDITS_EARNED'].astype((float))

        ## Cumulative Math AP level
        Math_AP_Classes = AP_Math.groupby('PUID_SIX',as_index=False).sum()
        Math_AP_Classes[['PUID_SIX','AP Math']] = Math_AP_Classes[['PUID_SIX','COURSE_TITLE_SHORT']]
        Math_AP_Classes = Math_AP_Classes[['PUID_SIX','AP Math']]

        ## Average AP Math Performance
        Math_AP_Classes_Perf = AP_Math.groupby('PUID_SIX',as_index=False).mean()
        Math_AP_Classes_Perf[['PUID_SIX','AP Math Score']] = Math_AP_Classes_Perf[['PUID_SIX','CREDITS_EARNED']]
        Math_AP_Classes_Perf = Math_AP_Classes_Perf[['PUID_SIX','AP Math Score']]
        Math_AP = pd.merge(Math_AP_Classes,Math_AP_Classes_Perf,on='PUID_SIX',how="inner")

        ## Work with AP Physics
        AP_Phys = AP[AP['Category'] == "Physics"]
        AP_Phys.loc[:,('COURSE_TITLE_SHORT')] = AP_Phys['COURSE_TITLE_SHORT'].apply(lambda x: ap_dict['Physics'][x])
        AP_Phys.loc[:,('CREDITS_EARNED')] = AP_Phys['CREDITS_EARNED'].astype(float)

        ## Cumulative Physics AP Level
        Physics_AP_Classes = AP_Phys.groupby('PUID_SIX',as_index=False).sum()
        Physics_AP_Classes[['PUID_SIX','AP Phys']] = Physics_AP_Classes[['PUID_SIX','COURSE_TITLE_SHORT']]

        ## Average AP Physics Performance
        Physics_AP_Classes_Perf = AP_Phys.groupby('PUID_SIX',as_index=False).mean()
        Physics_AP_Classes_Perf[['PUID_SIX','AP Phys Score']] = Physics_AP_Classes_Perf[['PUID_SIX','CREDITS_EARNED']]

        ## Merge class level and performance info
        Physics_AP = pd.merge(Physics_AP_Classes,Physics_AP_Classes_Perf,on='PUID_SIX',how='inner')

        ## Make new pd.DataFrame with unique keys from AP_Phys dataframe
        AP_Final = pd.DataFrame()
        AP_Final['PUID_SIX'] = AP_Phys['PUID_SIX'].unique()

        ## Merge dataframes to include people who took AP Physics and/or AP Math
        ## and people who took neither class
        AP_Final = pd.merge(AP_Final,Math_AP,on='PUID_SIX',how='outer')
        AP_Final = pd.merge(AP_Final,Physics_AP,on='PUID_SIX',how='outer')

        AP_Final = AP_Final[['PUID_SIX','AP Math','AP Math Score','AP Phys','AP Phys Score']]

        print('ap 2021')
        print(AP_Final['AP Phys Score'].isna().sum())

        return AP_Final

    def HS(self):

        ##########################
        #                        #
        #   Take Useful Feats    #
        #     from HS Info       #
        #                        #
        ##########################

        HS = self.data['HS'].loc[:,
             ('PUID_SIX', 'High_School_Grad_Date', 'High_School_GPA', 'SUBJECT_DESC', 'YEARS_TAKEN', 'GRADE', 'GPA')]

        ## Make new pd.DataFrame (series + key) for HS Math and HS Phys info
        ## To match corresponding columns from data_2020
        HS_MATH_GPA = HS[HS['SUBJECT_DESC'] == 'Academic Math Total'][['PUID_SIX','GPA']]
        HS_MATH_YR_TAKEN = HS[HS['SUBJECT_DESC'] == 'Academic Math Total'][['PUID_SIX','YEARS_TAKEN']]
        HS_PHYS_GPA = HS[HS['SUBJECT_DESC'] == 'Physics'][['PUID_SIX', 'GPA']]
        HS_PHYS_YR_TAKEN = HS[HS['SUBJECT_DESC'] == 'Physics'][['PUID_SIX', 'YEARS_TAKEN']]

        ## Rename to match data_2020
        HS_MATH_GPA[['PUID_SIX','MathGPA']] = HS_MATH_GPA[['PUID_SIX','GPA']]
        HS_MATH_YR_TAKEN[['PUID_SIX','MathYrTaken']] = HS_MATH_YR_TAKEN[['PUID_SIX','YEARS_TAKEN']]
        HS_PHYS_GPA[['PUID_SIX','PhysicsGPA']] = HS_PHYS_GPA[['PUID_SIX', 'GPA']]
        HS_PHYS_YR_TAKEN[['PUID_SIX','PhysicsYrTaken']] = HS_PHYS_YR_TAKEN[['PUID_SIX', 'YEARS_TAKEN']]

        ## Merge
        HS = pd.merge(HS,HS_MATH_GPA,how='outer',on='PUID_SIX')
        HS = pd.merge(HS, HS_MATH_YR_TAKEN, how='outer', on='PUID_SIX')
        HS = pd.merge(HS, HS_PHYS_GPA, how='outer', on='PUID_SIX')
        HS = pd.merge(HS, HS_PHYS_YR_TAKEN, how='outer', on='PUID_SIX')

        HS = HS.loc[:,('PUID_SIX', 'High_School_Grad_Date', 'High_School_GPA', 'MathGPA','MathYrTaken','PhysicsGPA','PhysicsYrTaken')]

        ## Create new feature 'Changed Schools' from duplicates in HS dataframe
        HS['Changed_Schools'] = HS['PUID_SIX'].apply(lambda x: HS['PUID_SIX'].to_list().count(x))

        return HS

    def curr_dem(self):
        d2021_cols = ('PUID_SIX', 'GENDER_DESC', 'REPORTING_ETHNICITY','UNDERREPRESENTED_MINORITY_IND', 'ADMISSIONS_ATTRIBUTE_DESC',
               'ACADEMIC_PERIOD', 'ACADEMIC_PERIOD_DESC', 'COURSE_IDENTIFICATION','FINAL_GRADE','REPEAT_COURSE_IND')

        data = self.data['Demographics_Courses'].loc[:,d2021_cols]

        return data
    def stud_curr(self):
        stCur = self.data['Student_Curricula']

        stCur = stCur[['PUID_SIX', 'PRIMARY_PROGRAM_IND', 'STUDENT_CLASSIFICATION_BOAP', 'COLLEGE','MAJOR']]

        return stCur

    def constructor(self):
        df = pd.merge(self.HS(),self.ACT_SAT(),how='outer',on='PUID_SIX')
        df = pd.merge(df,self.curr_dem(),on='PUID_SIX',how='outer')
        df = pd.merge(df,self.AP(),on='PUID_SIX',how='outer')
        df = pd.merge(df,self.GPA(),on='PUID_SIX',how='outer')
        df = pd.merge(df, self.stud_curr(), on='PUID_SIX', how='outer')

        df['Key'] = df['PUID_SIX']

        df.drop(['PUID_SIX', 'ACADEMIC_PERIOD_x','ACADEMIC_PERIOD_DESC_x'],axis=1,inplace=True)

        return df

class combine:

    def __init__(self,df_2020,df_2021):
        self.df1 = df_2020
        self.df2 = df_2021

    def standard(self):

        current_2021 = ['Key', 'High_School_Grad_Date', 'High_School_GPA', 'MathGPA', 'MathYrTaken',
                        'PhysicsGPA', 'PhysicsYrTaken', 'Changed_Schools', 'ACT/SAT Math (avg)',
                        'ACT/SAT Non-Math (avg)', 'GENDER_DESC', 'REPORTING_ETHNICITY',
                        'UNDERREPRESENTED_MINORITY_IND', 'ADMISSIONS_ATTRIBUTE_DESC', 'COURSE_IDENTIFICATION',
                        'FINAL_GRADE', 'REPEAT_COURSE_IND', 'AP Math', 'AP Math Score', 'AP Phys', 'AP Phys Score',
                        'ACADEMIC_PERIOD_DESC_y', 'Prior_Overall_GPA_Term', 'Prior_Term_Overall_GPA','COLLEGE',
                        'MAJOR','PRIMARY_PROGRAM_IND','STUDENT_CLASSIFICATION_BOAP' ]

        target = ['Key', 'hsGradDate', 'hsGPA', 'hsMathGPA', 'hsMathYr','hsPhysGPA', 'hsPhysYr', 'ChangedSchools',
                  'ACT/SAT Math (avg)','ACT/SAT Non-Math (avg)', 'gender', 'ethnicity', 'underRepMin','firstGenCollege',
                  'course', 'finalGrade', 'repeatInd', 'AP Math','AP Math Score', 'AP Phys', 'AP Phys Score',
                  'academicPeriodDesc', 'collegeGPATerm','collegeGPA','college','major','primaryProgram', 'studentClassification']


        current_2020 = ['Key', 'SECONDARY_SCHOOL_GRAD_DATE', 'SCHOOL_GPA','MathGPA', 'MathYrTaken',
                        'PhysicsGPA', 'PhysicsYrTaken', 'Changed_Schools', 'ACT/SAT Math (avg)',
                        'ACT/SAT Non-Math (avg)', 'GENDER_DESC', 'REPORTING_ETHNICITY',
                        'UNDERREPRESENTED_MINORITY_IND', 'ADMISSIONS_ATTRIBUTE_DESC', 'COURSE_IDENTIFICATION',
                        'FINAL_GRADE','REPEAT_COURSE_IND','AP Math Level','AP Math Score', 'AP Phys Level', 'AP Phys Score',
                        'ACADEMIC_PERIOD_DESC', 'Most_Recent_Prior_Term','Most_Recent_Prior_Term_GPA','COLLEGE', 'MAJOR',
                        'PRIMARY_PROGRAM_IND', 'STUDENT_CLASSIFICATION_BOAP']

        std_dict_2020 = dict(zip(target,current_2020))
        std_dict_2021 = dict(zip(target,current_2021))

        df_2020 = pd.DataFrame(columns=target)
        df_2021 = pd.DataFrame(columns=target)

        for col in df_2020.columns:
            df_2020[col] = self.df1[std_dict_2020[col]]


        for col in df_2021.columns:
            df_2021[col] = self.df2[std_dict_2021[col]]

        df = pd.concat([df_2020,df_2021])

        return df

d1 = Make_Data_2020().constructor()
d2 = Make_Data_2021().constructor()

Data = combine(d1,d2).standard()
Data.reset_index(inplace=True,drop=True)

print(Data[Data['college'].map(type)==float]['academicPeriodDesc'].value_counts())







