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


## Imports 2020 data from folder ##

class Data_2020:

    ## Imports 2020 data from folder ##
    files = [file for file in os.walk('./Aug2020_Data/')][0][2]
    data_2020 = {}
    print(files)

    ## Formats data as pd.DataFrame ##
    for file in files:
        data_2020[file.split('.')[0]] = pd.read_csv('./Aug2020_Data/' + file, dtype=str)

    def __init__(self):
        self.data = data_2020


    def GPA(self):

        ################################################
        #                                              #
        #        Useful Features are Taken From        #
        #           data_2020 and data_20201           #
        #               Prior Term GPA                 #
        #                                              #
        ################################################

        d2020_cols = ('Key', 'ACADEMIC_PERIOD', 'FINAL_GRADE', 'REPEAT_COURSE_IND', 'Most_Recent_Prior_Term',
                      'Most_Recent_Prior_Term_GPA')

        GPA = self.data_2020['Prior Term GPA'].loc[:, d2020_cols]
        return GPA


    def ACT_SAT(self):

        ###################
        #                 #
        #     ACT/SAT     #
        #                 #
        ###################

        ####################################### ACT/SAT Math ###########################################################
        ## ACT/SAT in 2 different records, grab relevant math columns ##
        ACTSAT_MATH_1 = self.data_2020['ACT - SAT'][['Key', 'ACT_Math', 'SAT_Mathematics']]
        ACTSAT_MATH_2 = self.data_2020['ACT2SAT'][['Key', 'SATR Math', 'ACT Math']]

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

        ACTSAT_NONMATH_1 = self.data_2020['ACT - SAT'].loc[:, dnm_cols]
        ACTSAT_NONMATH_2 = self.data_2020['ACT2SAT'].loc[:,dnm_cols2]
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

        return ACTSAT

    def dem(self):

        ##########################
        #                        #
        #   Take Useful Feats    #
        #    from Demographic    #
        #  for 2020 and 2021     #
        #                        #
        ##########################

        d2020_cols = ('Key', 'GENDER_DESC', 'REPORTING_ETHNICITY', 'UNDERREPRESENTED_MINORITY_IND',
                      'ADMISSIONS_ATTRIBUTE_DESC')

        return self.data_2020['Demographics'].loc[:, d2020_cols]

    def curr(self):

        ##########################
        #                        #
        #   Take Useful Feats    #
        #    from Curricula      #
        #  for 2020 and 2021     #
        #                        #
        ##########################

        d2020_cols = ('Key', 'ACADEMIC_PERIOD', 'COLLEGE', 'MAJOR', 'STUDENT_CLASSIFICATION_BOAP', 'STUDENT_STATUS')

        return self.data_2020['Curricula'].loc[:, d2020_cols]

    def AP(self):

        ##########################
        #                        #
        #   Take Useful Feats    #
        #    from AP Classes     #
        #   for 2020 and 2021    #
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
        AP = self.data_2020['AP Tests']
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
        keys = list(data_2020['AP Tests']['Key'].unique())
        AP_Final['Key'] = keys

        AP_Final = pd.merge(AP_Final, AP_Math, on='Key', how='outer')
        AP_Final = pd.merge(AP_Final, AP_Phys, on='Key', how='outer')
        # return AP_Final
        # # AP_Final = AP_Final.fillna(0)
        # # return AP_Final










