# Registrar-Data-Project

In this project, I obtained data from the university registrar. The data cannot be included per institutional review board for privacy reasons. 
Two data sets were obtained, prepared by different analysts. The data sets contained mostly the same features but with different names. Call them
ds1 and ds2. ds1 and ds2 contained academic behavior features distributed amongst 7 independent records. Essentially, we use these academic/behavior
features as variables and the final course grade as the target. We try to predict students who will score C+ or below to identify "at-risk" students.

### Make_Data.py ###

-Pre-process features
    #High School GPA
    #ACT/SAT score
    #Gender
    #First Generation College Student Status
    #.....
-Creates a map between feature names in ds1 and the preferred name:
    #e.g. 'STUDENT_CLASSIFICAITON_BOAP' ==> 'studentClass'
-Creates a map between feature names in ds2 and the preferred name
-Combine features from ds1 into single pandas dataframe
-Combine features from ds2 into single pandas dataframe

-Combine the dataframes from ds1 and ds2 into single data set

# dataProcessing.py

-Takes the dataframe from Make_Data.py
-Drops rows for which the target variable is missing
-Z-scale numerical features (x-average)/standard deviation
-Mean encode categorical features

#ML.py

-Takes processed dataframe from dataProcessing.py
-Drops unnecessary columns
-Perform 5-fold cross validation on training set for model selection/parameter tuning
-fit/predict
-errors are categorized by grade level because we want to observe the performance
 at predicted at-risk students, namely "DFW" students
