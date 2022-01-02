# Registrar-Data-Project

In this project, I obtained data from the university registrar. The data cannot be included per institutional review board for privacy reasons. 
Two data sets were obtained, prepared by different analysts. The data sets contained mostly the same features but with different names. Call them
ds1 and ds2. ds1 and ds2 contained academic behavior features distributed amongst 7 independent records. Essentially, we use these academic/behavior
features as variables and the final course grade as the target. We try to predict students who will score C+ or below to identify "at-risk" students.

### Make_Data.py ###

<ul>
    <li>Pre-process features</li>
        <ul>
            <li>High School GPA</li>
            <li>ACT/SAT score</li>
            <li>Gender</li>
            <li>First Generation College Student Status</li>
            <li>.....</li>
        </ul>
    <li>Creates a map between feature names in ds1 and the preferred name:</li>
        <ul>
            <li>e.g. 'STUDENT_CLASSIFICAITON_BOAP' ==> 'studentClass'</li>
        </ul>
    <li>Creates a map between feature names in ds2 and the preferred name</li>
    <li>Combine features from ds1 into single pandas dataframe</li>
    <li>Combine features from ds2 into single pandas dataframe</li>
    <li>Combine the dataframes from ds1 and ds2 into single data set</li>
 </ul>

# dataProcessing.py
<ul>
    <li>Takes the dataframe from Make_Data.py</li>
    <li>Drops rows for which the target variable is missing</li>
    <li>Z-scale numerical features (x-average)/standard deviation</li>
    <li>Mean encode categorical features</li>
</ul>    

# ML.py
<ul>
    <li>Takes processed dataframe from dataProcessing.py</li>
    <li>Drops unnecessary columns</li>
    <li>Perform 5-fold cross validation on training set for model selection/parameter tuning</li>
    <li>fit/predict</li>
    <li>errors are categorized by grade level because we want to observe the performance
        at predicted at-risk students, namely "DFW" students</li>
</ul>
