# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 19:02:03 2020

@author: jerem
"""


import pandas as pd
import numpy as np

#Import data as pandas dataframes
actsat = pd.read_csv('ACT - SAT.csv',dtype = str)
acttosat = pd.read_csv('ACT2SAT.csv',dtype = str)
aptest = pd.read_csv('AP Tests.csv',dtype = str)
ptgpa = pd.read_csv('Prior Term GPA.csv',dtype =str)
curr = pd.read_csv('Curricula.csv',dtype = str)
demo = pd.read_csv('Demographics.csv',dtype = str)
hs = pd.read_csv('Highschool.csv',dtype = str)
SP2020_220 = pd.read_csv('midterm.csv',dtype = str)
    
# Make a list of Phys 220 Sp 2020 Students
PUID = SP2020_220['Student ID'].to_list()
PUID = [str(ID) for ID in PUID]
PUID = [ID[len(ID)-7:len(ID)-1] for ID in PUID]

def act2sat(text):
    try:
        x = int(text)
        x = x*(800/36)
        return '{:.2f}'.format(x)    
    except:
        return text
    
def nan2zero(text):
    if type(text) == float:
        return 0
    else:
        return int(float(text))
    
IDS = actsat['Key'].to_list()
Data = []

Math = pd.merge(actsat,acttosat,on = 'Key')
Math = Math[['Key','ACT_Math','SAT_Mathematics','SATR Math']]
Math.fillna(0,inplace = True)
Math['ACT_Math'] = Math['ACT_Math'].apply(int).apply(lambda x: x*(800/36)).apply(int)
Math['SAT_Mathematics'] = Math['SAT_Mathematics'].apply(int)
Math['SATR Math'] = Math['SATR Math'].apply(int)
Math['Score'] = Math[['ACT_Math','SAT_Mathematics','SATR Math']].values.max(axis = 1)
    
