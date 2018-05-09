# -*- coding: utf-8 -*-
"""
Created on Wed May  2 10:49:55 2018

@author: sande
"""
import seaborn as sns
import pandas as pd

data = pd.read_csv('../../Data/data_for_student_case.csv')

g = sns.factorplot(x="currencycode", y="amount", hue="simple_journal", data=data,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("amount")