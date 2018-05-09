# -*- coding: utf-8 -*-
"""
Created on Tue May  1 12:22:26 2018

@author: sande
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#data1 = pd.read_csv('../../Data/data_for_student_case.csv')

data = pd.read_csv('../../Data/data_for_student_case.csv')
non_fraud_data = data[data['simple_journal'] == 'Settled']
fraud_data = data[data['simple_journal'] == 'Chargeback']
# print data
plt.figure()
plt.title('Non Fraud Data')
sns.stripplot(x="currencycode", y="amount", data=non_fraud_data, jitter=True)
plt.figure()
plt.title('Fraud Data')
sns.stripplot(x="currencycode", y="amount", data=fraud_data, jitter=True)