# -*- coding: utf-8 -*-
"""
Created on Tue May  1 10:42:59 2018

@author: sande
"""

import pandas as pd

data = pd.read_csv('../../Data/data_for_student_case.csv')
non_fraud_data = data[data['simple_journal'] == 'Authorised']
fraud_data = data[data['simple_journal'] == 'Chargeback']
print (fraud_data[:3])