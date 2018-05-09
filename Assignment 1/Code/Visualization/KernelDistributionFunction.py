# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:43:16 2018

@author: sande
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  1 10:42:59 2018

@author: sande
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('../../Data/data_for_student_case.csv')
non_fraud_data = data[data['simple_journal'] == 'Settled']
fraud_data = data[data['simple_journal'] == 'Chargeback']
#limited_fraud = fraud_data[fraud_data['amount'] < 900000] 
#limited_non_fraud = non_fraud_data[non_fraud_data['amount'] < 900000] 
#print (non_fraud_data[:3])

#fig = plt.figure(figsize=(20,20))
#plt.xlim(-1000,900000)
#fig.add_subplot(1,1,1)
#sns.kdeplot(non_fraud_data['amount'], label='not fraud')
#sns.kdeplot(fraud_data['amount'], label='fraud')
plt.xlabel("Amount")
plt.ylabel("Probability")
plt.show()

print ("hi")
sns.kdeplot(np.log(non_fraud_data['amount']), label='not fraud')
sns.kdeplot(np.log(fraud_data['amount']), label='fraud')


#plt.yticks(fig.get_yticks(), fig.get_yticks() * 100)
#plt.ylabel('Distribution [%]', fontsize=16)

print ("Difference")
#sns.kdeplot(fraud_data['amount'])

#df3 = fraud_data[fraud_data['amount'] < 100000] 
#print (df3)