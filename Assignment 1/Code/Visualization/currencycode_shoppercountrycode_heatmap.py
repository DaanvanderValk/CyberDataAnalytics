# -*- coding: utf-8 -*-
"""
Created on Tue May  1 10:36:03 2018

@author: Daan
"""
import time
import numpy as np
import seaborn as sns
import pandas as pd

def string_to_timestamp(date_string):#convert time string to float value
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)

if __name__ == "__main__":        
    # Import CSV file to a Panda dataframe
    df = pd.read_csv('../../Data/data_for_student_case.csv')
    print("Imported", len(df), "records")
    
#    print(df.keys())             # These are the keys:
#       'txid', 'bookingdate', 'issuercountrycode', 'txvariantcode', 'bin',
#       'amount', 'currencycode', 'shoppercountrycode', 'shopperinteraction',
#       'simple_journal', 'cardverificationcodesupplied', 'cvcresponsecode',
#       'creationdate', 'accountcode', 'mail_id', 'ip_id', 'card_id'
    
    # Only select items we need
    df = df[['simple_journal', 'currencycode', 'shoppercountrycode']]
    df = df.dropna()
    
    # Only take the verified fraud and non-fraud cases
    # Note that "Refused" records are ignored, as we don't know wether it's due to fraud or something else.
    df_settled = df[df['simple_journal'] == 'Settled'][['currencycode', 'shoppercountrycode']]
    df_chargeback = df[df['simple_journal'] == 'Chargeback'][['currencycode', 'shoppercountrycode']]
    print(len(df_settled), '"Settled" records')
    print(len(df_chargeback), '"Chargeback" records')
    
    currencies = df.currencycode.unique()
    shoppercountries = df.shoppercountrycode.unique()
    
    occurances_settled = []
    occurances_chargeback = []
    
    for i in range(len(currencies)):
        currency_occurances_fraud = []
        for j in range(len(shoppercountries)):
            currency_occurances_fraud.append(len(df_chargeback[df_chargeback['currencycode'] == currencies[i]][df_chargeback['shoppercountrycode'] == shoppercountries[j]]))
        
        occurances_settled.append(currency_occurances_fraud)
    
    df_chargeback_heatmap = pd.DataFrame(occurances_settled, columns = shoppercountries, index = currencies)
    #sns.clustermap(df_settled_heatmap)
    
    sns.heatmap(df_chargeback_heatmap)