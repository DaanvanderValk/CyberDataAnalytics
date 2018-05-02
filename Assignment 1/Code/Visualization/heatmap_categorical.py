# -*- coding: utf-8 -*-
"""
Created on Tue May  1 10:36:03 2018

@author: Daan

This script generates heatmaps for 2 arbitrary categorical values.
"""
import time
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def string_to_timestamp(date_string): #convert time string to float value
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)

if __name__ == "__main__":
    # SETTINGS
    
    # Dimensions of the heatmaps; useful for tweaking.
    x_size = 18
    y_size = 5
    
    # Select the features to be plotted here
    # The features should be categorical, as their unique values are used
    feature1 = 'currencycode'
    feature2 = 'shoppercountrycode'

#    print(df.keys())             # These are the features:
#       'txid', 'bookingdate', 'issuercountrycode', 'txvariantcode', 'bin',
#       'amount', 'currencycode', 'shoppercountrycode', 'shopperinteraction',
#       'simple_journal', 'cardverificationcodesupplied', 'cvcresponsecode',
#       'creationdate', 'accountcode', 'mail_id', 'ip_id', 'card_id'
    
    # Import CSV file to a Panda dataframe
    df = pd.read_csv('../../Data/data_for_student_case.csv')
    print("Imported", len(df), "records")
    
    # Only select items we need
    df = df[['simple_journal', feature1, feature2]]
    
    # Drop empty values
    df = df.dropna()
    
    # Show which distinct values the selected features can take
    print(feature1, "values:", df[feature1].unique())
    print(feature2, "values:", df[feature2].unique())
    
    # Only take the verified fraud and non-fraud cases
    # Note that "Refused" records are ignored, as we don't know wether it's due to fraud or something else.
    df_settled = df[df['simple_journal'] == 'Settled']
    print(len(df_settled), '"Settled" records')
    df_chargeback = df[df['simple_journal'] == 'Chargeback']
    print(len(df_chargeback), '"Chargeback" records')
    
    # To make the heatmap, we need to group the occurances of combinations of both features
    settled_grouped = df_settled.groupby([feature1, feature2]).size()
    settled_occurence = settled_grouped.reset_index()
    settled_occurence.columns = [feature1, feature2, '']

    chargeback_grouped = df_chargeback.groupby([feature1, feature2]).size()
    chargeback_occurence = chargeback_grouped.reset_index()
    chargeback_occurence.columns = [feature1, feature2, '']

    # Finally, a pivot of this data will be the input of our heatmap
    # This is a 2-dimensional dataframe (feature1 X feature2)
    # The values in the dataframe indicate the number of occurances of the x and y values
    settled_pivot = settled_occurence.pivot(feature1, feature2).fillna(0)
    chargeback_pivot = chargeback_occurence.pivot(feature1, feature2).fillna(0)

    # Because the distribution of such occurances is far from linear, we usually want to look
    # at the graph on a logarithmic scale. This is achieved by replacing each value in the
    # dataframe to log(1 + value). This maps the interval [0, <very high values>] to
    # [0, <relatively low value>], which is exactly what we want.
    settled_pivot_log = np.log(1 + settled_pivot)
    chargeback_pivot_log = np.log(1 + chargeback_pivot)

    # Plot the heatmaps
    # 1. Settled - linear scale
    plt.subplots(figsize=(x_size, y_size))
    ax_normal = plt.axes()
    sns.heatmap(settled_pivot, ax = ax_normal)
    ax_normal.set_title('1. Settled records on a linear scale')
    ax_normal.set_xlabel(feature2)
    ax_normal.set_ylabel(feature1)
    plt.show()
    
    # 2. Settled - logarithmic scale
    plt.subplots(figsize=(x_size, y_size))
    ax_normal = plt.axes()
    sns.heatmap(settled_pivot_log, ax = ax_normal)
    ax_normal.set_title('2. Settled records on a logarithmic scale')
    ax_normal.set_xlabel(feature2)
    ax_normal.set_ylabel(feature1)
    plt.show()
    
    # 3. Fraudulent - linear scale
    plt.subplots(figsize=(x_size, y_size))
    ax_normal = plt.axes()
    sns.heatmap(chargeback_pivot, ax = ax_normal)
    ax_normal.set_title('3. Fraudulent records on a linear scale')
    ax_normal.set_xlabel(feature2)
    ax_normal.set_ylabel(feature1)
    plt.show()
    
    # 4. Fraudulent - logarithmic scale
    plt.subplots(figsize=(x_size, y_size))
    ax_normal = plt.axes()
    sns.heatmap(chargeback_pivot_log, ax = ax_normal)
    ax_normal.set_title('4. Fraudulent records on a logarithmic scale')
    ax_normal.set_xlabel(feature2)
    ax_normal.set_ylabel(feature1)
    plt.show()