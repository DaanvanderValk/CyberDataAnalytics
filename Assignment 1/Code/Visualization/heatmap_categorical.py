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
import sys

from imblearn.over_sampling import SMOTE

def string_to_timestamp(date_string): #convert time string to float value
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)

if __name__ == "__main__":
    # SETTINGS
    
    # Dimensions of the heatmaps; useful for tweaking.
    x_size = 18
    y_size = 3
    
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
    feature1_values = df[feature1].unique()
    feature2_values = df[feature2].unique()
    
    print(feature1, "values:", feature1_values)
    print(feature2, "values:", feature2_values)
    
    # Only take the verified fraud and non-fraud cases
    # Note that "Refused" records are ignored, as we don't know wether it's due to fraud or something else.
    df_settled = df[df['simple_journal'] == 'Settled']
    print('"Settled" records:', len(df_settled))
    df_chargeback = df[df['simple_journal'] == 'Chargeback']
    print('"Chargeback" records:', len(df_chargeback))
    
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
    # Also relative matrixes are computed (containing percentages)
    settled_pivot = settled_occurence.pivot(feature1, feature2).fillna(0)
    chargeback_pivot = chargeback_occurence.pivot(feature1, feature2).fillna(0)
    
    
    for feature1_value in feature1_values:
        if feature1_value not in settled_pivot.index.values:
            # Force empty row into dataframe
            settled_pivot.loc[feature1_value] = 0.0
            
        if feature1_value not in chargeback_pivot.index.values:
            # Force empty row into dataframe
            chargeback_pivot.loc[feature1_value] = 0.0
            
    
    for feature2_value in feature2_values:
        if ('', feature2_value) not in settled_pivot.columns:
            # Force empty column into dataframe
            settled_pivot['', feature2_value] = 0.0;
            
        if ('', feature2_value) not in chargeback_pivot.columns:
            # Force empty column into dataframe
            chargeback_pivot['', feature2_value] = 0.0;
            
    # Reorder indexes of the dataframes
    settled_pivot = settled_pivot.sort_index(axis=1)
    chargeback_pivot = chargeback_pivot.sort_index(axis=1)
    
    # Compute percentages
    settled_percentages = settled_pivot / settled_pivot.sum().sum()
    chargeback_percentages = chargeback_pivot / chargeback_pivot.sum().sum()
    
    subtracted_percentages = settled_percentages - chargeback_percentages
    
    # Because the distribution of such occurances is far from linear, we usually want to look
    # at the graph on a logarithmic scale. This is achieved by replacing each value in the
    # dataframe to log(1 + value). This maps the interval [0, <very high values>] to
    # [0, <relatively low value>], which is exactly what we want.
    settled_pivot_log = np.log(1 + settled_pivot)
    chargeback_pivot_log = np.log(1 + chargeback_pivot)
    

        
#    # Plot the heatmaps
#    # 1. Settled - linear scale
#    plt.subplots(figsize=(x_size, y_size))
#    ax_normal = plt.axes()
#    sns.heatmap(settled_pivot, ax = ax_normal)
#    ax_normal.set_title('1. Settled records on a linear scale')
#    ax_normal.set_xlabel(feature2)
#    ax_normal.set_ylabel(feature1)

    # 2. Settled - logarithmic scale
    plt.subplots(figsize=(x_size, y_size))
    ax_normal = plt.axes()
    sns.heatmap(settled_pivot_log, ax = ax_normal, cmap="Blues")
    ax_normal.set_title('2. Settled records (logarithmic scale)')
    ax_normal.set_xlabel(feature2)
    ax_normal.set_ylabel(feature1)
    
#    # 3. Fraudulent - linear scale
#    plt.subplots(figsize=(x_size, y_size))
#    ax_normal = plt.axes()
#    sns.heatmap(chargeback_pivot, ax = ax_normal)
#    ax_normal.set_title('3. Fraudulent records on a linear scale')
#    ax_normal.set_xlabel(feature2)
#    ax_normal.set_ylabel(feature1)
#    
    # 4. Fraudulent - logarithmic scale
    plt.subplots(figsize=(x_size, y_size))
    ax_normal = plt.axes()
    sns.heatmap(chargeback_pivot_log, ax = ax_normal, cmap="Blues")
    ax_normal.set_title('4. Fraudulent records (logarithmic scale)')
    ax_normal.set_xlabel(feature2)
    ax_normal.set_ylabel(feature1)
    
    
    # 5. Settled - linear scale
    plt.subplots(figsize=(x_size, y_size))
    ax_normal = plt.axes()
    sns.heatmap(subtracted_percentages, ax = ax_normal, center=0, cmap="bwr_r")
    ax_normal.set_title('5. Differences in relative occurances (blue: less fraud, red: more fraud)')
    ax_normal.set_xlabel(feature2)
    ax_normal.set_ylabel(feature1)