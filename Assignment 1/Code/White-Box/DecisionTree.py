# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:26:46 2018

@author: sande
"""

# This first part of this script has been provided by the lecturers.
# We used their code to import the data and do some simple preprocessing
# The relevant code for this Task 2 (Imbalance) starts at line 144

get_ipython().magic(u'matplotlib inline')
import datetime
import time
import matplotlib.pyplot as plt
#import joblib
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from operator import itemgetter
from itertools import groupby
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn import tree
import graphviz
import scipy.sparse

def string_to_timestamp(date_string):#convert time string to float value
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)

if __name__ == "__main__":
    src = '../../Data/data_for_student_case.csv'
    ah = open(src, 'r')
    x = []#contains features
    y = []#contains labels
    data = []
    color = []
    (issuercountry_set, txvariantcode_set, currencycode_set, shoppercountry_set, interaction_set,
    verification_set, accountcode_set, mail_id_set, ip_id_set, card_id_set) = [set() for _ in range(10)]
    (issuercountry_dict, txvariantcode_dict, currencycode_dict, shoppercountry_dict, interaction_dict,
    verification_dict, accountcode_dict, mail_id_dict, ip_id_dict, card_id_dict) = [{} for _ in range(10)]
    #label_set
    #cvcresponse_set = set()
    ah.readline()#skip first line
    for line_ah in ah:
        if line_ah.strip().split(',')[9]=='Refused':# remove the row with 'refused' label, since it's uncertain about fraud
            continue
        if 'na' in str(line_ah.strip().split(',')[14]).lower() or 'na' in str(line_ah.strip().split(',')[4].lower()):
            continue
        bookingdate = string_to_timestamp(line_ah.strip().split(',')[1])# date reported flaud
        issuercountry = line_ah.strip().split(',')[2]#country code
        issuercountry_set.add(issuercountry)
        txvariantcode = line_ah.strip().split(',')[3]#type of card: visa/master
        txvariantcode_set.add(txvariantcode)
        issuer_id = float(line_ah.strip().split(',')[4])#bin card issuer identifier
        amount = float(line_ah.strip().split(',')[5])#transaction amount in minor units
        currencycode = line_ah.strip().split(',')[6]
        currencycode_set.add(currencycode)        
        shoppercountry = line_ah.strip().split(',')[7]#country code
        shoppercountry_set.add(shoppercountry)
        interaction = line_ah.strip().split(',')[8]#online transaction or subscription
        interaction_set.add(interaction)
        if line_ah.strip().split(',')[9] == 'Chargeback':
            label = 1#label fraud
        else:
            label = 0#label safe
        verification = line_ah.strip().split(',')[10]#shopper provide CVC code or not
        verification_set.add(verification)
        cvcresponse = line_ah.strip().split(',')[11]#0 = Unknown, 1=Match, 2=No Match, 3-6=Not checked
        if int(cvcresponse) > 2:
            cvcresponse = 3
        year_info = datetime.datetime.strptime(line_ah.strip().split(',')[12],'%Y-%m-%d %H:%M:%S').year
        month_info = datetime.datetime.strptime(line_ah.strip().split(',')[12],'%Y-%m-%d %H:%M:%S').month
        day_info = datetime.datetime.strptime(line_ah.strip().split(',')[12],'%Y-%m-%d %H:%M:%S').day
        creationdate = str(year_info)+'-'+str(month_info)+'-'+str(day_info)#Date of transaction 
        creationdate_stamp = string_to_timestamp(line_ah.strip().split(',')[12])#Date of transaction-time stamp
        accountcode = line_ah.strip().split(',')[13]#merchant’s webshop
        accountcode_set.add(accountcode)
        mail_id = int(float(line_ah.strip().split(',')[14].replace('email','')))#mail
        mail_id_set.add(mail_id)
        ip_id = int(float(line_ah.strip().split(',')[15].replace('ip','')))#ip
        ip_id_set.add(ip_id)
        card_id = int(float(line_ah.strip().split(',')[16].replace('card','')))#card
        card_id_set.add(card_id)
        data.append([issuercountry, txvariantcode, issuer_id, amount, currencycode,
                    shoppercountry, interaction, verification, cvcresponse, creationdate_stamp,
                     accountcode, mail_id, ip_id, card_id, label, creationdate])# add the interested features here
        #y.append(label)# add the labels
    data = sorted(data, key = lambda k: k[-1])

# In[6]:

for item in data:#split data into x,y
    x.append(item[0:-2])
    y.append(item[-2])
'''map number to each categorial feature'''
for item in list(issuercountry_set):
    issuercountry_dict[item] = list(issuercountry_set).index(item)
for item in list(txvariantcode_set):
    txvariantcode_dict[item] = list(txvariantcode_set).index(item)
for item in list(currencycode_set):
    currencycode_dict[item] = list(currencycode_set).index(item)
for item in list(shoppercountry_set):
    shoppercountry_dict[item] = list(shoppercountry_set).index(item)
for item in list(interaction_set):
    interaction_dict[item] = list(interaction_set).index(item)
for item in list(verification_set):
    verification_dict[item] = list(verification_set).index(item)
for item in list(accountcode_set):
    accountcode_dict[item] = list(accountcode_set).index(item)
print("Number of distinct card_ids:", len(list(card_id_set)))
#for item in list(card_id_set):
#    card_id_dict[item] = list(card_id_set).index(item)
'''modify categorial feature to number in data set'''
for item in x:
    item[0] = issuercountry_dict[item[0]]
    item[1] = txvariantcode_dict[item[1]]
    item[4] = currencycode_dict[item[4]]
    item[5] = shoppercountry_dict[item[5]]
    item[6] = interaction_dict[item[6]]
    item[7] = verification_dict[item[7]]
    item[10] = accountcode_dict[item[10]]

x_array = np.array(x)
y_array = np.array(y)

# Split data into training and testing records
x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size = 0.2, random_state=0)


print("decision tree Classifier")
clf = tree.DecisionTreeClassifier(max_depth=10)
clf.fit(x_train, y_train)

y_pred_smote_randomforest = clf.predict_proba(x_test)[:,1]
y_pred_smote_randomforest_bin = np.around(y_pred_smote_randomforest)
print("recall score:", recall_score(y_test, y_pred_smote_randomforest_bin))
print("precision score:", precision_score(y_test, y_pred_smote_randomforest_bin))
print("f1 score:", f1_score(y_test, y_pred_smote_randomforest_bin))            

dot_data = tree.export_graphviz(clf, out_file=None,class_names=["non-fraud","fraud"],filled=True,impurity=True,feature_names=["issuercountry", "txvariantcode", "issuer_id", "amount_in_eur", "currencycode",
                    "shoppercountry", "interaction", "verification", "cvcresponse", "creationdate_stamp",
                     "accountcode", "mail_id", "ip_id", "card_id"], node_ids=True) 
graph = graphviz.Source(dot_data) 
# Save the decision tree in a pdf file names cardtree
graph.render("cardtree") 


path = clf.decision_path(x_test)
#print(type(path))
#print(path)
pred = clf.predict(x_test)
#Flags to stop after finding one true positive and one true negative
flag1 = False
flag2 = False
index_of_TN = 0
index_of_TP = 0
for i in range(1000,len(x_test)):
    if flag1 == False :
        if (pred[i] == y_test[i] == 0):
            print ("True Negative at index ", i)
            index_of_TN = i
            print (pred[i],y_test[i])
            print ("for row ",x_test[i])
            flag1 = True
    if flag2 == False :
        if (pred[i] == y_test[i] == 1):
            print ("True positive at index ", i)
            print (pred[i],y_test[i])
            print ("for row ",x_test[i])
            index_of_TP = i
            if(flag1 == True):
                break
        

cx = scipy.sparse.coo_matrix(path)

#find relevant entries in the decision tree for True negative case and true positive case
for i,j,v in zip(cx.row, cx.col, cx.data):
    if i == index_of_TN :
        print ("Nodes for True Negative")
        print ("(%d, %d), %s" % (i,j,v))
    if i == index_of_TP:
        print ("Nodes for True positive")
        print ("(%d, %d), %s" % (i,j,v))
        
'''
#for i in range(0,len(x_test)):
#    if (pred[i] == 1):
#        print ("positive at index ", i)
#        print (pred[i],y_test[i])
#find the best SMOTE ratio
#i = 0.05               
#while (i<=1.0):
#    sm = SMOTE(ratio=i)
#    
#    x_train_smote = x_train.astype(float)
#    y_train_smote = y_train.astype(float)
#    x_train_res, y_train_res = sm.fit_sample(x_train_smote, y_train_smote) 
#    print ("for ratio ",i)
#    print("SMOTEd training set:", sum(y_train_res), "/", len(y_train_res), "records are fraud ==> smote_ratio =", i)
#    clf.fit(x_train_res, y_train_res)
#    
#    #print ("after SMOTE")
#    y_pred_smote_randomforest = clf.predict_proba(x_test)[:,1]
#    y_pred_smote_randomforest_bin = np.around(y_pred_smote_randomforest)
#    #print("recall score:", recall_score(y_test, y_pred_smote_randomforest_bin))
#    #print("precision score:", precision_score(y_test, y_pred_smote_randomforest_bin))
#    print("f1 score:", f1_score(y_test, y_pred_smote_randomforest_bin))
#    i = i+0.05

##Among all the SMOTE ratios, a ratio of 0.05 had the best f1 score
#sm = SMOTE(ratio=0.05)
#    
#x_train_smote = x_train.astype(float)
#y_train_smote = y_train.astype(float)
#x_train_res, y_train_res = sm.fit_sample(x_train_smote, y_train_smote) 
#print ("for ratio ",i)
#print("SMOTEd training set:", sum(y_train_res), "/", len(y_train_res), "records are fraud ==> smote_ratio =", i)
#clf.fit(x_train_res, y_train_res)
#    
#print ("after SMOTE")
#y_pred_smote_randomforest = clf.predict_proba(x_test)[:,1]
#y_pred_smote_randomforest_bin = np.around(y_pred_smote_randomforest)
#print("recall score:", recall_score(y_test, y_pred_smote_randomforest_bin))
#print("precision score:", precision_score(y_test, y_pred_smote_randomforest_bin))
#print("f1 score:", f1_score(y_test, y_pred_smote_randomforest_bin))
        
#try for difference combinationsfor max depth and min sample split and obtain the better performing ones
#for i in [2,3,4,5,6,7,8,9]:
#    #for j in [6,9,12,15,21,27,30,33,37]:
#    for j in [30,33,37]:
#        print("decision tree Classifier for ",i,j)
#        clf = tree.DecisionTreeClassifier(min_samples_split = i,max_depth=j, random_state = 12)
#        clf.fit(x_train, y_train)
#        y_pred_smote_randomforest = clf.predict_proba(x_test)[:,1]
#        y_pred_smote_randomforest_bin = np.around(y_pred_smote_randomforest)
#        print("f1 score:", f1_score(y_test, y_pred_smote_randomforest_bin))
#    

'''