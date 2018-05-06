# -*- coding: utf-8 -*-
"""
Created on Sat May  5 11:08:56 2018

@author: sande
"""


# coding: utf-8

# # Credit Card  Fraud Detection Lab!
# Welcome! This is the lab section of CS4035-Cyber Data Analytics, TUDelft.
# Questions: email Sicco Verwer s.e.verwer@tudelft.nl, Andre Teixeira andre.teixeira@tudelft.nl and Qin Lin q.lin@tudelft.nl.
# 
# ## Download libruary
# * Machine learning libruary: http://scikit-learn.org/stable/
# * Sampling: https://github.com/fmfn/UnbalancedDataset
# 
# ## Reference
# * Unbalanced data
#     1. Learning from Imbalanced Data, He et al. (**improtant!**)
#     2. Cost-sensitve boosting for classification of imbalanced data, Sun et al.
#     3. SMOTE: Synthetic Minority Over-sampling Technique, Chawla et al.
# * Fraud detection
#     1. Data mining for credit card fraud: A comparative study, Bhattacharyya et al.
#     2. Minority Report in Fraud Detection: Classification of Skewed Data, Phua et al.
# 

# In[7]:

get_ipython().magic(u'matplotlib inline')
import datetime
import time
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from operator import itemgetter
from itertools import groupby
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

def string_to_timestamp(date_string):#convert time string to float value
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)
def aggregate(before_aggregate, aggregate_feature):
    if aggregate_feature == 'day':
        after_aggregate = []
        pos_date = -1
        before_aggregate.sort(key = itemgetter(9))#sort by timestamp
        temp = groupby(before_aggregate, itemgetter(-2))
        group_unit = []
        mean = []
        for i, item in temp:# i is group id
            for jtem in item:# unit in each group
                group_unit.append(jtem)
            #for feature_i in range(6):
            #    mean.append(zip(group_unit)[feature_i])
            #after_aggregate.append(group_unit)
            after_aggregate.append(mean)
            group_unit = []
        #print after_aggregate[0]
        #print before_aggregate[0]
    if aggregate_feature == 'client':
        after_aggregate = []
        pos_client = -3
        before_aggregate.sort(key = itemgetter(pos_client))#sort with cardID firstly，if sort with 2 feature, itemgetter(num1,num2)
        temp = groupby(before_aggregate, itemgetter(pos_client))#group
        group_unit = []
        for i, item in temp:# i is group id
            for jtem in item:# unit in each group
                group_unit.append(jtem)
            after_aggregate.append(group_unit)
            group_unit = []
    return after_aggregate
def aggregate_mean(before_aggregate):
    #print before_aggregate[0]
    if True:
        after_aggregate = []
        pos_date = -1
        before_aggregate.sort(key = itemgetter(-1))#sort by timestamp
        temp = groupby(before_aggregate, itemgetter(-1))
        group_unit = []
        mean = []
        for i, item in temp:# i is group id
            for jtem in item:# unit in each group
                group_unit.append(list(jtem))
            #print group_unit
            if len(zip(group_unit)) < 2:
                after_aggregate.append(group_unit)
                group_unit = []
            if len(zip(group_unit)) >= 2:
                #print zip(group_unit)
                for feature_i in range(14):
                    #print zip(group_unit)[feature_i]
                    mean.append(sum(zip(*group_unit)[feature_i])/len(zip(group_unit)))
                after_aggregate.append(mean)
                group_unit = []
                mean = []
        #print after_aggregate[0]
        #print before_aggregate[0]
    return after_aggregate
if __name__ == "__main__":
    #src = '/Users/Qlin/Documents/TA/fraud_credit_card/data_for_student_case.csv'
    src = '../Data/data_for_student_case.csv'
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
            label = 0#label save
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
    '''    
    day_aggregate = aggregate(data,'day')
    client_aggregate = aggregate(data,'client')
    transaction_num_day = []
    for item in day_aggregate:
        transaction_num_day.append(len(item))
    plt.figure(1)
    plt.plot(transaction_num_day, color = 'c', linewidth = 2)
    plt.plot()
    plt.text(2,0.0,'Date: 2015-10-8')
    plt.xlabel('Date')
    plt.ylabel('Number of Transactions')
    plt.xlim([0,125])
    plt.axis('tight')
    plt.savefig('Day Aggregating.png')
    transaction_num_client = []
    for item in client_aggregate:
        transaction_num_client.append(len(item))
    plt.figure(2)
    plt.plot(transaction_num_client, color = 'c', linewidth = 2)
    #plt.text(99,9668,'Date: 2015-10-8')
    plt.xlabel('Client ID')
    plt.ylabel('Number of Transactions')
    plt.axis('tight')
    plt.savefig('Client Aggregating.png')
    '''

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
print (len(list(card_id_set)))
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

#x_mean = []
#x_mean = aggregate_mean(x);
x_mean = x;
#des = '/Users/Qlin/Documents/TA/fraud_credit_card/original_data.csv'
des = 'des_csv.csv'
#des1 = '/Users/Qlin/Documents/TA/fraud_credit_card/aggregate_data.csv'
ch_dfa = open(des,'w')
#ch_dfa.write('txid,bookingdate,issuercountrycode,txvariantcode,bin,amount,'+
#             'currencycode,shoppercountrycode,shopperinteraction,simple_journal,'+
 #            'cardverificationcodesupplied,cvcresponsecode,creationdate,accountcode,mail_id,ip_id,card_id')
#ch_dfa.write('\n')
sentence = []
for i in range(len(x_mean)):
    for j in range(len(x_mean[i])):
        sentence.append(str(x_mean[i][j]))
    sentence.append(str(y[i]))
    ch_dfa.write(' '.join(sentence))
    ch_dfa.write('\n')
    sentence=[]
    ch_dfa.flush()    
TP, FP, FN, TN = 0, 0, 0, 0
x_array = np.array(x)
y_array = np.array(y)
usx = x_array
usy = y_array

x_train, x_test, y_train, y_test = train_test_split(usx, usy, test_size = 0.2, random_state=42)#test_size: proportion of train/test data
#apply random forest classifier with default parameters 
print ("Random Forest Classifier")
clf = RandomForestClassifier()
clf.fit(x_train, y_train)
#trained_model = random_forest_classifier(train_x, train_y)
    #print ("Trained model :: ", trained_model)
predictions = clf.predict(x_test)

for i in range(0, 5):
    print ("Actual outcome :: {} and Predicted outcome :: {}".format(list(y_test)[i], predictions[i]))

#print ("Train Accuracy :: ", accuracy_score(x_train, clf.predict(x_train)))
#print ("Test Accuracy  :: ", accuracy_score(y_test, predictions))
print ("recall score")
print (recall_score(y_test, predictions))
print ("accuracy score")    
print (accuracy_score(y_test, predictions))

#apply SMOTE 
print ("after SMOTEing")
sm = SMOTE(random_state=12, ratio = 1.0) 
x_train = x_train.astype(float)
y_train = y_train.astype(float)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train) 
clf.fit(x_train_res, y_train_res)
pred = clf.predict(x_test)
print ("recall score")
print (recall_score(y_test, pred))
print ("accuracy score")
print (accuracy_score(y_test, pred))



print("after logistic regression")
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
x_test = x_test.astype(float)
predictionsLogis = logreg.predict(x_test)

print (recall_score(y_test, predictionsLogis))
print ("accuracy score")    
print (accuracy_score(y_test, predictionsLogis))

print ("After applying SMOTE")
x_train_res, y_train_res = sm.fit_sample(x_train, y_train) 

logreg.fit(x_train_res, y_train_res)
pred = logreg.predict(x_test)
print ("recall score")
print (recall_score(y_test, pred))
print ("accuracy score")
print (accuracy_score(y_test, pred))

print ("KNN algorithm")
clf = neighbors.KNeighborsClassifier(algorithm = 'kd_tree')
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)

print (recall_score(y_test, y_predict))
print ("accuracy score")    
print (accuracy_score(y_test, y_predict))


print ("After applying SMOTE")
x_train_res, y_train_res = sm.fit_sample(x_train, y_train) 

clf.fit(x_train_res, y_train_res)
pred = clf.predict(x_test)
print ("recall score")
print (recall_score(y_test, pred))
print ("accuracy score")
print (accuracy_score(y_test, pred))

print("ROC curve")
pred_proba = clf.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test[:, 1], pred_proba)

plt.figure()
lw = 1
#plt.plot(fpr[2], tpr[2], color='darkorange',
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()