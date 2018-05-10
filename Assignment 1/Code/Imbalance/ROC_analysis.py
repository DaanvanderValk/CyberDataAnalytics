# This first part of this script has been provided by the lecturers.
# We used their code to import the data and do some simple preprocessing
# The relevant code for this Task 2 (Imbalance) starts at line 144

get_ipython().magic(u'matplotlib inline')
import datetime
import time
import matplotlib.pyplot as plt
import joblib
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
print("Test set:", sum(y_test), "/", len(y_test), "records are fraud")
print("Training set:", sum(y_train), "/", len(y_train), "records are fraud")
#print("Hashes of x_train, x_test, y_train, y_test:")
#print(joblib.hash(x_train))

# Apply SMOTE
# Using experiments (we ran this code in a loop for smote_ratio = [0.05, 0.10, 0.15, ..., 0.5])
# we found that the smote ratio = 0.3 (the fraction of the total datset that should be (generated) minority instances)
# gives the best results.
smote_ratio = 0.3
sm = SMOTE(random_state=0, ratio = smote_ratio)
x_train_smote = x_train.astype(float)
y_train_smote = y_train.astype(float)
x_train_res, y_train_res = sm.fit_sample(x_train_smote, y_train_smote)
print("SMOTEd training set:", sum(y_train_res), "/", len(y_train_res), "records are fraud ==> smote_ratio =", smote_ratio)
#print("Hashes of x_train_smote, y_train_smote, x_train_res, y_train_res:")
#print(joblib.hash(x_train_smote))

# Run three classifiers on unSMOTEd and SMOTEd training sets
print("Random Forest performance without smote")
clf = RandomForestClassifier(random_state=0)
clf.fit(x_train, y_train)
y_pred_nosmote_randomforest = clf.predict_proba(x_test)[:,1]
y_pred_nosmote_randomforest_bin = np.around(y_pred_nosmote_randomforest)
print("precision score:", precision_score(y_test, y_pred_nosmote_randomforest_bin))
print("recall score:", recall_score(y_test, y_pred_nosmote_randomforest_bin))
print("accuracy score:", accuracy_score(y_test, y_pred_nosmote_randomforest_bin))
print("f1 score:", f1_score(y_test, y_pred_nosmote_randomforest_bin))

print("Random Forest performance after applying smote")
clf.fit(x_train_res, y_train_res)
y_pred_smote_randomforest = clf.predict_proba(x_test)[:,1]
y_pred_smote_randomforest_bin = np.around(y_pred_smote_randomforest)
print("precision score:", precision_score(y_test, y_pred_smote_randomforest_bin))
print("recall score:", recall_score(y_test, y_pred_smote_randomforest_bin))
print("accuracy score:", accuracy_score(y_test, y_pred_smote_randomforest_bin))
print("f1 score:", f1_score(y_test, y_pred_smote_randomforest_bin))



print("Logistic regression performance without smote")
logreg = LogisticRegression(random_state=0)
logreg.fit(x_train, y_train)
x_test_logreg = x_test.astype(float)
y_pred_nosmote_logisticregression = logreg.predict_proba(x_test_logreg)[:,1]
y_pred_nosmote_logisticregression_bin = np.around(y_pred_nosmote_logisticregression)
print("precision score:", precision_score(y_test, y_pred_nosmote_logisticregression_bin))
print("recall score:", recall_score(y_test, y_pred_nosmote_logisticregression_bin))
print("accuracy score:", accuracy_score(y_test, y_pred_nosmote_logisticregression_bin))
print("f1 score:", f1_score(y_test, y_pred_nosmote_logisticregression_bin))

print("Logistic regression performance after applying smote")
logreg.fit(x_train_res, y_train_res)
y_pred_smote_logisticregression = logreg.predict_proba(x_test_logreg)[:,1]
y_pred_smote_logisticregression_bin = np.around(y_pred_smote_logisticregression)
print("precision score:", precision_score(y_test, y_pred_smote_logisticregression_bin))
print("recall score:", recall_score(y_test, y_pred_smote_logisticregression_bin))
print("accuracy score:", accuracy_score(y_test, y_pred_smote_logisticregression_bin))
print("f1 score:", f1_score(y_test, y_pred_smote_logisticregression_bin))



print("KNN performance without smote")
clf = neighbors.KNeighborsClassifier(algorithm = 'kd_tree')
clf.fit(x_train, y_train)
y_pred_nosmote_knn = clf.predict_proba(x_test)[:,1]
y_pred_nosmote_knn_bin = np.around(y_pred_nosmote_knn)
print("precision score:", precision_score(y_test, y_pred_nosmote_knn_bin))
print("recall score:", recall_score(y_test, y_pred_nosmote_knn_bin))
print("accuracy score:", accuracy_score(y_test, y_pred_nosmote_knn_bin))
print("f1 score:", f1_score(y_test, y_pred_nosmote_knn_bin))

print("KNN performance after applying smote")
clf.fit(x_train_res, y_train_res)
y_pred_smote_knn = clf.predict_proba(x_test)[:,1]
y_pred_smote_knn_bin = np.around(y_pred_smote_knn)
print("precision score:", precision_score(y_test, y_pred_smote_knn_bin))
print("recall score:", recall_score(y_test, y_pred_smote_knn_bin))
print("accuracy score:", accuracy_score(y_test, y_pred_smote_knn_bin))
print("f1 score:", f1_score(y_test, y_pred_smote_knn_bin))



print("ROC curve")
# Compute ROC data for all different classiers
# Random forest without smote
fpr_nosmote_randomforest, tpr_nosmote_randomforest, _ = roc_curve(y_test, y_pred_nosmote_randomforest)
roc_auc_nosmote_randomforest = auc(fpr_nosmote_randomforest, tpr_nosmote_randomforest)
# Random forest with smote
fpr_smote_randomforest, tpr_smote_randomforest, _ = roc_curve(y_test, y_pred_smote_randomforest)
roc_auc_smote_randomforest = auc(fpr_smote_randomforest, tpr_smote_randomforest)

# Logistic regression without smote
fpr_nosmote_logisticregression, tpr_nosmote_logisticregression, _ = roc_curve(y_test, y_pred_nosmote_logisticregression)
roc_auc_nosmote_logisticregression = auc(fpr_nosmote_logisticregression, tpr_nosmote_logisticregression)
# Logistic regression with smote
fpr_smote_logisticregression, tpr_smote_logisticregression, _ = roc_curve(y_test, y_pred_smote_logisticregression)
roc_auc_smote_logisticregression = auc(fpr_smote_logisticregression, tpr_smote_logisticregression)

# KNN without smoke
fpr_nosmote_knn, tpr_nosmote_knn, _ = roc_curve(y_test, y_pred_nosmote_knn)
roc_auc_nosmote_knn = auc(fpr_nosmote_knn, tpr_nosmote_knn)
# KNN with smoke
fpr_smote_knn, tpr_smote_knn, _ = roc_curve(y_test, y_pred_smote_knn)
roc_auc_smote_knn = auc(fpr_smote_knn, tpr_smote_knn)

# SETTING: SAVE IMAGES?
saveToFiles = True
# If the heatmaps should be saved, use current datetime to avoid overwriting existing files
preFileName = datetime.datetime.now().strftime("%H.%M.%S ") + str(smote_ratio)

plt.subplots(figsize=(9, 6))
lw = 2
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--') # straight line through the middle
plt.plot(fpr_nosmote_randomforest, tpr_nosmote_randomforest, color='lightgreen',
 lw=lw, label='Random forest')
plt.plot(fpr_smote_randomforest, tpr_smote_randomforest, color='darkgreen',
 lw=lw, label='Random forest (SMOTEd training set)')
plt.plot(fpr_nosmote_logisticregression, tpr_nosmote_logisticregression, color='magenta',
 lw=lw, label='Logistic regression')
plt.plot(fpr_smote_logisticregression, tpr_smote_logisticregression, color='darkred',
 lw=lw, label='Logistic regression (SMOTEd training set)')
plt.plot(fpr_nosmote_knn, tpr_nosmote_knn, color='lightblue',
 lw=lw, label='k-NN')
plt.plot(fpr_smote_knn, tpr_smote_knn, color='darkblue',
 lw=lw, label='k-NN (SMOTEd training set)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

if saveToFiles:
    plt.savefig(preFileName + " ROC.svg", bbox_inches='tight')


print("PR curve")
plt.subplots(figsize=(9, 6))
precision_nosmote_randomforest, recall_nosmote_randomforest, _ = precision_recall_curve(y_test, y_pred_nosmote_randomforest)
precision_smote_randomforest, recall_smote_randomforest, _ = precision_recall_curve(y_test, y_pred_smote_randomforest)
precision_nosmote_logisticregression, recall_nosmote_logisticregression, _ = precision_recall_curve(y_test, y_pred_nosmote_logisticregression)
precision_smote_logisticregression, recall_smote_logisticregression, _ = precision_recall_curve(y_test, y_pred_smote_logisticregression)
precision_nosmote_knn, recall_nosmote_knn, _ = precision_recall_curve(y_test, y_pred_nosmote_knn)
precision_smote_knn, recall_smote_knn, _ = precision_recall_curve(y_test, y_pred_smote_knn)

plt.step(recall_nosmote_randomforest, precision_nosmote_randomforest, color='lightgreen', where='post', label='Random forest', lw=lw)
plt.step(recall_smote_randomforest, precision_smote_randomforest, color='darkgreen', where='post', label='Random forest (after SMOTE)', lw=lw)
plt.step(recall_nosmote_logisticregression, precision_nosmote_logisticregression, color='magenta', where='post', label='Logistic regression', lw=lw)
plt.step(recall_smote_logisticregression, precision_smote_logisticregression, color='darkred', where='post', label='Logistic regression (after SMOTE)', lw=lw)
plt.step(recall_nosmote_knn, precision_nosmote_knn, color='lightblue', where='post', label='k-NN', lw=lw)
plt.step(recall_smote_knn, precision_smote_knn, color='darkblue', where='post', label='k-NN (after SMOTE)', lw=lw)

plt.plot([0.5, 0.5], color='gray', lw=1, linestyle='--') # straight line through the middle

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(loc="upper right")
plt.title('Precision-Recall')
if saveToFiles:
    plt.savefig(preFileName + " PR.svg", bbox_inches='tight')