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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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

'''RandomForestClassifier(
            n_estimators=10,                    # Interesting: 10 seems 
            criterion=’gini’,
            max_depth=None,                     # Not interesting; no max by default. If there is a max (5, 10, 15) all scores get worse.
            min_samples_split=2,                # Not interesting: 2 is a good choice?
            min_samples_leaf=1,                 # Not interesting: 1 provides best results?
            min_weight_fraction_leaf=0.0,
            max_features=’auto’,                # Not interesting: 'auto'='sqrt' and 'log2' have the same results, None gets worse
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,          # 0.0 is the best, increasing value ==> worse results
            min_impurity_split=None,            # Deprecated: use min_impurity_decrease 
            bootstrap=True,                     # In all cases, bootstrap=True yield better results
            oob_score=False,
            n_jobs=1,
            random_state=None,                  # Put in loop
            verbose=0,                          # Not interesting
            warm_start=False,                   # Should be false
            class_weight=None                   # Already covered by SMOTE
'''

print(x_array[0])

f = open('random_forest_results_nothing_removed.txt', 'w')


if True:
    x_array_smaller = x_array

    for random_state_data in range(5):
        # Split data into training and testing records
        x_train, x_test, y_train, y_test = train_test_split(x_array_smaller, y_array, test_size = 0.2, random_state=random_state_data)
        print("DATA RANDOM STATE:", random_state_data, ". Test set:", sum(y_test), "/", len(y_test), "records are fraud. Training set:", sum(y_train), "/", len(y_train), "records are fraud")
        f.write("DATA RANDOM STATE: " + str(random_state_data) + ". Test set: " + str(sum(y_test)) + "/" + str(len(y_test)) + " records are fraud. Training set: " + str(sum(y_train)) + "/" + str(len(y_train)) + " records are fraud\n")
        
        for x in range(10, 15):
            # Apply SMOTE
            # Using experiments (we ran this code in a loop for smote_ratio = [0.05, 0.10, 0.15, ..., 0.5])
            # we found that the smote ratio = 0.3 (the fraction of the total datset that should be (generated) minority instances)
            # gives the best results.
            smote_ratio = x/40
            sm = SMOTE(random_state=random_state_data, ratio = smote_ratio)
            x_train_smoted, y_train_smoted = sm.fit_sample(x_train.astype(float), y_train.astype(float))
            print("SMOTEd training set:", sum(y_train_smoted), "/", len(y_train_smoted), "records are fraud ==> smote_ratio =", smote_ratio)
            f.write("SMOTEd training set: " + str(sum(y_train_smoted)) + "/" + str(len(y_train_smoted)) + " records are fraud ==> smote_ratio = " + str(smote_ratio) + "\n")
            #print("Hashes of x_train_smote, y_train_smote, x_train_res, y_train_res:")
            #print(joblib.hash(x_train_smote))
            
            for number_of_trees in range(8,13):
                if True:
                    
                    precision, recall, accuracy, f1 = 0.0, 0.0, 0.0, 0.0
                    print("Results for a random forest with smote_ratio =", smote_ratio, "-- n_estimators =", number_of_trees)
                    f.write("Results for a random forest with smote_ratio = " + str(smote_ratio) + " -- n_estimators = " + str(number_of_trees) + "\n")
                    for random_state_trees in range(5):
                        clf = RandomForestClassifier(random_state=random_state_trees, n_estimators=number_of_trees, max_depth=None, bootstrap=True, min_samples_leaf=1)
                        clf.fit(x_train_smoted, y_train_smoted)
                        y_pred = clf.predict(x_test)
                        precision += precision_score(y_test, y_pred)
                        recall += recall_score(y_test, y_pred)
                        accuracy += accuracy_score(y_test, y_pred)
                        f1 += f1_score(y_test, y_pred)
                    
                    print("Average ==> precision:", np.around(precision/5, 4), "-- recall:", np.around(recall/5, 4), "-- accuracy:", np.around(accuracy/5, 4), "-- f1 score:", np.around(f1/5, 4))
                    f.write("Average ==> precision: " + str(np.around(precision/5, 4)) + " -- recall:" + str(np.around(recall/5, 4)) + " -- accuracy: " + str(np.around(accuracy/5, 4)) + " -- f1 score: " + str(np.around(f1/5, 4)) + "\n")


f.close()