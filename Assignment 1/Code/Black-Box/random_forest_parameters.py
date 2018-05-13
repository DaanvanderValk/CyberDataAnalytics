# Interesting part of script starts at line 116

get_ipython().magic(u'matplotlib inline')
import datetime
import time
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import collections

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

## SCRIPT TO OPTIMIZE PARAMETERS FOR THE RANDOM FOREST

# Write output to file.
f = open('random_forest_parameter_optimization.txt', 'w')
x_train = {}
x_test = {}
y_train = {}
y_test = {}

x_train_smoted = collections.defaultdict(dict)
y_train_smoted = collections.defaultdict(dict)

smote_ratios = [0.3, 0.35, 0.4]


# Settings: you take more measurements if you want
random_data_sample = 1      # number of test-train data splits to consider
random_forest_sample = 1    # number of random forests to consider (for each iteration)

# First, the data is processed (splitted and SMOTEd)
for random_data_state in range(random_data_sample):
    # Split data into training and testing records
    x_train[random_data_state], x_test[random_data_state], y_train[random_data_state], y_test[random_data_state] = train_test_split(x_array, y_array, test_size = 0.1, random_state=random_data_state)
    print("DATA RANDOM STATE:", random_data_state, ". Test set:", sum(y_test[random_data_state]), "/", len(y_test[random_data_state]), "records are fraud. Training set:", sum(y_train[random_data_state]), "/", len(y_train[random_data_state]), "records are fraud")
    f.write("DATA RANDOM STATE: " + str(random_data_state) + ". Test set: " + str(sum(y_test[random_data_state])) + "/" + str(len(y_test[random_data_state])) + " records are fraud. Training set: " + str(sum(y_train[random_data_state])) + "/" + str(len(y_train[random_data_state])) + " records are fraud\n")

    # Generate SMOTEd data
    for smote_ratio in smote_ratios:
        sm = SMOTE(random_state=0, ratio = smote_ratio)
        x_train_smoted[random_data_state][smote_ratio], y_train_smoted[random_data_state][smote_ratio] = sm.fit_sample(x_train[random_data_state].astype(float), y_train[random_data_state].astype(float))
        print("SMOTEd training set:", sum(y_train_smoted[random_data_state][smote_ratio]), "/", len(y_train_smoted[random_data_state][smote_ratio]), "records are fraud ==> smote_ratio =", smote_ratio)
        f.write("SMOTEd training set: " + str(sum(y_train_smoted[random_data_state][smote_ratio])) + "/" + str(len(y_train_smoted[random_data_state][smote_ratio])) + " records are fraud ==> smote_ratio = " + str(smote_ratio) + "\n")
            
        
print("Data generated")

# Try different smote ratios
for smote_ratio in smote_ratios:
    # Try a different number of trees in the forest (n_estimators)
    for number_of_trees in range(11,25):
        # Try the two criterion techniques (entropy or gini)
        for criterion in ['entropy', 'gini']:
            
            precision, recall, accuracy, f1 = 0.0, 0.0, 0.0, 0.0
            print("Results for a random forest with smote_ratio =", smote_ratio, "-- n_estimators =", number_of_trees, "-- criterion:", criterion)
            f.write("Results for a random forest with smote_ratio = " + str(smote_ratio) + " -- n_estimators = " + str(number_of_trees) + "-- criterion:" + str(criterion) + "\n")
            
            # Loop over all datasets
            for random_data_state in range(random_data_sample):
                # Generate random forest(s)
                for random_state_trees in range(random_forest_sample):
                    # Generate random forest and measure results
                    clf = RandomForestClassifier(random_state=random_state_trees, n_estimators=number_of_trees, max_depth=None, bootstrap=True, min_samples_leaf=1, criterion=criterion)
                    clf.fit(x_train_smoted[random_data_state][smote_ratio], y_train_smoted[random_data_state][smote_ratio])
                    y_pred = clf.predict(x_test[random_data_state])
                    precision += precision_score(y_test[random_data_state], y_pred)
                    recall += recall_score(y_test[random_data_state], y_pred)
                    accuracy += accuracy_score(y_test[random_data_state], y_pred)
                    f1 += f1_score(y_test[random_data_state], y_pred)
                        
            print("Average ==> precision:", np.around(precision/(random_data_sample * random_forest_sample), 5), "-- recall:", np.around(recall/(random_data_sample * random_forest_sample), 5), "-- accuracy:", np.around(accuracy/(random_data_sample * random_forest_sample), 5), "-- f1 score:", np.around(f1/(random_data_sample * random_forest_sample), 5))
            f.write("Average ==> precision: " + str(np.around(precision/(random_data_sample * random_forest_sample), 5)) + " -- recall:" + str(np.around(recall/(random_data_sample * random_forest_sample), 5)) + " -- accuracy: " + str(np.around(accuracy/(random_data_sample * random_forest_sample), 5)) + " -- f1 score: " + str(np.around(f1/(random_data_sample * random_forest_sample), 5)) + "\n")

f.close()

# Final result: optimal parameters are:
#  - SMOTE ratio: 0.4
#  - Number of trees: 21
#  - Criterion: entropy