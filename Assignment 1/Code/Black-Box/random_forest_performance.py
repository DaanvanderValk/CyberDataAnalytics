# Interesting part of script starts at line 114

get_ipython().magic(u'matplotlib inline')
import datetime
import time
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

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

## Testing Performance of the black-box Random Forest classifier

## As we want to SMOTE our data, we cannot simply use cross_val_score().
## However, scikit-learn offers a tool to split the data in 10 folds:
##  after making this split, we SMOTE the training part of the data
##  and calculate the performance on that particular fold
## Finally, we aggregate and present the results.

# Split the data in 10 folds
kf = KFold(n_splits=10, random_state=0, shuffle=True)
splitted = kf.split(x_array)

# Prepare the measurements
precision, recall, f1 = 0.0, 0.0, 0.0
TP, FP, FN, TN = 0, 0, 0, 0

# Run through all 10 combinations: 9/10 of the data is used for training, 1/10 for testing
for train_index, test_index in splitted:
    x_train, x_test = x_array[train_index], x_array[test_index]
    y_train, y_test = y_array[train_index], y_array[test_index]
    
    # Only SMOTE the training data (!)
    sm = SMOTE(random_state=0, ratio = 0.4)
    x_smoted, y_smoted = sm.fit_sample(x_train.astype(float), y_train.astype(float))
    
    # Train classifier and predict the cases
    clf = RandomForestClassifier(random_state=0, n_estimators=21, criterion='entropy')
    clf.fit(x_smoted, y_smoted)
    y_pred = clf.predict(x_test)
    
    # Compute performance
    precision += precision_score(y_test, y_pred)
    recall += recall_score(y_test, y_pred)
    f1 += f1_score(y_test, y_pred)
    
    for i in range(len(y_pred)):
        if y_test[i]==1 and y_pred[i]==1:
            TP += 1
        if y_test[i]==0 and y_pred[i]==1:
            FP += 1
        if y_test[i]==1 and y_pred[i]==0:
            FN += 1
        if y_test[i]==0 and y_pred[i]==0:
            TN += 1

# Finally, present results
print("Precision:", np.around(precision/10, 4))
print("Recall:", np.around(precision/10, 4))
print("F1 score:", np.around(precision/10, 4))

print('True Positives:', str(TP), "(good: found fraud cases)")
print('False Positives:', str(FP), "(bad: benign transactions marked as fraud)")
print('False Negatives:', str(FN), "(bad: missed fraud cases)")
print('True Negatives:', str(TN), "(good: correctly classified benign transactions)")