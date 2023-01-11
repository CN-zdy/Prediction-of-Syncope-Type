"""
Deyun Zhang, May 2022
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, precision_score, f1_score, recall_score, auc
import warnings
from sklearn.calibration import CalibratedClassifierCV
import shap

shap.initjs() 
warnings.filterwarnings("ignore")


# ---------------------------------------- Required functions of research -------------------------------------------- #

def rm_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print('Create path - %s' % dir_path)

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def ROC(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds = roc_curve(label, y_prob)
    roc_auc = auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, float(optimal_th), optimal_point



# ------------------------------------- Read dataset and Convert type of dataset ------------------------------------- #

#  read train and valid(internal test) data
data = pd.read_excel('./data/2022_10_24_tz_15.xlsx')
#  read external data, if have
ext_test_data = pd.read_excel('./data/ext_test.xlsx')

k = 5
seed = 0
column_names = data.columns
df_missing_rate = data.isnull().mean().sort_values().reset_index()


# -------------------------- The rank of importance features based on five cross-validation -------------------------- #

all_features = np.array(data[column_names[:-1]], dtype=float)
all_labels = np.array(data[column_names[-1]], dtype=int)

kf = KFold(n_splits=k, shuffle=True, random_state=0)
feature_scores = []
for train_index, test_index in tqdm(kf.split(all_features), desc='Ranking...'):
    X_train, X_test = all_features[train_index], all_features[test_index]
    y_train, y_test = all_labels[train_index], all_labels[test_index]

    rf = RFC(n_estimators=100, random_state=seed)
    rf.fit(X_train, y_train)

    feature_scores.append(rf.feature_importances_)

importance = sum(feature_scores) / k
imp_result = np.argsort(importance)[::-1][:]
feat_labels = [column_names[i] for i in imp_result]

plt.figure(dpi=300)
plt.bar(range(len(imp_result)), importance[imp_result], color='lightblue', align='center')
plt.xticks(range(len(imp_result)), feat_labels, rotation=90)
plt.yticks(rotation=90)
plt.xlim([-1, len(imp_result)])
plt.ylabel('Feature Importance', labelpad=10, size=12)
plt.tight_layout()
f = plt.gcf()
f.savefig('./Rank_of_Features.jpg')
f.clear()

df_imp = pd.DataFrame({'col': column_names[:-1], 'score': importance})
df_imp = df_imp.sort_values(by='score', ascending=False)
df_imp.to_csv('./results/importance_scores.csv', index=False)


# -------------------------- Training different models based on five cross-validation -------------------------- #

ext_test_frame = ext_test_data[column_names[:-1]]
ext_test_label = np.array(ext_test_data[column_names[-1]], dtype=int)
x_train_frame = pd.DataFrame(all_features, columns=column_names[:-1])

model_name_list = ["LR", "PPN", "SVM", "DTC", "RFC", "KN"]

# select model
name = model_name_list[4]

df_imp = pd.read_csv('./results/importance_scores.csv')
max_n_features = 15
n_features = list(range(1, max_n_features + 1))

all_roc_list = []
test_all_roc_list = []
ext_test_all_roc_list = []

for topK in tqdm(n_features, desc='Evaluating...'):

    x_cols = df_imp[:topK].col.values
    X = x_train_frame[x_cols].values
    ext_test_value = ext_test_frame[x_cols].values

    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    roc_list = []
    pre_list = []
    rec_list = []
    f1s_list = []

    ext_test_roc_list = []
    ext_test_pre_list = []
    ext_test_rec_list = []
    ext_test_f1s_list = []

    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = all_labels[train_index], all_labels[test_index]

        if name == 'LR':
            lr = LogisticRegression()
        elif name == 'PPN':
            lr1 = Perceptron()
            lr = CalibratedClassifierCV(lr1, method='isotonic')
        elif name == 'SVM':
            lr = SVC(probability=True)
        elif name == "DTC":
            lr = DecisionTreeClassifier(criterion='entropy', splitter="random", max_depth=5, random_state=seed)
        elif name == "RFC":
            lr = RFC(n_estimators=108, random_state=0, criterion='entropy', max_depth=5, n_jobs=2, max_leaf_nodes=9, max_samples=272)
        else:
            lr = KNeighborsClassifier(n_neighbors=10, weights='distance') 

        lr.fit(x_train, y_train)

        y_predict = lr.predict(x_test)
        y_predict_prob = lr.predict_proba(x_test)
        fpr, tpr, roc_auc_, optimal_th, optimal_point = ROC(y_test, y_predict_prob[:, 1])

        precision = precision_score(y_test, y_predict, average='weighted')
        recall = recall_score(y_test, y_predict, average='weighted')
        f1 = f1_score(y_test, y_predict, average='weighted')
        
        ext_test_y_predict = lr.predict(np.array(ext_test_value))
        ext_test_y_predict_prob = lr.predict_proba(np.array(ext_test_value))
        ext_test_fpr, ext_test_tpr, ext_test_roc_auc_, optimal_th, optimal_point = ROC(ext_test_label, ext_test_y_predict_prob[:, 1])

        ext_test_precision = precision_score(ext_test_label, ext_test_y_predict, average='weighted')
        ext_test_recall = recall_score(ext_test_label, ext_test_y_predict, average='weighted')
        ext_test_f1 = f1_score(ext_test_label, ext_test_y_predict, average='weighted')

        roc_list.append(roc_auc_)
        pre_list.append(precision)
        rec_list.append(recall)
        f1s_list.append(f1)

        ext_test_roc_list.append(ext_test_roc_auc_)
        ext_test_pre_list.append(ext_test_precision)
        ext_test_rec_list.append(ext_test_recall)
        ext_test_f1s_list.append(ext_test_f1)

    with open(f'./result_test_{name}_{topK}.csv', 'w') as file:
        file.write('roc, precision, recall, F1\n')
        for n, p, c, z in zip(roc_list, pre_list, rec_list, f1s_list):
            file.write("{},{},{},{}\n".format(n, p, c, z))

    with open(f'./result_ext_test_{name}_{topK}.csv', 'w') as file:
        file.write('roc, precision, recall, F1\n')
        for n, p, c, z in zip(ext_test_roc_list, ext_test_pre_list, ext_test_rec_list, ext_test_f1s_list):
            file.write("{},{},{},{}\n".format(n, p, c, z))
    
    
