import os
import pandas as pd
import numpy as np
from scipy import interp
import joblib
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import  as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as LR
from collections import Counter, OrderedDict
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, precision_score, f1_score, recall_score, auc, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, label_binarize
import warnings
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

# ---------------------------------------- Required functions of research -------------------------------------------- #

def rm_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print('Create path - %s' % dir_path)

def label_one_hot(label, n_classes):

    out_label = []
    for l in label:
        tmp_label = np.zeros(n_classes)
        tmp_label[l] = 1
        out_label.append(tmp_label)
    return np.array(out_label)

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def ROC(label, y_prob):
    fpr, tpr, thresholds = roc_curve(label, y_prob)
    roc_auc = auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, float(optimal_th), optimal_point

# ----------------------------------------------- Create save path --------------------------------------------------- #

model_name_list = ["LR", "PPN", "SVM", "DTC", "RFC", "KN", "MLP"]
name = model_name_list[4]
rm_mkdir('./results/rank/')
rm_mkdir('./results/data_columns/')
rm_mkdir('./results/models/')
rm_mkdir('./results/ROC/')
rm_mkdir('./results/result/')
rm_mkdir(f'./results/res_flod/{name}/')
rm_mkdir(f'./results/fpr/{name}/')
rm_mkdir(f'./results/tpr/{name}/')

# ------------------------------------- Read dataset and Convert type of dataset ------------------------------------- #

data = pd.read_excel('./Features.xlsx')
print('Read data shape:', data.shape)

column_names = data.columns
features, x_test_o, labels, y_test_o = train_test_split(np.array(data[column_names[:-1]], dtype=float),
                                              np.array(data[column_names[-1]], dtype=int),
                                              test_size=1/6,
                                              random_state=0)

features_frame = pd.DataFrame(features, columns=column_names[:-1])
test_frame = pd.DataFrame(x_test_o, columns=column_names[:-1])

k = 5
df_missing_rate = data.isnull().mean().sort_values().reset_index()

# -------------------------- The rank of importance features based on five cross-validation -------------------------- #

all_features = np.array(data[column_names[:-1]], dtype=float)
all_labels = np.array(data[column_names[-1]], dtype=int)

kf = KFold(n_splits=k, shuffle=True, random_state=0)
feature_scores = []
for train_index, test_index in tqdm(kf.split(all_features), desc='Ranking...'):
    X_train, X_test = all_features[train_index], all_features[test_index]
    y_train, y_test = all_labels[train_index], all_labels[test_index]

    rf = RFC(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)

    feature_scores.append(rf.feature_importances_)

importance = np.mean(np.array(feature_scores), axis=0)
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
f.savefig(r'./results/rank/Rank_of_Features.jpg')
f.clear()

df_imp = pd.DataFrame({'col': column_names[:-1], 'score': importance})
df_imp = df_imp.sort_values(by='score', ascending=False)
df_imp.to_csv('./results/rank/importance_scores.csv', index=False)

# -------------------------- Training different models based on five cross-validation -------------------------- #

df_imp = pd.read_csv('./results/rank/importance_scores.csv')
max_n_features = 19
n_features = list(range(1, max_n_features + 1))
all_res = []

mra = []
mpr = []
mre = []
mf1 = []

for topK in tqdm(n_features, desc='Evaluating...'):
    roc_auc = 0.
    pre = 0.
    rec = 0.
    f1s = 0.
    tmp_res = []
    roc_list = []
    pre_list = []
    rec_list = []
    f1s_list = []

    x_cols = df_imp[:topK].col.values
    X = features_frame[x_cols].values
    joblib.dump(x_cols, './results/data_columns/data_columns_%s.pkl' % (topK))
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    flod = 0
    tpr_list = []
    fpr_list = []
    index = []
    mean_fpr = np.linspace(0, 1, 68)

    for train_index, test_index in kf.split(X):
        flod +=1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        if name == 'LR':
            lr = LogisticRegression()
        elif name == 'PPN':
            lr = Perceptron()
            lr = CalibratedClassifierCV(lr, method='isotonic')
        elif name == 'SVM':
            lr = SVC(probability=True)
        elif name == "DTC":
            lr = DecisionTreeClassifier(criterion='entropy', splitter="random", max_depth=3, random_state=0)
        elif name == "RFC":
            lr = RFC(n_estimators=108, random_state=0, criterion='entropy', max_depth=5, n_jobs=2, max_leaf_nodes=5, max_samples=226)
        elif name == "KN":
            lr = KNeighborsClassifier(n_neighbors=3, weights='distance')
        else:
            lr = MLPClassifier()
        lr.fit(X_train, y_train)

        y_predict = lr.predict(X_test)
        y_predict_prob = lr.predict_proba(X_test)

        fpr, tpr, roc_auc_, optimal_th, optimal_point = ROC(y_test, y_predict_prob[:, 1])
        tpr_list.append(interp(mean_fpr, fpr, tpr))

        precision = precision_score(y_test, y_predict)
        recall = recall_score(y_test, y_predict)
        f1 = f1_score(y_test, y_predict)

        if topK <= 11 or topK == 19:
            joblib.dump(lr, f'./results/models/model_{name}_{flod}_{topK}.pkl')

        roc_auc += roc_auc_
        pre += precision
        rec += recall
        f1s += f1

        roc_list.append(roc_auc_)
        pre_list.append(precision)
        rec_list.append(recall)
        f1s_list.append(f1)

        if topK <= 6 or topK == 19:
            nproc = np.array(roc_list)
            np.savetxt('./results/res_flod/' + name + '/roc_auc_%s_' % (topK) + name + '.csv', nproc)
            nppre = np.array(pre_list)
            np.savetxt('./results/res_flod/' + name + '/pre_%s_' % (topK) + name + '.csv', nppre)
            nprec = np.array(rec_list)
            np.savetxt('./results/res_flod/' + name + '/rec_%s_' % (topK) + name + '.csv', nprec)
            npf1s = np.array(f1s_list)
            np.savetxt('./results/res_flod/' + name + '/f1s_%s_' % (topK) + name + '.csv', npf1s)

    mean_tpr = np.mean(tpr_list, axis=0)
    mean_tpr[-1] = 1.0

    if topK <= 6 or topK == 19:
        np.save('./results/fpr/' + name + '/mean_fpr_%s_' % (topK) + name + '.npy', mean_fpr)
        np.save('./results/tpr/' + name + '/mean_tpr_%s_' % (topK) + name + '.npy', mean_tpr)

    mean_auc = auc(mean_fpr, mean_tpr)

    if topK <= 6 or topK == 19:
        plt.figure(1)
        plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean AUC=%0.3f' % mean_auc, lw=2, alpha=.8)
        plt.plot([0, 1], [0, 1], linestyle='--',color='r')
        plt.xlim([0., 1.])
        plt.ylim([0., 1.])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc='lower right')
        plt.savefig('./results/ROC/roc_featur_%s_' %(topK) + name + '.jpg')
        plt.close()

    var_roc = np.var(roc_list)
    var_pre = np.var(pre_list)
    var_rec = np.var(rec_list)
    var_f1s = np.var(f1s_list)

    mean_roc_auc = mean_auc
    mean_pre = pre / k
    mean_rec = rec / k
    mean_f1s = f1s / k

    mra.append(mean_roc_auc)
    mpr.append(mean_pre)
    mre.append(mean_rec)
    mf1.append(mean_f1s)

result = np.zeros((4, 19))
result[0, :] += mra
result[1, :] += mpr
result[2, :] += mre
result[3, :] += mf1

data = pd.DataFrame(result, index=['ROCAUC', 'Precision', 'Recall', 'F1-score'],
                    columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6' , 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19'])
data.to_csv('./results/result/result_' + name + '.csv')
