import pandas as pd
import numpy as np
import os
from scipy import interp
import joblib
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_score, f1_score, recall_score, auc
import warnings
from sklearn.calibration import CalibratedClassifierCV

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

    fpr, tpr, thresholds = roc_curve(label, y_prob)
    roc_auc = auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, float(optimal_th), optimal_point

# ----------------------------------------------- Create save path --------------------------------------------------- #

path_save_rank = './results/rank/'
rm_mkdir(path_save_rank)

path_save_models = './results/models/'
rm_mkdir(path_save_models)
path_save_models_DT = './results/models/DT/'
rm_mkdir(path_save_models_DT)
path_save_models_KNN = './results/models/KNN/'
rm_mkdir(path_save_models_KNN)
path_save_models_LR = './results/models/LR/'
rm_mkdir(path_save_models_LR)
path_save_models_PPN = './results/models/PPN/'
rm_mkdir(path_save_models_PPN)
path_save_models_RF = './results/models/RF/'
rm_mkdir(path_save_models_RF)
path_save_models_SVM = './results/models/SVM/'
rm_mkdir(path_save_models_SVM)

path_save_mertics = './results/mertics/'
rm_mkdir(path_save_mertics)
path_save_mertics_DT = './results/mertics/DT/'
rm_mkdir(path_save_mertics_DT)
path_save_mertics_KNN = './results/mertics/KNN/'
rm_mkdir(path_save_mertics_KNN)
path_save_mertics_LR = './results/mertics/LR/'
rm_mkdir(path_save_mertics_LR)
path_save_mertics_PPN = './results/mertics/PPN/'
rm_mkdir(path_save_mertics_PPN)
path_save_mertics_RF = './results/mertics/RF/'
rm_mkdir(path_save_mertics_RF)
path_save_mertics_SVM = './results/mertics/SVM/'
rm_mkdir(path_save_mertics_SVM)

path_save_tpr = './results/ROC/tpr/'
rm_mkdir(path_save_tpr)
path_save_tpr_DT = './results/ROC/tpr/DT/'
rm_mkdir(path_save_tpr_DT)
path_save_tpr_KNN = './results/ROC/tpr/KNN/'
rm_mkdir(path_save_tpr_KNN)
path_save_tpr_LR = './results/ROC/tpr/LR/'
rm_mkdir(path_save_tpr_LR)
path_save_tpr_PPN = './results/ROC/tpr/PPN/'
rm_mkdir(path_save_tpr_PPN)
path_save_tpr_RF = './results/ROC/tpr/RF/'
rm_mkdir(path_save_tpr_RF)
path_save_tpr_SVM = './results/ROC/tpr/SVM/'
rm_mkdir(path_save_tpr_SVM)

path_save_fpr = './results/ROC/fpr/'
rm_mkdir(path_save_fpr)
path_save_fpr_DT = './results/ROC/fpr/DT/'
rm_mkdir(path_save_fpr_DT)
path_save_fpr_KNN = './results/ROC/fpr/KNN/'
rm_mkdir(path_save_fpr_KNN)
path_save_fpr_LR = './results/ROC/fpr/LR/'
rm_mkdir(path_save_fpr_LR)
path_save_fpr_PPN = './results/ROC/fpr/PPN/'
rm_mkdir(path_save_fpr_PPN)
path_save_fpr_RF = './results/ROC/fpr/RF/'
rm_mkdir(path_save_fpr_RF)
path_save_fpr_SVM = './results/ROC/fpr/SVM/'
rm_mkdir(path_save_fpr_SVM)

path_save_ROC = './results/ROC/'
rm_mkdir(path_save_ROC)

path_save_res = './results/'

# ------------------------------------- Read dataset and Convert type of dataset ------------------------------------- #

data = pd.read_csv('./data_syncope.csv')
print('Read data shape:', data.shape)
column_names = data.columns

features = np.array(data[column_names[:-1]], dtype=float)
labels = np.array(data[column_names[-1]], dtype=int)

k = 5
df_missing_rate = data.isnull().mean().sort_values().reset_index()

# -------------------------- The rank of importance features based on five cross-validation -------------------------- #

kf = KFold(n_splits=k, shuffle=True, random_state=0)
feature_scores = []
for train_index, test_index in tqdm(kf.split(features), desc='Ranking...'):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    rf = RandomForestClassifier(n_estimators=1000, random_state=0)
    rf.fit(X_train, y_train)

    feature_scores.append(rf.feature_importances_)

importance = np.mean(np.array(feature_scores), axis=0)
imp_result = np.argsort(importance)[::-1][:]

for i in range(len(imp_result)):
     print("%2d. %-*s %f" % (i + 1, 30, column_names[imp_result[i]], importance[imp_result[i]]))

feat_labels = [column_names[i] for i in imp_result]

plt.title('Feature Importance')
plt.bar(range(len(imp_result)), importance[imp_result], color='lightblue', align='center')
plt.xticks(range(len(imp_result)), feat_labels, rotation=90)
plt.xlim([-1, len(imp_result)])
plt.tight_layout()
f = plt.gcf()
f.savefig(path_save_rank + 'Rank of Features.jpg')
f.clear()

df_imp = pd.DataFrame({'col': column_names[:-1], 'score': importance})
df_imp = df_imp.sort_values(by='score', ascending=False)
df_imp.to_csv(path_save_rank + 'importance_scores.csv', index=False)

# -------------------------- The rank of importance features based on five cross-validation -------------------------- #

df_imp = pd.read_csv(path_save_rank + 'importance_scores.csv')
max_n_features = 38
n_features = list(range(1, max_n_features + 1))
all_res = []

mra = []
mpr = []
mre = []
mf1 = []

for topK in tqdm(n_features, desc='Evaluating'):
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
    X = data[x_cols].values
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

        # --------------------------- LogisticRegression ----------------------------- #
        # lr = LogisticRegression(class_weight='balanced')   # 初始化
        # name = "LR"
        # ------------------------------- Perceptron --------------------------------- #
        # lr = Perceptron()
        # lr = CalibratedClassifierCV(lr, method='isotonic')
        # name = "PPN"
        # -------------------------- Support vector machine -------------------------- #
        # lr = SVC(probability=True) ##'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
        # name = "SVM"
        # -------------------------- DecisionTreeClassifier -------------------------- #
        # lr = DecisionTreeClassifier(criterion='entropy', splitter="random", max_depth=3, random_state=0)
        # name = "DT"
        # -------------------------- RandomForestClassifier -------------------------- #
        lr = RandomForestClassifier(criterion='entropy', max_depth=3, random_state=0, n_jobs=2)
        name = "RF"
        # --------------------------- KNeighborsClassifier --------------------------- #
        # lr = KNeighborsClassifier(n_neighbors=3, weights='distance') # 'auto', 'ball_tree', 'kd_tree', 'brute'
        # name = "KNN"
        lr.fit(X_train, y_train)

        y_predict = lr.predict(X_test)
        y_predict_prob = lr.predict_proba(X_test)

        fpr, tpr, roc_auc_, optimal_th, optimal_point = ROC(y_test, y_predict_prob[:, 1])
        tpr_list.append(interp(mean_fpr, fpr, tpr))

        precision = precision_score(y_test, y_predict)
        recall = recall_score(y_test, y_predict)
        f1 = f1_score(y_test, y_predict)

        if topK <= 6:
            joblib.dump(lr, path_save_models + name + '/model_%s_%s_%s_%s_%s_%s_%s.pth'
                        % (name, flod, roc_auc_, precision, recall, f1, topK))
        elif topK == 38:
            joblib.dump(lr, path_save_models + name + '/model_%s_%s_%s_%s_%s_%s_%s.pth'
                        % (name, flod, roc_auc_, precision, recall, f1, topK))

        roc_auc += roc_auc_
        pre += precision
        rec += recall
        f1s += f1

        roc_list.append(roc_auc_)
        pre_list.append(precision)
        rec_list.append(recall)
        f1s_list.append(f1)

        if topK <= 6:
            nproc = np.array(roc_list)
            np.savetxt(path_save_mertics + name + '/roc_auc_%s_' % (topK) + name + '.csv', nproc)
            nppre = np.array(pre_list)
            np.savetxt(path_save_mertics + name + '/pre_%s_' % (topK) + name + '.csv', nppre)
            nprec = np.array(rec_list)
            np.savetxt(path_save_mertics + name + '/rec_%s_' % (topK) + name + '.csv', nprec)
            npf1s = np.array(f1s_list)
            np.savetxt(path_save_mertics + name + '/f1s_%s_' % (topK) + name + '.csv', npf1s)
        elif topK == 38:
            nproc = np.array(roc_list)
            np.savetxt(path_save_mertics + name + '/roc_auc_%s_' % (topK) + name + '.csv', nproc)
            nppre = np.array(pre_list)
            np.savetxt(path_save_mertics + name + '/pre_%s_' % (topK) + name + '.csv', nppre)
            nprec = np.array(rec_list)
            np.savetxt(path_save_mertics + name + '/rec_%s_' % (topK) + name + '.csv', nprec)
            npf1s = np.array(f1s_list)
            np.savetxt(path_save_mertics + name + '/f1s_%s_' % (topK) + name + '.csv', npf1s)

    mean_tpr = np.mean(tpr_list, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    if topK <= 6:
        np.save(path_save_fpr + name + '/mean_fpr_%s_' % (topK) + name + '.npy', mean_fpr)
        np.save(path_save_tpr + name + '/mean_tpr_%s_' % (topK) + name + '.npy', mean_tpr)
    elif topK ==38:
        np.save(path_save_fpr + name + '/mean_fpr_%s_' % (topK) + name + '.npy', mean_fpr)
        np.save(path_save_tpr + name + '/mean_tpr_%s_' % (topK) + name + '.npy', mean_tpr)

    if topK <= 6:
        plt.figure(1)
        plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean AUC=%0.3f' % mean_auc, lw=2, alpha=.8)
        plt.plot([0, 1], [0, 1], linestyle='--',color='r')
        plt.xlim([0., 1.])
        plt.ylim([0., 1.])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc='lower right')
        plt.savefig(path_save_ROC + 'roc_featur_%s_' %(topK) + name + '.jpg')
        plt.close()
    elif topK == 38:
        plt.figure(1)
        plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean AUC=%0.3f' % mean_auc, lw=2, alpha=.8)
        plt.plot([0, 1], [0, 1], linestyle='--',color='r')
        plt.xlim([0., 1.])
        plt.ylim([0., 1.])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc='lower right')
        plt.savefig(path_save_ROC + 'roc_featur_%s_' %(topK) + name + '.jpg')
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


result = np.zeros((4, 38))
result[0, :] += mra
result[1, :] += mpr
result[2, :] += mre
result[3, :] += mf1
data = pd.DataFrame(result, index=['ROCAUC', 'Precision', 'Recall', 'F1-score'],
                        columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6',
                                 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_12',
                                 'feature_13', 'feature_14', 'feature_15', 'feature_16', 'feature_17', 'feature_18',
                                 'feature_19', 'feature_20', 'feature_21', 'feature_22', 'feature_23', 'feature_24',
                                 'feature_25', 'feature_26', 'feature_27', 'feature_28', 'feature_29', 'feature_30',
                                 'feature_31', 'feature_32', 'feature_33', 'feature_34', 'feature_35', 'feature_36',
                                 'feature_37', 'feature_38'])
data.to_csv(path_save_res + 'result_' + name + '.csv')