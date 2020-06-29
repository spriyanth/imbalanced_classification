############### PACKAGES ###############

### CORE PACKAGES

import numpy as np 
import pandas as pd 

### SKLEARN

from sklearn import metrics, preprocessing
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, RepeatedStratifiedKFold

### MACHINE LEARNING METHODS

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier 

### IMBALANCED LEARN

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import EasyEnsembleClassifier

from imblearn.pipeline import Pipeline

### ADDITIONAL

import os 
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

############### DIRECTORY ###############

data_dir = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin/data'

############### LOAD DATASET ###############

df = pd.read_csv(os.path.join(data_dir, 'breast_cancer_winsor_trafo.csv'), sep = ',') 

seed1 = 123

############### UNDERSAMPLE THE BREAST CANCER DATASET ###############

df_1 = df.loc[df['diagnosis'] == 1] # n=212
df_0 = df.loc[df['diagnosis'] == 0] # n=357

np.random.seed(seed1)

remove_n = 170 # we have to delete 170 pos. obs to achieve the 1:10 ratio.
drop_indices = np.random.choice(df_1.index, remove_n, replace=False)
df_1_subset = df_1.drop(drop_indices)

df_undersampled = pd.concat([df_1_subset, df_0], axis = 0) # merge the dataset back together => n_total = 429
df_undersampled = df_undersampled.sample(frac=1).reset_index(drop=True) # shuffle the dataset

X = df_undersampled.drop(['diagnosis'], axis = 1)
Y = df_undersampled.diagnosis

############### SPLIT THE DATA ###############

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = seed1, stratify = Y)

# argument 'stratify' keeps the class distributions equal across training and test data 
# split dataset into training and test data

classprop_train = sum(Y_train)/len(Y_train)
classprop_test = sum(Y_test)/len(Y_test) # check: alpha = 0.1 for train and test.

featurenames = X_train.columns

############### PERFORM STANDARDIZATION ###############

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std  = scaler.transform(X_test)

X_train = pd.DataFrame(data = X_train_std, columns = featurenames)
X_test = pd.DataFrame(data = X_test_std, columns = featurenames)

Y_train = pd.Series.to_numpy(Y_train)

############### DEFINE HYBRID CLASSIFIERS ###############

DT = DecisionTreeClassifier(criterion = 'gini', splitter = "best", min_samples_split = 2, max_depth = None, max_features = None, random_state = 1)
SVM = SVC(kernel = 'rbf', probability = True, random_state = seed1)
ADA = AdaBoostClassifier(n_estimators = 25, random_state = seed1)
XGB = XGBClassifier(booster = 'gbtree', n_estimators = 200, depth_range = 2, min_child_weight = 2, random_state = seed1, early_stopping_rounds = 10)

UBDT = BalancedBaggingClassifier(base_estimator = DT, n_estimators = 50, sampling_strategy = 1.0, max_samples = 1.0, random_state = seed1, n_jobs = -1)
UBSVM = BalancedBaggingClassifier(base_estimator = SVM, n_estimators = 20, sampling_strategy = 0.8, max_samples = 1.0, random_state = seed1, n_jobs = -1)

EASYADA = EasyEnsembleClassifier(base_estimator = ADA, sampling_strategy = 0.9, n_estimators = 10)
EASYXGB = EasyEnsembleClassifier(base_estimator = XGB, sampling_strategy = 0.9, n_estimators = 20)

############### CV EVALUATION FOR BOXPLOTS ###############

def pipeline_singleevaluationCV(X_train, Y_train, pipeline_object):
    """
    Analytics-function for pipeline-opjects that reports F2-Score for imbalanced binary classifcation (Cross Validation)
    pipeline_object = pipeline which combines classifier and sampling method.
    X_train = input (training data)
    Y_train = output (training data)
    """

    RSKCV = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = seed1)
    f2 = make_scorer(fbeta_score, beta = 2)
    scores = cross_val_score(pipeline_object, X_train, Y_train, scoring = f2, cv = RSKCV, n_jobs = -1)    
    return scores

def pipeline_multievaluationCV(X_train, Y_train, pipeline_object):
    """
    Analytics-function for pipeline-opjects that reports multiple performance metrics for imbalanced binary classifcation (Cross Validation)
    pipeline_object = pipeline which combines classifier and sampling method.
    X_train = input (training data)
    Y_train = output (training data)
    
    performance metrics = F2-Score, bacc, precision, recall
    """
    
    metricslist = {'F2': make_scorer(metrics.fbeta_score, beta = 2), 
                   'bacc': make_scorer(metrics.balanced_accuracy_score),
                   'Precision': make_scorer(metrics.precision_score),
                   'Recall': make_scorer(metrics.recall_score)}
    
    scorenames = list(['F2-Score Mean', 'F2-Score Std',
                       'bacc Mean', 'bacc Std',
                       'Precision Mean', 'Precision Std',
                       'Recall Mean', 'Recall Std'])

    RSKCV = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = seed1)
    
    scores = pd.DataFrame(np.zeros((1,8)), columns = scorenames)
    scores_temp = cross_validate(pipeline_object, X_train, Y_train, cv = RSKCV, scoring = metricslist, return_estimator = True)
    
    scores.iloc[0,0] = np.mean(scores_temp['test_F2'])
    scores.iloc[0,1] = np.std(scores_temp['test_F2'])
    scores.iloc[0,2] = np.mean(scores_temp['test_bacc'])
    scores.iloc[0,3] = np.std(scores_temp['test_bacc'])
    scores.iloc[0,4] = np.mean(scores_temp['test_Precision'])
    scores.iloc[0,5] = np.std(scores_temp['test_Precision'])
    scores.iloc[0,6] = np.mean(scores_temp['test_Recall'])
    scores.iloc[0,7] = np.std(scores_temp['test_Recall']) 
    
    scores = np.round(scores, 3)
    
    return scores

############### DEFINE COMBINATIONS VIA PIPELINES (FOR BOXPLOTS) ###############

models = []

pip_UBDT = Pipeline(steps=[('m', UBDT)])
models.append(('UBDT', pip_UBDT))
pip_UBSVM = Pipeline(steps=[('m', UBSVM)])
models.append(('UBSVM', pip_UBSVM))
pip_EASYADA = Pipeline(steps=[('m', EASYADA)])
models.append(('EASYADA', pip_EASYADA))
pip_EASYXGB = Pipeline(steps=[('m', EASYXGB)])
models.append(('EASYXGB', pip_EASYXGB))

results = []
names = []
for name, model in models:
	cv_results_new = pipeline_singleevaluationCV(X_train, Y_train, model)
	results.append(cv_results_new)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results_new.mean(), cv_results_new.std())
	print(msg)
    
fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(results, showmeans=True)
ax.set_xticklabels(names)
plt.show()