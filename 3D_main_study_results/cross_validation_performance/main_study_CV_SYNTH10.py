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
from sklearn.svm import SVC
from xgboost import XGBClassifier 

### IMBALANCED LEARN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

from imblearn.under_sampling import RandomUnderSampler 
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule

from imblearn.pipeline import Pipeline

### ADDITIONAL

import warnings
warnings.filterwarnings("ignore")
import os 
import matplotlib.pyplot as plt

############### DIRECTORY ###############

data_dir = r'/Users/i_pri/Desktop/Thesis/Data/synthetic_data'

############### LOAD DATASET ###############

df = pd.read_csv(os.path.join(data_dir, 'syntheticdata_10to1.csv'), sep = ',') 

X = df.drop(['C'], axis = 1)
Y = df.C

############### SPLIT THE DATA ###############
seed1 = 123

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = seed1, stratify = Y)

# argument 'stratify' keeps the class distributions equal across training and test data 
# split dataset into training and test data

classprop_train = sum(Y_train)/len(Y_train)
classprop_test = sum(Y_test)/len(Y_test)

############### PERFORM STANDARDIZATION ###############

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std  = scaler.transform(X_test)

X_train = pd.DataFrame(data = X_train_std, columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10'])
X_test = pd.DataFrame(data = X_test_std, columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10'])

Y_train = pd.Series.to_numpy(Y_train)

############### DEFINE CLASSIFIERS ###############

model_BAG = BaggingClassifier(base_estimator = SVC(kernel = 'rbf', probability = True, random_state = seed1),
                           n_estimators = 50, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed1)

model_XGB = XGBClassifier(booster = 'gbtree', n_estimators = 200, depth_range = 2, min_child_weight = 1, random_state = seed1, early_stopping_rounds = 10)

model_WBAG = BaggingClassifier(base_estimator = SVC(kernel = 'rbf', class_weight = {0:1, 1:2}, probability = True, random_state = seed1),
                               n_estimators = 50, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed1)

model_WXGB = XGBClassifier(booster = 'gbtree', n_estimators = 200, max_depth = 2, min_child_weight = 2, scale_pos_weight = 9, random_state = seed1, early_stopping_rounds = 10)

############### DEFINE SAMPLING METHODS ###############

ROS1 = RandomOverSampler(sampling_strategy = 0.6, random_state = seed1)
SMOTE1 = SMOTE(sampling_strategy = 0.5, k_neighbors = 3, random_state = seed1)
ADASYN1 = ADASYN(sampling_strategy = 0.4, n_neighbors = 3, random_state = seed1)

RUS1 = RandomUnderSampler(sampling_strategy = 0.5, random_state = seed1)
ENN1 = EditedNearestNeighbours(sampling_strategy = 'auto', n_neighbors = 10, kind_sel = 'all')
NCL1 = NeighbourhoodCleaningRule(sampling_strategy = 'auto', n_neighbors = 6, threshold_cleaning = 0.5, n_jobs = -1)

ROS2 = RandomOverSampler(sampling_strategy = 0.5, random_state = seed1)
SMOTE2 = SMOTE(sampling_strategy = 0.8, k_neighbors = 3, random_state = seed1)
ADASYN2 = ADASYN(sampling_strategy = 0.7, n_neighbors = 7, random_state = seed1)

RUS2 = RandomUnderSampler(sampling_strategy = 0.5, random_state = seed1)
ENN2 = EditedNearestNeighbours(sampling_strategy = 'auto', n_neighbors = 8, kind_sel = 'all')
NCL2 = NeighbourhoodCleaningRule(sampling_strategy = 'auto', n_neighbors = 6, threshold_cleaning = 0.5, n_jobs = -1)

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

############### DEFINE COMBINATIONS VIA PIPELINES (FOR LATEX TABLE) ###############

models = []

pip_BAG0 = Pipeline(steps=[('m',model_BAG)])
models.append(pip_BAG0)
pip_WBAG = Pipeline(steps=[('m',model_WBAG)])
models.append(pip_WBAG)

pip_BAGROS = Pipeline(steps=[('s', ROS1), ('m', model_BAG)])
models.append(pip_BAGROS)
pip_BAGSMOTE = Pipeline(steps=[('s', SMOTE1), ('m', model_BAG)])
models.append(pip_BAGSMOTE)
pip_BAGADASYN = Pipeline(steps=[('s', ADASYN1), ('m', model_BAG)])
models.append(pip_BAGADASYN)

pip_BAGRUS = Pipeline(steps=[('s', RUS1), ('m', model_BAG)])
models.append(pip_BAGRUS)
pip_BAGENN = Pipeline(steps=[('s', ENN1), ('m', model_BAG)])
models.append(pip_BAGENN)
pip_BAGNCL = Pipeline(steps=[('s', NCL1), ('m', model_BAG)])
models.append(pip_BAGNCL)

pip_XGB0 = Pipeline(steps=[('m',model_XGB)])
models.append(pip_XGB0)
pip_WXGB = Pipeline(steps=[('m',model_WXGB)])
models.append(pip_WXGB)

pip_XGBROS = Pipeline(steps=[('s', ROS2), ('m', model_XGB)])
models.append(pip_XGBROS)
pip_XGBSMOTE = Pipeline(steps=[('s', SMOTE2), ('m', model_XGB)])
models.append(pip_XGBSMOTE)
pip_XGBADASYN = Pipeline(steps=[('s', ADASYN2), ('m', model_XGB)])
models.append(pip_XGBADASYN)

pip_XGBRUS = Pipeline(steps=[('s', RUS2), ('m', model_XGB)])
models.append(pip_XGBRUS)
pip_XGBENN = Pipeline(steps=[('s', ENN2), ('m', model_XGB)])
models.append(pip_XGBENN)
pip_XGBNCL = Pipeline(steps=[('s', NCL2), ('m', model_XGB)])
models.append(pip_XGBNCL)

frames = []

for model in models:
	cv_results = pipeline_multievaluationCV(X_train, Y_train, model)
	frames.append(cv_results)
    
table_index = ['BAGSVM', 'BAGSVM+CSL',
            'BAGSVM+ROS', 'BAGSVM+SMOTE', 'BAGSVM+ADASYN',
            'BAGSVM+RUS', 'BAGSVM+ENN', 'BAGSVM+NCL', 
            'XGB', 'XGB+CSL',
            'XGB+ROS', 'XGB+SMOTE', 'XGB+ADASYN',
            'XGB+RUS', 'XGB+ENN', 'XGB+NCL']

cv_table = pd.concat(frames)
cv_table.index = table_index

print(pd.DataFrame.to_latex(cv_table, index = True))  

############### DEFINE COMBINATIONS VIA PIPELINES (FOR BOXPLOTS) ###############

BAG_models = []

pip_BAG0 = Pipeline(steps=[('m',model_BAG)])
BAG_models.append(('BASE', pip_BAG0))
pip_WBAG = Pipeline(steps=[('m',model_WBAG)])
BAG_models.append(('CSL', pip_WBAG))

pip_BAGROS = Pipeline(steps=[('s', ROS1), ('m', model_BAG)])
BAG_models.append(('ROS', pip_BAGROS))
pip_BAGSMOTE = Pipeline(steps=[('s', SMOTE1), ('m', model_BAG)])
BAG_models.append(('SMOTE', pip_BAGSMOTE))
pip_BAGADASYN = Pipeline(steps=[('s', ADASYN1), ('m', model_BAG)])
BAG_models.append(('ADASYN', pip_BAGADASYN))

pip_BAGRUS = Pipeline(steps=[('s', RUS1), ('m', model_BAG)])
BAG_models.append(('RUS', pip_BAGRUS))
pip_BAGENN = Pipeline(steps=[('s', ENN1), ('m', model_BAG)])
BAG_models.append(('ENN', pip_BAGENN))
pip_BAGNCL = Pipeline(steps=[('s', NCL1), ('m', model_BAG)])
BAG_models.append(('NCL', pip_BAGNCL))

XGB_models = []

pip_XGB0 = Pipeline(steps=[('m',model_XGB)])
XGB_models.append(('BASE', pip_XGB0))
pip_WXGB = Pipeline(steps=[('m',model_WXGB)])
XGB_models.append(('CSL', pip_WXGB))

pip_XGBROS = Pipeline(steps=[('s', ROS2), ('m', model_XGB)])
XGB_models.append(('ROS', pip_XGBROS))
pip_XGBSMOTE = Pipeline(steps=[('s', SMOTE2), ('m', model_XGB)])
XGB_models.append(('SMOTE', pip_XGBSMOTE))
pip_XGBADASYN = Pipeline(steps=[('s', ADASYN2), ('m', model_XGB)])
XGB_models.append(('ADASYN', pip_XGBADASYN))

pip_XGBRUS = Pipeline(steps=[('s', RUS2), ('m', model_XGB)])
XGB_models.append(('RUS', pip_XGBRUS))
pip_XGBENN = Pipeline(steps=[('s', ENN2), ('m', model_XGB)])
XGB_models.append(('ENN', pip_XGBENN))
pip_XGBNCL = Pipeline(steps=[('s', NCL2), ('m', model_XGB)])
XGB_models.append(('NCL', pip_XGBNCL))

############### BOXPLOTS ###############

results = []
names = []
for name, model in BAG_models:
	cv_results_new = pipeline_singleevaluationCV(X_train, Y_train, model)
	results.append(cv_results_new)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results_new.mean(), cv_results_new.std())
	print(msg)
    
# BOXPLOT
fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(results, showmeans=True)
ax.set_xticklabels(names)
plt.show()

results = []
names = []
for name, model in XGB_models:
	cv_results_new = pipeline_singleevaluationCV(X_train, Y_train, model)
	results.append(cv_results_new)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results_new.mean(), cv_results_new.std())
	print(msg)
    
# BOXPLOT
fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(results, showmeans=True)
ax.set_xticklabels(names)
plt.show()