############### PACKAGES ###############

### CORE PACKAGES

import numpy as np 
import pandas as pd 

### SKLEARN

from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split

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

import os 
import warnings
warnings.filterwarnings("ignore")

############### DIRECTORY ###############

data_dir = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin/data'

############### LOAD DATASET ###############

df = pd.read_csv(os.path.join(data_dir, 'breast_cancer_winsor_trafo.csv'), sep = ',') 

X = df.drop(['diagnosis'], axis = 1)
Y = df.diagnosis

seed_custom = 123

############### UNDERSAMPLE THE BREAST CANCER DATASET ###############

df_1 = df.loc[df['diagnosis'] == 1] # n=212
df_0 = df.loc[df['diagnosis'] == 0] # n=357

# we want to achieve a 1:10 ratio

np.random.seed(seed_custom)

remove_n = 170 # we have to delete 170 pos. obs to achieve the 1:10 ratio.
drop_indices = np.random.choice(df_1.index, remove_n, replace=False)
df_1_subset = df_1.drop(drop_indices)

df_undersampled = pd.concat([df_1_subset, df_0], axis = 0) # merge the dataset back together => n_total = 429
df_undersampled = df_undersampled.sample(frac=1).reset_index(drop=True) # shuffle the dataset

X = df_undersampled.drop(['diagnosis'], axis = 1)
Y = df_undersampled.diagnosis

############### SPLIT THE DATA ###############
seed1 = 123
seed2 = 456
seed3 = 789

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = seed1, stratify = Y)

# argument 'stratify' keeps the class distributions equal across training and test data 
# split dataset into training and test data

classprop_train = sum(Y_train)/len(Y_train)
classprop_test = sum(Y_test)/len(Y_test)

featurenames = X_train.columns

############### PERFORM STANDARDIZATION ###############

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std  = scaler.transform(X_test)

X_train = pd.DataFrame(data = X_train_std, columns = featurenames)
X_test = pd.DataFrame(data = X_test_std, columns = featurenames)

Y_train = pd.Series.to_numpy(Y_train)

############### DEFINE CLASSIFIERS ###############

model_BAG = BaggingClassifier(base_estimator = SVC(kernel = 'rbf', probability = True, random_state = seed1),
                           n_estimators = 50, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed1)

model_XGB = XGBClassifier(booster = 'gbtree', n_estimators = 200, depth_range = 2, min_child_weight = 2, random_state = seed1, early_stopping_rounds = 10)

model_WBAG = BaggingClassifier(base_estimator = SVC(kernel = 'rbf', class_weight = 'balanced', probability = True, random_state = seed1),
                               n_estimators = 50, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed1)

xgb_weight = len(Y_train)/sum(Y_train)

model_WXGB = XGBClassifier(booster = 'gbtree', n_estimators = 200, max_depth = 2, min_child_weight = 2, scale_pos_weight = xgb_weight, random_state = seed1, early_stopping_rounds = 10)

############### DEFINE SAMPLING METHODS ###############

ROS1 = RandomOverSampler(sampling_strategy = 1.0, random_state = seed1)
SMOTE1 = SMOTE(sampling_strategy = 1.0, k_neighbors = 5, random_state = seed1)
ADASYN1 = ADASYN(sampling_strategy = 1.0, n_neighbors = 5, random_state = seed1)

RUS1 = RandomUnderSampler(sampling_strategy = 1.0, random_state = seed1)
ENN1 = EditedNearestNeighbours(sampling_strategy = 'auto', n_neighbors = 3, kind_sel = 'all')
NCL1 = NeighbourhoodCleaningRule(sampling_strategy = 'auto', n_neighbors = 3, threshold_cleaning = 0.5, n_jobs = -1)

ROS2 = RandomOverSampler(sampling_strategy = 1.0, random_state = seed1)
SMOTE2 = SMOTE(sampling_strategy = 1.0, k_neighbors = 5, random_state = seed1)
ADASYN2 = ADASYN(sampling_strategy = 1.0, n_neighbors = 5, random_state = seed1)

RUS2 = RandomUnderSampler(sampling_strategy = 1.0, random_state = seed1)
ENN2 = EditedNearestNeighbours(sampling_strategy = 'auto', n_neighbors = 5, kind_sel = 'all')
NCL2 = NeighbourhoodCleaningRule(sampling_strategy = 'auto', n_neighbors = 5, threshold_cleaning = 0.5, n_jobs = -1)

############### DEFINE COMBINATIONS VIA PIPELINES ###############

model_combo = []

pip_BAG0 = Pipeline(steps=[('m',model_BAG)])
model_combo.append(pip_BAG0)
pip_WBAG = Pipeline(steps=[('m',model_WBAG)])
model_combo.append(pip_WBAG)

pip_BAGROS = Pipeline(steps=[('s', ROS1), ('m', model_BAG)])
model_combo.append(pip_BAGROS)
pip_BAGSMOTE = Pipeline(steps=[('s', SMOTE1), ('m', model_BAG)])
model_combo.append(pip_BAGSMOTE)
pip_BAGADASYN = Pipeline(steps=[('s', ADASYN1), ('m', model_BAG)])
model_combo.append(pip_BAGADASYN)

pip_BAGRUS = Pipeline(steps=[('s', RUS1), ('m', model_BAG)])
model_combo.append(pip_BAGRUS)
pip_BAGENN = Pipeline(steps=[('s', ENN1), ('m', model_BAG)])
model_combo.append(pip_BAGENN)
pip_BAGNCL = Pipeline(steps=[('s', NCL1), ('m', model_BAG)])
model_combo.append(pip_BAGNCL)

pip_XGB0 = Pipeline(steps=[('m',model_XGB)])
model_combo.append(pip_XGB0)
pip_WXGB = Pipeline(steps=[('m',model_WXGB)])
model_combo.append(pip_WXGB)

pip_XGBROS = Pipeline(steps=[('s', ROS2), ('m', model_XGB)])
model_combo.append(pip_XGBROS)
pip_XGBSMOTE = Pipeline(steps=[('s', SMOTE2), ('m', model_XGB)])
model_combo.append(pip_XGBSMOTE)
pip_XGBADASYN = Pipeline(steps=[('s', ADASYN2), ('m', model_XGB)])
model_combo.append(pip_XGBADASYN)

pip_XGBRUS = Pipeline(steps=[('s', RUS2), ('m', model_XGB)])
model_combo.append(pip_XGBRUS)
pip_XGBENN = Pipeline(steps=[('s', ENN2), ('m', model_XGB)])
model_combo.append(pip_XGBENN)
pip_XGBNCL = Pipeline(steps=[('s', NCL2), ('m', model_XGB)])
model_combo.append(pip_XGBNCL)

############### ANALYTICS FUNCTION ###############

def predictive_performance(classifier_object, X_train, Y_train, X_test, Y_test):
    """
    Function to compute performance criteria for imbalanced binary classification (with continous features only!).
    
    Considered Performance Criteria:
    - F2-Score
    - Balanced Accuracy
    - Precision
    - Recall
    
    Inputs:
    classifier_object = model, e.g. BaggingClassifier()

    X_train = features of the training data (must be in pd.Dataframe format!!)
    Y_train = outcome of the training data (must be in pd.Series format!!)
    
    X_test = features of the test data (must be in pd.Dataframe format!!)
    Y_test = outcome of the test data (must be in pd.Series format!!)
    """  
    test_performance = pd.DataFrame(np.zeros((1,4)), columns = list(['F2-Score', 'bacc', 'Precision', 'Recall']))
    
    classifier_object.fit(X_train, Y_train)
    Y_pred = classifier_object.predict(X_test)
    
    f2              = round(metrics.fbeta_score(Y_test, Y_pred, beta = 2), 3)
    balacc          = round(metrics.balanced_accuracy_score(Y_test, Y_pred), 3)
    precision       = round(metrics.precision_score(Y_test, Y_pred), 3)
    recall          = round(metrics.recall_score(Y_test, Y_pred), 3)
    
    test_performance.iloc[0,0] = f2
    test_performance.iloc[0,1] = balacc
    test_performance.iloc[0,2] = precision
    test_performance.iloc[0,3] = recall

    return test_performance

frames = []

for pipe in model_combo:
    score = predictive_performance(pipe, X_train, Y_train, X_test, Y_test)
    frames.append(score)

############### FINAL TABLE FOR LATEX ###############

table_index = ['BAGSVM', 'BAGSVM+CSL',
            'BAGSVM+ROS', 'BAGSVM+SMOTE', 'BAGSVM+ADASYN',
            'BAGSVM+RUS', 'BAGSVM+ENN', 'BAGSVM+NCL', 
            'XGB', 'XGB+CSL',
            'XGB+ROS', 'XGB+SMOTE', 'XGB+ADASYN',
            'XGB+RUS', 'XGB+ENN', 'XGB+NCL']

test_table = pd.concat(frames)
test_table.index = table_index
test_table = np.round(test_table, 3)

print(pd.DataFrame.to_latex(test_table, index = True))  