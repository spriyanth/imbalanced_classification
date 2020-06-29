############### PACKAGES ###############

### CORE PACKAGES

import numpy as np 
import pandas as pd 

### SKLEARN

from sklearn import preprocessing
from sklearn import metrics 
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

from imblearn.pipeline import Pipeline

### ADDITIONAL

import os 
import random
import warnings
warnings.filterwarnings("ignore")

############### DIRECTORY ###############

data_dir = r'/Users/i_pri/Desktop/Thesis/Data/synthetic_data'

############### LOAD DATASET ###############

df = pd.read_csv(os.path.join(data_dir, 'syntheticdata_10to1.csv'), sep = ',') 

X = df.drop(['C'], axis = 1)
Y = df.C

############### SPLIT THE DATA ###############
seed1 = 123
seed2 = 456
seed3 = 789

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


############### DEFINE SAMPLING METHODS ###############

ROS = RandomOverSampler
SMOTE = SMOTE
ADASYN = ADASYN
RUS = RandomUnderSampler

### OPTIMIZED PARAMS

ROS_params1 = dict(sampling_strategy = 0.6)
SMOTE_params1 = dict(sampling_strategy = 0.5, k_neighbors = 3)
ADASYN_params1 = dict(sampling_strategy = 0.4, n_neighbors = 3)
RUS_params1 = dict(sampling_strategy = 0.6)

ROS_params2 = dict(sampling_strategy = 0.5)
SMOTE_params2 = dict(sampling_strategy = 0.8, k_neighbors = 3)
ADASYN_params2 = dict(sampling_strategy = 0.7, n_neighbors = 7)
RUS_params2 = dict(sampling_strategy = 0.5)

### DEFAULT PARAMS

ROS_params3 = dict(sampling_strategy = 1.0)
SMOTE_params3 = dict(sampling_strategy = 1.0, k_neighbors = 5)
ADASYN_params3 = dict(sampling_strategy = 1.0, n_neighbors = 5)
RUS_params3 = dict(sampling_strategy = 1.0)

############### REPEATED EXPERIMENTS ###############

def RepeatedSampling(X_train, Y_train, X_test, Y_test, classifier, sampling, sampling_params, no_seeds):
    """
    Repeated sampling experiments for algorithms with randomized sampling.
    
    Considered Performance Criteria:
    - F2-Score
    - Balanced Accuracy
    - Precision
    - Recall
    
    Inputs:
    X_train = features of the training data (must be in pd.Dataframe format!!)
    Y_train = outcome of the training data (must be in pd.Series format!!)
    
    X_test = features of the test data (must be in pd.Dataframe format!!)
    Y_test = outcome of the test data (must be in pd.Series format!!)
    
    classifier = model, e.g. BaggingClassifier()
    
    sampling = sampling object, e.g. RandomOverSampler()
    sampling_params = parameters for the sampling object
    
    no_seeds = number of experiments to be executed/number of different seeds to be considered.
    """  
    
    seeds = random.sample(range(1, 10000), no_seeds)
    i=0
    
    test_frames = []

    for seed in seeds:
        sampling_new = sampling(**sampling_params, random_state = seeds[i])
        pipe = Pipeline(steps=[('s', sampling_new), ('m', classifier)])
        
        test_performance = pd.DataFrame(np.zeros((1,4)), columns = list(['F2-Score', 'bacc', 'Precision', 'Recall']))
    
        pipe.fit(X_train, Y_train)
        Y_pred = pipe.predict(X_test)

        f2              = metrics.fbeta_score(Y_test, Y_pred, beta = 2)
        bacc            = metrics.balanced_accuracy_score(Y_test, Y_pred)
        precision       = metrics.precision_score(Y_test, Y_pred)
        recall          = metrics.recall_score(Y_test, Y_pred)
    
        test_performance.iloc[0,0] = f2
        test_performance.iloc[0,1] = bacc
        test_performance.iloc[0,2] = precision
        test_performance.iloc[0,3] = recall
        
        test_frames.append(test_performance)
        i = i+1
        
    test_table = pd.concat(test_frames)
    
    f2_vals = test_table['F2-Score']
    bacc_vals = test_table['bacc']
    precision_vals = test_table['Precision']
    recall_vals = test_table['Recall']
    
    final_performance = pd.DataFrame(np.zeros((1,8)), columns = list(['F2-Score MEAN', 'F2-Score STD', 'bacc MEAN', 'bacc STD', 'Precision MEAN', 'Precision STD', 'Recall MEAN', 'Recall STD']))
    
    final_performance.iloc[0,0] = np.mean(f2_vals)
    final_performance.iloc[0,2] = np.mean(bacc_vals)
    final_performance.iloc[0,4] = np.mean(precision_vals)
    final_performance.iloc[0,6] = np.mean(recall_vals)
    
    final_performance.iloc[0,1] = np.std(f2_vals)
    final_performance.iloc[0,3] = np.std(bacc_vals)
    final_performance.iloc[0,5] = np.std(precision_vals)
    final_performance.iloc[0,7] = np.std(recall_vals)
    
    final_performance = round(final_performance, 3)
    
    return final_performance
    
############### BAGGING RESULTS W/ OPTIMIZED PARAMS ###############
    
BAGROS_performance = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_BAG, ROS, ROS_params1, 100)
BAGSMOTE_performance = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_BAG, SMOTE, SMOTE_params1, 100)
BAGADASYN_performance = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_BAG, ADASYN, ADASYN_params1, 100)
BAGRUS_performance = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_BAG, RUS, RUS_params1, 100)

############### XGB RESULTS W/ OPTIMIZED PARAMS ###############

XGBROS_performance = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_XGB, ROS, ROS_params1, 100)
XGBSMOTE_performance = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_XGB, SMOTE, SMOTE_params1, 100)
XGBADASYN_performance = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_XGB, ADASYN, ADASYN_params1, 100)
XGBRUS_performance = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_XGB, RUS, RUS_params1, 100)

############### BAGGING RESULTS W/ OPTIMIZED PARAMS ###############
    
BAGROS_performance_def = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_BAG, ROS, ROS_params3, 100)
BAGSMOTE_performance_def = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_BAG, SMOTE, SMOTE_params3, 100)
BAGADASYN_performance_def = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_BAG, ADASYN, ADASYN_params3, 100)
BAGRUS_performance_def = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_BAG, RUS, RUS_params3, 100)

############### XGB RESULTS W/ OPTIMIZED PARAMS ###############

XGBROS_performance_def = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_XGB, ROS, ROS_params3, 100)
XGBSMOTE_performance_def = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_XGB, SMOTE, SMOTE_params3, 100)
XGBADASYN_performance_def = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_XGB, ADASYN, ADASYN_params3, 100)
XGBRUS_performance_def = RepeatedSampling(X_train, Y_train, X_test, Y_test, model_XGB, RUS, RUS_params3, 100)