############### PACKAGES ###############

### CORE PACKAGES

import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn import metrics 
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, train_test_split, RepeatedStratifiedKFold, StratifiedKFold, RandomizedSearchCV

import warnings
warnings.filterwarnings("ignore")

### MACHINE LEARNING METHODS

from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier 

### ADDITIONAL

import os 
import random
import math

############### DIRECTORY ###############

data_dir = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin/data'

############### LOAD DATASET ###############

df = pd.read_csv(os.path.join(data_dir, 'breast_cancer_winsor_trafo.csv'), sep = ',') 

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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = seed_custom, stratify = Y)

# argument 'stratify' keeps the class distributions equal across training and test data 
# split dataset into training and test data

classprop_train = sum(Y_train)/len(Y_train)
classprop_test = sum(Y_test)/len(Y_test) # check: alpha = 0.2 for train and test.

featurenames = X_train.columns

############### PERFORM STANDARDIZATION ###############

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std  = scaler.transform(X_test)

X_train = pd.DataFrame(data = X_train_std, columns = featurenames)
X_test = pd.DataFrame(data = X_test_std, columns = featurenames)

Y_train = pd.Series.to_numpy(Y_train)

############### ANALYTICS FUNCTION ###############

def algo_CVmetrics(classifier_object, X_train, Y_train):
    """
    Analytics-function that reports performance metrics for imbalanced binary classifcation (Cross Validation)
    classifier object = classification method e.g. DecisionTreeClassifier()
    X_train = input (training data)
    Y_train = output (training data)
    """
    
    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = seed_custom)
    
    metricslist = {'f2': make_scorer(metrics.fbeta_score, beta = 2), 
                   'balacc': make_scorer(metrics.balanced_accuracy_score),
                   'precision': make_scorer(metrics.precision_score),
                   'recall': make_scorer(metrics.recall_score)}
    
    cv_results = cross_validate(classifier_object, X_train, Y_train, cv = cv, scoring = metricslist, return_estimator = True)
    
    f2_mean           = np.mean(cv_results['test_f2'])
    f2_std            = np.std(cv_results['test_f2'])
    
    balacc_mean       = np.mean(cv_results['test_balacc'])
    balacc_std        = np.std(cv_results['test_balacc'])

    precision_mean    = np.mean(cv_results['test_precision'])
    precision_std     = np.std(cv_results['test_precision'])
    
    recall_mean       = np.mean(cv_results['test_recall'])
    recall_std        = np.std(cv_results['test_recall'])
    
    scorebox = pd.DataFrame(np.zeros((1,8)), columns = list(['F2-Score Mean', 'F2-Score STD', 'Balanced Accuracy Mean', 'Balanced Accuracy STD',
                                                             'Precision Mean', 'Precision STD', 'Recall Mean', 'Recall STD']))
    
    scorebox.iloc[0,0] = f2_mean
    scorebox.iloc[0,1] = f2_std
    scorebox.iloc[0,2] = balacc_mean
    scorebox.iloc[0,3] = balacc_std
    scorebox.iloc[0,4] = precision_mean
    scorebox.iloc[0,5] = precision_std
    scorebox.iloc[0,6] = recall_mean
    scorebox.iloc[0,7] = recall_std  
    
    scorebox = np.round(scorebox, 3)
    
    print("Model has a mean CV balanced accuracy of {0}, (Std: {1})".format(round(balacc_mean,3), round(balacc_std,3)))
    print("Model has a mean CV F2_Score of {0}, (Std: {1})".format(round(f2_mean,3), round(f2_std,3)))
    print("Model has a mean CV Precision of {0}, (Std: {1})".format(round(precision_mean,3), round(precision_std,3)))
    print("Model has a mean CV Recall of {0}, (Std: {1})".format(round(recall_mean,3), round(recall_std,3)))
    
    return scorebox

############### (SEQUENTIAL) RANDOM SEARCH FUNCTION ###############

def multi_RSCV(method, grid, X, Y, metric, n_candidates, it):
    """
    Perform multiple explorations of Random Search and gather the best candidate with the respective parameters and metrics for each round.
    Number of rounds and number of iterations within each round are free to choose.
    Good starting point: 3 rounds with each 100 iterations, if results are not similar, expand the numbers and change the grid.
    
    method = classifier object
    grid = parameter grid settings for the to-be-optimized classifier
    X = Input (Training Data)
    Y = Output (Training Data)
    metric = to-be-optimized metric
    n_candidates = number of candidates we want = number of iterations we run the random search optimization
    it = number of iterations/settings to test out of all possibilities from the grid
    """
    
    params_box = [None] * n_candidates
    metrics_box = pd.DataFrame(np.zeros((n_candidates, 1)), columns = list(['Score']))
    
    cv_KF = StratifiedKFold(n_splits = 5, shuffle = True)
    
    for i in range(n_candidates):
        seed_temp = math.ceil(random.uniform(1,1000))
        model = RandomizedSearchCV(method, grid, n_iter = it, cv = cv_KF, n_jobs = -1, scoring = metric, random_state = seed_temp)    
        model.fit(X,Y)
        params_box[i] = model.best_params_
        metrics_box.iloc[i,0] = model.best_score_
            
    return params_box, metrics_box

############### HYPERPARAMETER TUNING FOR BASELINE MODELS ###############

f2_metric = make_scorer(metrics.fbeta_score, beta = 2)

############### BAGGED SVM OPTIMIZATION ###############

SVM_rbf   = SVC(kernel = 'rbf', probability = True, random_state = seed_custom) 
BAGSVM = BaggingClassifier(base_estimator = SVM_rbf, n_estimators = 50, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed_custom)

BAGSVM_metrics = algo_CVmetrics(classifier_object = BAGSVM, X_train = X_train, Y_train = Y_train)

# use default settings for a more conservative and generalized setup => bacc = 0.861, F2-Score = 0.757

############### XGB OPTIMIZATION ###############

XGBDT = XGBClassifier(booster = 'gbtree', n_estimators = 200, random_state = seed_custom, early_stopping_rounds = 10)
XGBDT_metrics = algo_CVmetrics(classifier_object = XGBDT, X_train = X_train, Y_train = Y_train)

# base performance: bacc = 0.892, F2-Score = 0.811

############### GRID SETUP ###############

depth_range = np.arange(1, 4, 1) # not too high => leads to overfitting
child_weight_range = np.arange(1, 4, 1) # small weight => allows for complex models => overfitting

XGB_custom = XGBClassifier(booster = 'gbtree', n_estimators = 200, random_state = seed_custom, early_stopping_rounds = 10)
xgb_grid_final = dict(max_depth = depth_range, min_child_weight = child_weight_range)

############### PERFORM MULTI RANDOM SEARCH ###############

XGBopt_params_final, XGBopt_metrics_final = multi_RSCV(method = XGB_custom, grid = xgb_grid_final, X = X_train, Y = Y_train, metric = f2_metric, n_candidates = 5, it = 100)

xgb_opt = XGBClassifier(booster = 'gbtree', n_estimators = 200, max_depth = 2, min_child_weight = 2, random_state = seed_custom, early_stopping_rounds = 10)
XGBopt_metrics = algo_CVmetrics(classifier_object = xgb_opt, X_train = X_train, Y_train = Y_train)

# post-optimization performance: bacc = 0.889, F2-Score = 0.806 => marginally worse but simpler trees => seems better to avoid overfitting + save computational costs