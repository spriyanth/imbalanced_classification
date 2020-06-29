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

data_dir = r'/Users/i_pri/Desktop/Thesis/Data/synthetic_data'

############### LOAD DATASET ###############

df = pd.read_csv(os.path.join(data_dir, 'syntheticdata_10to1.csv'), sep = ',') 

X = df.drop(['C'], axis = 1)
Y = df.C

############### SPLIT THE DATA ###############
seed_custom = 123

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = seed_custom, stratify = Y)

# argument 'stratify' keeps the class distributions equal across training and test data 
# split dataset into training and test data

classprop_train = sum(Y_train)/len(Y_train)
classprop_test = sum(Y_test)/len(Y_test)

############### PERFORM STANDARDIZATION ###############

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std  = scaler.transform(X_test)

X_train = X_train_std
X_test = X_test_std

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

############### BAGGED SVM OPTIMIZATION ###############

SVM_rbf   = SVC(kernel = 'rbf', probability = True, random_state = seed_custom) 
BAGSVM = BaggingClassifier(base_estimator = SVM_rbf, n_estimators = 50, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed_custom)

BAGSVM_metrics = algo_CVmetrics(classifier_object = BAGSVM, X_train = X_train, Y_train = Y_train)

f2_metric = make_scorer(metrics.fbeta_score, beta = 2)

# ############### GRID SETUP ###############

# ## Grid-input inspired by Scikit-Learn. Source: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

# # Parameter 'C' defines the tradeoff between correct classification of training observations (insample performance) against maximization of the decision function's margin
# # C = regularziation parameter of SVM
# # Large C => smaller margin will be accepted
# # Small C => larger margin will be accepted => simpler decision function

# # Parameter 'gamma' = kernel smoothing coefficient

# # C_range = np.arange(10, 110, 10) # C = 10 => best candidate in previous grid setup
# gamma_range = np.arange(0.05, 0.26, 0.01)
# # samplesize_range = np.arange(0.5, 0.9, 0.1) => max_samples = 0.8 => best candidate (naturally, more data = better fit)
# # BAGSVM_grid = dict(base_estimator__C = C_range, base_estimator__gamma = gamma_range)
# BAGSVM_grid = dict(base_estimator__gamma = gamma_range)

# SVM_custom = SVC(kernel = 'rbf', C = 10, probability = True, random_state = seed_custom)
# BAGSVM_custom = BaggingClassifier(base_estimator = SVM_custom, n_estimators = 50, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed_custom)

# ############### PERFORM MULTI RANDOM SEARCH ###############

# BAGSVMopt_params1, BAGSVMopt_metrics1 = multi_RSCV(method = BAGSVM_custom, grid = BAGSVM_grid, X = X_train, Y = Y_train, metric = f2_metric, n_candidates = 5, it = 100)

# ## Define the optimized SVM and optimized BAGSVM
# SVM_opt = SVC(kernel = 'rbf', C = 10, gamma = 0.1, probability = True, random_state = seed_custom)
# BAGSVM_opt = BaggingClassifier(base_estimator = SVM_opt, n_estimators = 100, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed_custom)
# BAGSVM_opt_metrics = algo_CVmetrics(classifier_object = BAGSVM_opt, X_train = X_train, Y_train = Y_train, K = 5)

## RS optimization leads to strong overfitting.. => not used for the SVMs in Bagging!

############### XGB OPTIMIZATION ###############

XGBDT = XGBClassifier(booster = 'gbtree', n_estimators = 200, random_state = seed_custom, early_stopping_rounds = 10)
XGBDT_metrics = algo_CVmetrics(classifier_object = XGBDT, X_train = X_train, Y_train = Y_train)

############### GRID SETUP ###############

## Source for parameter description: https://xgboost.readthedocs.io/en/latest/parameter.html

# eta = learning parameter (step size shrinkage) => defines how strongly the next boosting iteration impacts the entire ensemble 
# small eta => less prone to overfitting

# gamma = minimum loss required to split an arbitrary leaf node into additional leaf nodes
# high gamma = smaller trees = conservative/simplified DT

## gamma and eta did not converge to a clear result. optimal values were different in each run.. (left out of the tuning => use default values)

# max_depth = individual tree size of each learner, the higher the more likely to overfit due to complexity
# we want rather small trees for the purpose of "weak learners"

depth_range = np.arange(1, 4, 1) # not too high => leads to overfitting
child_weight_range = np.arange(1, 4, 1) # small weight => allows for complex models => overfitting

XGB_custom = XGBClassifier(booster = 'gbtree', n_estimators = 200, random_state = seed_custom, early_stopping_rounds = 10)
xgb_grid_final = dict(max_depth = depth_range, min_child_weight = child_weight_range)

############### PERFORM MULTI RANDOM SEARCH ###############

XGBopt_params_final, XGBopt_metrics_final = multi_RSCV(method = XGB_custom, grid = xgb_grid_final, X = X_train, Y = Y_train, metric = f2_metric, n_candidates = 5, it = 100)

xgb_opt = XGBClassifier(booster = 'gbtree', n_estimators = 200, depth_range = 2, min_child_weight = 1, random_state = seed_custom, early_stopping_rounds = 10)
XGBopt_metrics = algo_CVmetrics(classifier_object = xgb_opt, X_train = X_train, Y_train = Y_train)
