############### PACKAGES ###############

### CORE PACKAGES

import numpy as np 
import pandas as pd 

### SKLEARN

from sklearn import preprocessing
from sklearn import metrics 
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate, train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV

### MACHINE LEARNING METHODS
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier 

### ADDITIONAL

import os 
import warnings
warnings.filterwarnings("ignore")

############### DIRECTORY ###############

data_dir = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin/data'

############### LOAD DATASET ###############

df = pd.read_csv(os.path.join(data_dir, 'breast_cancer_winsor_trafo.csv'), sep = ',') 

seed_custom = 123

############### UNDERSAMPLE THE BREAST CANCER DATASET ###############

df_1 = df.loc[df['diagnosis'] == 1] # n=212
df_0 = df.loc[df['diagnosis'] == 0] # n=357

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
    
    cv_KF = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = seed_custom)
    
    for i in range(n_candidates):
        # seed_temp = math.ceil(random.uniform(1,1000))
        model = RandomizedSearchCV(method, grid, n_iter = it, cv = cv_KF, n_jobs = -1, scoring = metric)    
        model.fit(X,Y)
        params_box[i] = model.best_params_
        metrics_box.iloc[i,0] = model.best_score_
            
    return params_box, metrics_box

############### BASELINE PERFORMANCE ###############
    
SVM = SVC(kernel = 'rbf', probability = True, random_state = seed_custom)
BAGSVM_opt = BaggingClassifier(base_estimator = SVM, n_estimators = 50, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed_custom)
BAGSVM_base_performance = algo_CVmetrics(classifier_object = BAGSVM_opt, X_train = X_train, Y_train = Y_train)

## Balanced Accuracy: 0.861 / F2-Score: 0.757


XGB_opt = XGBClassifier(booster = 'gbtree', n_estimators = 200, max_depth = 3, min_child_weight = 1, random_state = seed_custom, early_stopping_rounds = 10)
XGB_base_performance = algo_CVmetrics(classifier_object = XGB_opt, X_train = X_train, Y_train = Y_train)

## Balanced Accuracy: 0.878 / F2-Score: 0.788

############### BAGGING + WEIGHTED SVM (WITH HEURISTIC WEIGHTS) ###############

WSVM = SVC(kernel = 'rbf', class_weight = 'balanced', probability = True, random_state = seed_custom) 
BAGWSVM = BaggingClassifier(base_estimator = WSVM, n_estimators = 50, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed_custom)
BAGWSVM_metrics = algo_CVmetrics(classifier_object = BAGWSVM, X_train = X_train, Y_train = Y_train) # bacc = 0.872, F2-Score = 0.774 => minimal improvement (performance is already very high, it's hard to improve the model further w/o losing the ability to generalize on unseen data)

############### BAGGING + WEIGHTED SVM (WITH OPTIMIZED WEIGHTS) ###############

weight_range = [{0:1,1:1}, {0:1,1:1.5}, {0:1,1:2}, {0:1,1:2.5}, {0:1,1:3}, {0:1,1:3.5}, {0:1,1:4}, {0:1,1:4.5}, {0:1,1:5},{0:1,1:5.5},
                {0:1,1:6}, {0:1,1:6.5}, {0:1,1:7}, {0:1,1:7.5}, {0:1,1:8}, {0:1,1:8.5}, {0:1,1:9}, {0:1,1:9.5}, {0:1,1:10}]

RS_SVM_grid = dict(base_estimator__class_weight = weight_range)

f2_metric = make_scorer(metrics.fbeta_score, beta = 2)

BAGWSVM_params, BAGWSVM_scores = multi_RSCV(method = BAGSVM_opt, grid = RS_SVM_grid, X = X_train, Y = Y_train, metric = f2_metric, n_candidates = 3, it = 100)

### FIT OPTIMIZED BAGWSVM

WSVM_opt = SVC(kernel = 'rbf', class_weight = {0:1, 1:2.5}, probability = True, random_state = seed_custom) 
BAGWSVM_opt = BaggingClassifier(base_estimator = WSVM_opt, n_estimators = 50, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed_custom)
BAGWSVM_opt_metrics = algo_CVmetrics(classifier_object = BAGWSVM_opt, X_train = X_train, Y_train = Y_train) # bacc = 0.874, F2-Score = 0.778 

## optimized weights and heuristic weights are approx. the same..
    
############### WEIGHTED XGB (WITH HEURISTIC WEIGHTS) ###############
    
inverse_classprop = len(Y_train)/sum(Y_train)

WXGB = XGBClassifier(booster = 'gbtree', scale_pos_weight = inverse_classprop, n_estimators = 200, depth_range = 3, min_child_weight = 1, random_state = seed_custom, early_stopping_rounds = 10)
WXGB_metrics = algo_CVmetrics(classifier_object = WXGB, X_train = X_train, Y_train = Y_train) # bacc = 0.878, F2-Score = 0.779 => worse than baseline.

############### WEIGHTED XGB (WITH OPTIMIZED WEIGHTS) ###############

scale = np.arange(1, 10, 0.5) # wider grid is prone to overfitting => always chooses max value of the grid..
depth = np.arange(1, 3, 1) # we want small trees (weak learners) for boosting
child = np.arange(1, 3, 1)

RS_XGB_grid = dict(scale_pos_weight = scale, max_depth = depth, min_child_weight = child)

XGB_base = XGBClassifier(booster = 'gbtree', n_estimators = 200, random_state = seed_custom, early_stopping_rounds = 10)

WXGB_params, WXGB_scores = multi_RSCV(method = XGB_base, grid = RS_XGB_grid, X = X_train, Y = Y_train, metric = f2_metric, n_candidates = 3, it = 100)

WXGB_opt = XGBClassifier(booster = 'gbtree', n_estimators = 200, scale_pos_weight = 7, max_depth = 2, min_child_weight = 2, random_state = seed_custom, early_stopping_rounds = 10)
WXGB_opt_metrics = algo_CVmetrics(classifier_object = WXGB_opt, X_train = X_train, Y_train = Y_train)  # BAcc = 0.921, F2-Score = 0.852 =>  better than heuristic weights