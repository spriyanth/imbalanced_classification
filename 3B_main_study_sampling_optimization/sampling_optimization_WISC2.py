############### PACKAGES ###############

### CORE PACKAGES

import numpy as np 
import pandas as pd 

### SKLEARN
from sklearn import preprocessing
from sklearn import metrics 
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import cross_validate, train_test_split, RepeatedStratifiedKFold, StratifiedKFold

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

############### SPLIT THE DATA ###############
seed_custom = 123

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = seed_custom, stratify = Y)

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

############### DEFINE BASELINE MODELS (1A & 2A) ###############
    
BAGSVM = BaggingClassifier(base_estimator = SVC(kernel = 'rbf', probability = True, random_state = seed_custom),
                           n_estimators = 100, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed_custom)
BAGSVM_base_performance = algo_CVmetrics(classifier_object = BAGSVM, X_train = X_train, Y_train = Y_train)

## Balanced Accuracy: 0.941 / F2-Score: 0.918


XGBDT = XGBClassifier(booster = 'gbtree', n_estimators = 200, max_depth = 3, min_child_weight = 1, random_state = seed_custom, early_stopping_rounds = 10)
XGBDT_base_performance = algo_CVmetrics(classifier_object = XGBDT, X_train = X_train, Y_train = Y_train)

## Balanced Accuracy: 0.938 / F2-Score: 0.918

# => approx. equal performance.

############### BUILD PIPELINE FOR SAMPLING & MODEL FITTING ###############
    
def balanced_fit(X, y, model, model_params, sampling, sampling_params):
    """
    Manual stratified K-fold CV to ensure that only the training set is over-/undersampled.
    Returns the cross-validated performance metrics of the fitted model on the balanced training data.
    
    X = features of the training data (must be in pd.Dataframe format!!)
    y = outcome of the training data (must be in pd.Series format!!)
    model = classifier object, e.g. BaggingClassifier()
    model_params = parameters for the classifier object
    sampling = sampling algorithm to balance the dataset, e.g. RandomOverSampler()
    sampling_params = sampling ratio = number of minority examples after resampling divided by number of majority examples (if undefined => ratio = 1)
    """
    SKF = StratifiedKFold(n_splits = 5, random_state = seed_custom, shuffle = False)
      
    sampling = sampling(**sampling_params)

    f2_fold_scores = []
  
    for train_fold_index, val_fold_index in SKF.split(X_train, Y_train):
        
        # Define training data
        X_train_fold, Y_train_fold = X_train.iloc[train_fold_index], Y_train[train_fold_index]
        
        # Define validation data
        X_val_fold, Y_val_fold = X_train.iloc[val_fold_index], Y_train[val_fold_index]

        # Resample only training data
        X_train_fold_balanced, Y_train_fold_balanced = sampling.fit_resample(X_train_fold, Y_train_fold)
        
        # Fit the model to the balanced training data
        model_obj = model(**model_params).fit(X_train_fold_balanced, Y_train_fold_balanced)
        
        # Compute performance metrics
        
        Y_pred = model_obj.predict(X_val_fold)
        f2_val = fbeta_score(Y_val_fold, Y_pred, beta = 2)
        f2_fold_scores.append(f2_val)
        
    f2_mean = np.mean(f2_fold_scores)
    f2_std  = np.std(f2_fold_scores)

    print("Model has a mean CV F2-Score of {0}, (Std: {1})".format(round(f2_mean,3), round(f2_std,3)))
    
    return f2_mean

### MODEL INPUT

BAG = BaggingClassifier
BAG_params = dict(base_estimator = SVC(kernel = 'rbf', probability = True, random_state = seed_custom),
                           n_estimators = 100, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed_custom)

XGB = XGBClassifier
XGB_params = dict(booster = 'gbtree', n_estimators = 200, max_depth = 3, min_child_weight = 1, random_state = seed_custom, early_stopping_rounds = 10)

############### MODEL 1B: BAGGING + RANDOM OVERSAMPLING  ###############
    
ROS = RandomOverSampler 
ROS_params = dict(sampling_strategy = 'auto', random_state = seed_custom)

BAG_ROS_f2 = balanced_fit(X_train, Y_train, BAG, BAG_params, ROS, ROS_params) 

### GRID SEARCH OPTIMIZATION

ROS_grid = dict(sampling_strategy = [0.7, 0.8, 0.9, 1.0])

BAG_ROS_scoreboard = []

for alpha in ROS_grid['sampling_strategy']:
        ROS_params_new = dict(sampling_strategy = alpha, random_state = seed_custom)
        ROS_params_new['F2-Score'] = balanced_fit(X_train, Y_train, BAG, BAG_params, ROS, ROS_params_new)
        BAG_ROS_scoreboard.append(ROS_params_new)
        
BAG_ROS_scoreboard = sorted(BAG_ROS_scoreboard, key=lambda x: x['F2-Score'], reverse=True) 

### OPTIMIZED MODEL 1B

ROSopt_params = dict(sampling_strategy = 0.9, random_state = seed_custom)
BAG_ROSopt_f2 = balanced_fit(X_train, Y_train, BAG, BAG_params, ROS, ROSopt_params) # optmized setup: F2-Score = 0.945

############### MODEL 1C: BAGGING + SYNTHETIC MINORITY OVERSAMPLING TECHNIQUE  ###############

SMOTE_params = dict(sampling_strategy = 'auto', k_neighbors = 5, random_state = seed_custom)
BAG_SMOTE_f2 = balanced_fit(X_train, Y_train, BAG, BAG_params, SMOTE, SMOTE_params) 

### GRID SEARCH OPTIMIZATION

SMOTE_grid = dict(sampling_strategy = [0.7, 0.8, 0.9, 1.0],
                  k_neighbors = [3, 4, 5, 6, 7, 8, 9, 10])

BAG_SMOTE_scoreboard = []

for alpha in SMOTE_grid['sampling_strategy']:
    for neighbors in SMOTE_grid['k_neighbors']:
        SMOTE_params_new = dict(sampling_strategy = alpha, k_neighbors = neighbors, random_state = seed_custom)
        SMOTE_params_new['F2-Score'] = balanced_fit(X_train, Y_train, BAG, BAG_params, SMOTE, SMOTE_params_new)
        BAG_SMOTE_scoreboard.append(SMOTE_params_new)
        
BAG_SMOTE_scoreboard = sorted(BAG_SMOTE_scoreboard, key=lambda x: x['F2-Score'], reverse=True) 

### OPTIMIZED MODEL 1C

SMOTEopt_params = dict(sampling_strategy = 1.0, k_neighbors = 3, random_state = seed_custom) # best 3 candidates have approx. same CV stats => choose simplest configuration.
BAG_SMOTEopt_f2 = balanced_fit(X_train, Y_train, BAG, BAG_params, SMOTE, SMOTEopt_params) # optmized setup: F2-Score = 0.942

############### MODEL 1D: BAGGING + ADAPTIVE SYNTHETIC SAMPLING ###############

ADASYN_params = dict(sampling_strategy = 'auto', n_neighbors = 5, random_state = seed_custom)
BAG_ADASYN_f2 = balanced_fit(X_train, Y_train, BAG, BAG_params, ADASYN, ADASYN_params) 

### GRID SEARCH OPTIMIZATION

ADASYN_grid = dict(sampling_strategy = [0.7, 0.8, 0.9, 1.0],
                  n_neighbors = [3, 4, 5, 6, 7, 8, 9, 10])   

BAG_ADASYN_scoreboard = []

for alpha in ADASYN_grid['sampling_strategy']:
    for neighbors in ADASYN_grid['n_neighbors']:
        ADASYN_params_new = dict(sampling_strategy = alpha, n_neighbors = neighbors, random_state = seed_custom)
        ADASYN_params_new['F2-Score'] = balanced_fit(X_train, Y_train, BAG, BAG_params, ADASYN, ADASYN_params_new)
        BAG_ADASYN_scoreboard.append(ADASYN_params_new)
        
BAG_ADASYN_scoreboard = sorted(BAG_ADASYN_scoreboard, key=lambda x: x['F2-Score'], reverse=True)

### OPTIMIZED MODEL 1D

ADASYNopt_params = dict(sampling_strategy = 0.8, n_neighbors = 5, random_state = seed_custom)
BAG_ADASYN_opt_f2 = balanced_fit(X_train, Y_train, BAG, BAG_params, ADASYN, ADASYNopt_params) # optmized setup: F2-Score = 0.945

############### MODEL 1E: BAGGING + RANDOM UNDERSAMPLING ###############

RUS = RandomUnderSampler
RUS_params = dict(sampling_strategy = 'auto', random_state = seed_custom)

BAG_RUS_f2 = balanced_fit(X_train, Y_train, BAG, BAG_params, RUS, RUS_params)

### GRID SEARCH OPTIMIZATION

RUS_grid = dict(sampling_strategy = [0.7, 0.8, 0.9, 1.0])

BAG_RUS_scoreboard = []

for alpha in RUS_grid['sampling_strategy']:
        RUS_params_new = dict(sampling_strategy = alpha, random_state = seed_custom)
        RUS_params_new['F2-Score'] = balanced_fit(X_train, Y_train, BAG, BAG_params, RUS, RUS_params_new)
        BAG_RUS_scoreboard.append(RUS_params_new)
        
BAG_RUS_scoreboard = sorted(BAG_RUS_scoreboard, key=lambda x: x['F2-Score'], reverse=True)

### OPTIMIZED MODEL 1E

RUSopt_params = dict(sampling_strategy = 1.0, random_state = seed_custom) 
BAG_RUSopt_f2 = balanced_fit(X_train, Y_train, BAG, BAG_params, RUS, RUSopt_params) # optmized setup: F2-Score = 0.942

############### MODEL 1F: BAGGING + EDITED NEAREST NEIGHBOURS ###############

ENN = EditedNearestNeighbours
ENN_params = dict(sampling_strategy = 'auto', n_neighbors = 3, kind_sel = 'all') # note: no random_state argument possible (deterministic method!)

BAG_ENN_f2 = balanced_fit(X_train, Y_train, BAG, BAG_params, ENN, ENN_params) 

# TEST THE EFFECT OF NEIGHBORS ON SAMPLE SIZE REDUCTION

# x_res, y_res = ENN(sampling_strategy = 'auto', n_neighbors = 80, kind_sel = 'all').fit_resample(X_train, Y_train)
# print(sum(y_res)/len(y_res))

### GRID SEARCH OPTIMIZATION

ENN_grid = dict(n_neighbors = np.arange(3, 11, 1)) # n_neighbors = 3 barely removes any majority examples. grid needs to be wider than "typically" implemented for NN-methods.

BAG_ENN_scoreboard = []

for neighbors in ENN_grid['n_neighbors']:
    ENN_params_new = dict(sampling_strategy = 'auto', n_neighbors = neighbors, kind_sel = 'all')
    ENN_params_new['F2-Score'] = balanced_fit(X_train, Y_train, BAG, BAG_params, ENN, ENN_params_new)
    BAG_ENN_scoreboard.append(ENN_params_new)
            
BAG_ENN_scoreboard = sorted(BAG_ENN_scoreboard, key=lambda x: x['F2-Score'], reverse=True)

### OPTIMIZED MODEL 1F

ENNopt_params = dict(sampling_strategy = 'auto', n_neighbors = 4, kind_sel = 'all') 
BAG_ENNopt_f2 = balanced_fit(X_train, Y_train, BAG, BAG_params, ENN, ENNopt_params) # optmized setup: F2-Score = 0.94

############### MODEL 1G: BAGGING + NEIGHBOR CLEANING RULE ###############

NCR = NeighbourhoodCleaningRule
NCR_params = dict(n_neighbors = 3, threshold_cleaning = 0.5, sampling_strategy = 'auto', n_jobs = -1) # note: no random_state argument possible (deterministic method!)

BAG_NCR_f2 = balanced_fit(X_train, Y_train, BAG, BAG_params, NCR, NCR_params) 

### GRID SEARCH OPTIMIZATION

NCR_grid = dict(n_neighbors = np.arange(3, 11, 1)) # threshold for cleaning does not have a strong impact. not necessary to tune.

BAG_NCR_scoreboard = []

for neighbors in NCR_grid['n_neighbors']:
    NCR_params_new = dict(sampling_strategy = 'auto', n_neighbors = neighbors, threshold_cleaning = 0.5, n_jobs = -1)
    NCR_params_new['F2-Score'] = balanced_fit(X_train, Y_train, BAG, BAG_params, NCR, NCR_params_new)
    BAG_NCR_scoreboard.append(NCR_params_new)
            
BAG_NCR_scoreboard = sorted(BAG_NCR_scoreboard, key=lambda x: x['F2-Score'], reverse=True) 

### OPTIMIZED MODEL 1G

NCRopt_params = dict(sampling_strategy = 'auto', n_neighbors = 3, threshold_cleaning = 0.5, n_jobs = -1) # top 2 candidates approx. equal CV stats => choose simpler configuration.
BAG_NCRopt_f2 = balanced_fit(X_train, Y_train, BAG, BAG_params, NCR, NCRopt_params) # optmized setup: F2-Score = 0.946

############### MODEL 2B: EXTREME GRADIENT BOOSTING + RANDOM OVERSAMPLING  ###############

XGB_ROS_f2 = balanced_fit(X_train, Y_train, XGB, XGB_params, ROS, ROS_params) 

### GRID SEARCH OPTIMIZATION

XGB_ROS_scoreboard = []

for alpha in ROS_grid['sampling_strategy']:
        ROS_params_new = dict(sampling_strategy = alpha, random_state = seed_custom)
        ROS_params_new['F2-Score'] = balanced_fit(X_train, Y_train, XGB, XGB_params, ROS, ROS_params_new)
        XGB_ROS_scoreboard.append(ROS_params_new)
        
XGB_ROS_scoreboard = sorted(XGB_ROS_scoreboard, key=lambda x: x['F2-Score'], reverse=True)

### OPTIMIZED MODEL 2B

ROSopt_params2 = dict(sampling_strategy = 1.0, random_state = seed_custom)
XGB_ROSopt_f2 = balanced_fit(X_train, Y_train, XGB, XGB_params, ROS, ROSopt_params2) # optmized setup = default, F2-Score = 0.935

############### MODEL 2C: EXTREME GRADIENT BOOSTING + SYNTHETIC MINORITY OVERSAMPLING TECHNIQUE  ###############

XGB_SMOTE_f2 = balanced_fit(X_train, Y_train, XGB, XGB_params, SMOTE, SMOTE_params) 

### GRID SEARCH OPTIMIZATION

XGB_SMOTE_scoreboard = []

for alpha in SMOTE_grid['sampling_strategy']:
    for neighbors in SMOTE_grid['k_neighbors']:
        SMOTE_params_new = dict(sampling_strategy = alpha, k_neighbors = neighbors, random_state = seed_custom)
        SMOTE_params_new['F2-Score'] = balanced_fit(X_train, Y_train, XGB, XGB_params, SMOTE, SMOTE_params_new)
        XGB_SMOTE_scoreboard.append(SMOTE_params_new)
        
XGB_SMOTE_scoreboard = sorted(XGB_SMOTE_scoreboard, key=lambda x: x['F2-Score'], reverse=True)

### OPTIMIZED MODEL 2C

SMOTEopt_params2 = dict(sampling_strategy = 0.9, k_neighbors = 4, random_state = seed_custom) 
XGB_SMOTEopt_f2 = balanced_fit(X_train, Y_train, XGB, XGB_params, SMOTE, SMOTEopt_params2) # optmized setup: F2-Score = 0.935

############### MODEL 2D: EXTREME GRADIENT BOOSTING + ADAPTIVE SYNTHETIC SAMPLING ###############

XGB_ADASYN_f2 = balanced_fit(X_train, Y_train, XGB, XGB_params, ADASYN, ADASYN_params)

### GRID SEARCH OPTIMIZATION

ADASYN_grid2 = dict(sampling_strategy = [0.7, 0.8, 0.9, 1.0],
                  n_neighbors = [3, 4, 5, 6, 7, 8, 9, 10])   

XGB_ADASYN_scoreboard = []

for alpha in ADASYN_grid2['sampling_strategy']:
    for neighbors in ADASYN_grid2['n_neighbors']:
        ADASYN_params_new = dict(sampling_strategy = alpha, n_neighbors = neighbors, random_state = seed_custom)
        ADASYN_params_new['F2-Score'] = balanced_fit(X_train, Y_train, XGB, XGB_params, ADASYN, ADASYN_params_new)
        XGB_ADASYN_scoreboard.append(ADASYN_params_new)
        
XGB_ADASYN_scoreboard = sorted(XGB_ADASYN_scoreboard, key=lambda x: x['F2-Score'], reverse=True)

### OPTIMIZED MODEL 2D

ADASYNopt_params2 = dict(sampling_strategy = 1.0, n_neighbors = 4, random_state = seed_custom) 
XGB_ADASYN_opt_f2 = balanced_fit(X_train, Y_train, XGB, XGB_params, ADASYN, ADASYNopt_params2) # optmized setup: F2-Score = 0.937

############### MODEL 2E: EXTREME GRADIENT BOOSTING + RANDOM UNDERSAMPLING ###############

XGB_RUS_f2 = balanced_fit(X_train, Y_train, XGB, XGB_params, RUS, RUS_params) 

### GRID SEARCH OPTIMIZATION

XGB_RUS_scoreboard = []

for alpha in RUS_grid['sampling_strategy']:
        RUS_params_new = dict(sampling_strategy = alpha, random_state = seed_custom)
        RUS_params_new['F2-Score'] = balanced_fit(X_train, Y_train, XGB, XGB_params, RUS, RUS_params_new)
        XGB_RUS_scoreboard.append(RUS_params_new)
        
XGB_RUS_scoreboard = sorted(XGB_RUS_scoreboard, key=lambda x: x['F2-Score'], reverse=True)

### OPTIMIZED MODEL 2E

RUSopt_params2 = dict(sampling_strategy = 1.0, random_state = seed_custom)
XGB_RUSopt_f2 = balanced_fit(X_train, Y_train, XGB, XGB_params, RUS, RUSopt_params2) # optmized setup = default => F2-Score = 0.927 

############### MODEL 2F: EXTREME GRADIENT BOOSTING + EDITED NEAREST NEIGHBOURS ###############

XGB_ENN_f2 = balanced_fit(X_train, Y_train, XGB, XGB_params, ENN, ENN_params) 

### GRID SEARCH OPTIMIZATION

XGB_ENN_scoreboard = []

for neighbors in ENN_grid['n_neighbors']:
    ENN_params_new = dict(sampling_strategy = 'auto', n_neighbors = neighbors, kind_sel = 'all')
    ENN_params_new['F2-Score'] = balanced_fit(X_train, Y_train, XGB, XGB_params, ENN, ENN_params_new)
    XGB_ENN_scoreboard.append(ENN_params_new)
            
XGB_ENN_scoreboard = sorted(XGB_ENN_scoreboard, key=lambda x: x['F2-Score'], reverse=True)

### OPTIMIZED MODEL 2F

ENNopt_params2 = dict(sampling_strategy = 'auto', n_neighbors = 5, kind_sel = 'all') 
XGB_ENNopt_f2 = balanced_fit(X_train, Y_train, XGB, XGB_params, ENN, ENNopt_params2) # optmized setup: F2-Score = 0.941 


############### MODEL 2G: EXTREME GRADIENT BOOSTING + NEIGHBOR CLEANING RULE ###############

XGB_NCR_f2 = balanced_fit(X_train, Y_train, XGB, XGB_params, NCR, NCR_params) 

### GRID SEARCH OPTIMIZATION

XGB_NCR_scoreboard = []

for neighbors in NCR_grid['n_neighbors']:
    NCR_params_new = dict(sampling_strategy = 'auto', n_neighbors = neighbors, threshold_cleaning = 0.5, n_jobs = -1)
    NCR_params_new['F2-Score'] = balanced_fit(X_train, Y_train, XGB, XGB_params, NCR, NCR_params_new)
    XGB_NCR_scoreboard.append(NCR_params_new)
            
XGB_NCR_scoreboard = sorted(XGB_NCR_scoreboard, key=lambda x: x['F2-Score'], reverse=True)

### OPTIMIZED MODEL 2G

NCRopt_params2 = dict(sampling_strategy = 'auto', n_neighbors = 3, threshold_cleaning = 0.5, n_jobs = -1) 
XGB_NCRopt_f2 = balanced_fit(X_train, Y_train, XGB, XGB_params, NCR, NCRopt_params2) # optmized setup: F2-Score = 0.93