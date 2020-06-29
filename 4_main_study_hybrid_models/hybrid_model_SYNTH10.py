############### PACKAGES ###############

### CORE PACKAGES

import numpy as np 
import pandas as pd 

### SKLEARN

from sklearn import metrics, preprocessing
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV,  StratifiedKFold

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

X_train = pd.DataFrame(data = X_train_std, columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10'])
X_test = pd.DataFrame(data = X_test_std, columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10'])

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

############### (SEQUENTIAL OR MULTI) RANDOM SEARCH FUNCTION ###############

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
        model = RandomizedSearchCV(method, grid, n_iter = it, cv = cv_KF, n_jobs = -1, scoring = metric, random_state = seed_custom)    
        model.fit(X,Y)
        params_box[i] = model.best_params_
        metrics_box.iloc[i,0] = model.best_score_
            
    return params_box, metrics_box

# def multi_RSCV_robust(method, grid, X, Y, metric, n_candidates, it):
#     """
#     Perform multiple explorations of Random Search and gather the best candidate with the respective parameters and metrics for each round.
#     Number of rounds and number of iterations within each round are free to choose.
#     Good starting point: 3 rounds with each 100 iterations, if results are not similar, expand the numbers and change the grid.
    
    
#     method = classifier object
#     grid = parameter grid settings for the to-be-optimized classifier
#     X = Input (Training Data)
#     Y = Output (Training Data)
#     metric = to-be-optimized metric
#     n_candidates = number of candidates we want = number of iterations we run the random search optimization
#     it = number of iterations/settings to test out of all possibilities from the grid
    
#     *with repeated stratified K-fold CV to avoid spurious results.
#     """
#     params_box = [None] * n_candidates
#     metrics_box = pd.DataFrame(np.zeros((n_candidates, 1)), columns = list(['Score']))
    
#     cv_KF = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = seed_custom)
    
#     for i in range(n_candidates):
#         model = RandomizedSearchCV(method, grid, n_iter = it, cv = cv_KF, n_jobs = -1, scoring = metric, random_state = seed_custom)    
#         model.fit(X,Y)
#         params_box[i] = model.best_params_
#         metrics_box.iloc[i,0] = model.best_score_
            
#     return params_box, metrics_box
    
# too costly to execute..

############### DEFINE BASELINE MODELS (1A & 2A) ###############
    
BAGSVM = BaggingClassifier(base_estimator = SVC(kernel = 'rbf', probability = True, random_state = seed_custom),
                           n_estimators = 50, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed_custom)
BAGSVM_base_performance = algo_CVmetrics(classifier_object = BAGSVM, X_train = X_train, Y_train = Y_train)

## Balanced Accuracy: 0.794 / F2-Score: 0.633


XGBDT = XGBClassifier(booster = 'gbtree', n_estimators = 200, depth_range = 2, min_child_weight = 1, random_state = seed_custom, early_stopping_rounds = 10)
XGBDT_base_performance = algo_CVmetrics(classifier_object = XGBDT, X_train = X_train, Y_train = Y_train)

## Balanced Accuracy: 0.777 / F2-Score: 0.605

## TEST => UNDERBAGGING BETTER W/ DT (ORIGINAL PROPOSAL) OR WITH SVM?
## TEST => EASYENSEMBLE BETTER W/ ADA (ORIGINAL PROPOSAL) OR WITH XGB?

############### HYBRID ALGORITHM 1B: UNDERBAGGING W/ DT ###############
DT = DecisionTreeClassifier(criterion = 'gini', splitter = "best", min_samples_split = 2, max_depth = None, max_features = None, random_state = seed_custom)

UBDT = BalancedBaggingClassifier(base_estimator = DT, max_samples = 1.0, random_state = seed_custom, n_jobs = -1)

classdist_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bag_grid = [10, 20, 30, 40, 50, 100]

f2_metric = make_scorer(metrics.fbeta_score, beta = 2)
ub_grid = dict(sampling_strategy = classdist_grid, n_estimators = bag_grid)

UBDTopt_params_final, UBDTopt_metrics_final = multi_RSCV(method = UBDT, grid = ub_grid, X = X_train, Y = Y_train, metric = f2_metric, n_candidates = 5, it = 50)

UBDT_opt = BalancedBaggingClassifier(base_estimator = DT, n_estimators = 100, sampling_strategy = 0.9, max_samples = 1.0, random_state = seed_custom, n_jobs = -1)
UBDT_performance = algo_CVmetrics(classifier_object = UBDT_opt, X_train = X_train, Y_train = Y_train)

############### HYBRID ALGORITHM 1B: UNDERBAGGING W/ SVM ###############
SVM = SVC(kernel = 'rbf', probability = True, random_state = seed_custom)

UBSVM = BalancedBaggingClassifier(base_estimator = SVM, max_samples = 1.0, random_state = seed_custom, n_jobs = -1)

bag_grid2 = [10, 20, 30, 40, 50]

ub_grid2 = dict(sampling_strategy = classdist_grid, n_estimators = bag_grid2)

UBSVMopt_params_final, UBSVMopt_metrics_final = multi_RSCV(method = UBSVM, grid = ub_grid2, X = X_train, Y = Y_train, metric = f2_metric, n_candidates = 5, it = 50)

UBSVM_opt = BalancedBaggingClassifier(base_estimator = SVM, n_estimators = 20, sampling_strategy = 0.6, max_samples = 1.0, random_state = seed_custom, n_jobs = -1)
UBSVM_performance = algo_CVmetrics(classifier_object = UBSVM_opt, X_train = X_train, Y_train = Y_train)

############### HYBRID ALGORITHM 2A: EASYENSEMBLE W/ ADA ###############
ADA = AdaBoostClassifier(n_estimators = 50, random_state = seed_custom)
ADA_performance = algo_CVmetrics(classifier_object = ADA, X_train = X_train, Y_train = Y_train)

EASYADA = EasyEnsembleClassifier(base_estimator = ADA)

easy_grid = ub_grid2
EASYADAopt_params_final, EASYADAopt_metrics_final = multi_RSCV(method = EASYADA, grid = easy_grid, X = X_train, Y = Y_train, metric = f2_metric, n_candidates = 5, it = 50)

EASYADA_opt = EasyEnsembleClassifier(base_estimator = ADA, sampling_strategy = 0.7, n_estimators = 50)
EASYADA_performance = algo_CVmetrics(classifier_object = EASYADA_opt, X_train = X_train, Y_train = Y_train)

############### HYBRID ALGORITHM 2B: EASYENSEMBLE W/ XGB ###############
XGB = XGBClassifier(booster = 'gbtree', n_estimators = 200, depth_range = 2, min_child_weight = 2, random_state = seed_custom, early_stopping_rounds = 10)

EASYXGB = EasyEnsembleClassifier(base_estimator = XGB)

EASYXGBopt_params_final, EASYXGBopt_metrics_final = multi_RSCV(method = EASYXGB, grid = easy_grid, X = X_train, Y = Y_train, metric = f2_metric, n_candidates = 5, it = 50)

EASYXGB_opt = EasyEnsembleClassifier(base_estimator = XGB, sampling_strategy = 0.5, n_estimators = 20)
EASYXGB_performance = algo_CVmetrics(classifier_object = EASYXGB_opt, X_train = X_train, Y_train = Y_train)

############### CV EVALUATION ###############

def pipeline_singleevaluationCV(X_train, Y_train, pipeline_object):
    """
    Analytics-function for pipeline-opjects that reports F2-Score for imbalanced binary classifcation (Cross Validation)
    pipeline_object = pipeline which combines classifier and sampling method.
    X_train = input (training data)
    Y_train = output (training data)
    """

    RSKCV = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = seed_custom)
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

    RSKCV = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = seed_custom)
    
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

models = []

pip_BAGSVM = Pipeline(steps=[('m',BAGSVM)])
models.append(pip_BAGSVM)
pip_UBDT = Pipeline(steps=[('m',UBDT_opt)])
models.append(pip_UBDT)
pip_UBSVM = Pipeline(steps=[('m',UBSVM_opt)])
models.append(pip_UBSVM)
pip_XGB = Pipeline(steps=[('m',XGBDT)])
models.append(pip_XGB)
pip_EASYADA = Pipeline(steps=[('m',EASYADA_opt)])
models.append(pip_EASYADA)
pip_EASYXGB = Pipeline(steps=[('m',EASYXGB_opt)])
models.append(pip_EASYXGB)

frames = []

for model in models:
	cv_results = pipeline_multievaluationCV(X_train, Y_train, model)
	frames.append(cv_results)
    
table_index = ['BAGSVM','UBDT','UBSVM',
               'XGB','EASYADA','EASYXGB']

cv_table = pd.concat(frames)
cv_table.index = table_index

print(pd.DataFrame.to_latex(cv_table, index = True))  

############### TEST PERFORMANCE ###############

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

frames2 = []

for pipe in models:
    score = predictive_performance(pipe, X_train, Y_train, X_test, Y_test)
    frames2.append(score)

############### FINAL TABLE FOR LATEX ###############

table_index = ['BAGSVM','UBDT','UBSVM',
               'XGB','EASYADA','EASYXGB']

test_table = pd.concat(frames2)
test_table.index = table_index
test_table = np.round(test_table, 3)

print(pd.DataFrame.to_latex(test_table, index = True))  