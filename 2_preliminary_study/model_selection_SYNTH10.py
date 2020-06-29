############### PACKAGES ###############

### CORE PACKAGES

import numpy as np 
import pandas as pd 
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

### MACHINE LEARNING METHODS

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from xgboost import XGBClassifier # needs: pip install xgboost 

from sklearn import model_selection


### ADDITIONAL

import os 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.metrics import make_scorer
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import plot_precision_recall_curve
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from hellinger_distance_criterion import HellingerDistanceCriterion 
import matplotlib.pyplot as plt

## Note for hellinger_distance_criterion

# needs manual installation of sklearn from github
# make sure to have microsoft visual studio c++ installed and updated setuptools via pip install
# files can be downloaded from dubov's github: https://github.com/EvgeniDubov/hellinger-distance-criterion
# follow the instructions on the github-page.

############### DIRECTORY ###############

data_dir = r'/Users/i_pri/Desktop/Thesis/Data/synthetic_data'

############### LOAD DATASET ###############

df = pd.read_csv(os.path.join(data_dir, 'syntheticdata_10to1.csv'), sep = ',') 

X = df.drop(['C'], axis = 1)
Y = df.C

############### SPLIT THE DATA ###############
seed = 123

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = seed, stratify = Y)

# argument 'stratify' keeps the class distributions equal across training and test data 
# split dataset into training and test data

classprop_train = sum(Y_train)/len(Y_train)
classprop_test = sum(Y_test)/len(Y_test)

# approximately the same class-distribution (seems like a "significant" change due to small sample size in general)

############### PERFORM STANDARDIZATION ###############

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std  = scaler.transform(X_test)

X_df = pd.DataFrame(data = X_train_std)

############### HISTOGRAM ###############

# X_df.hist(bins = 25)

############### CORRELATION PLOT ###############

# f,ax = plt.subplots(figsize=(18, 18))
# sns.heatmap(X_df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

########################################
############### PRESTUDY ###############
########################################

X_train = X_train_std
X_test = X_test_std
    
############### ANALYTICS FUNCTION ###############

def algo_CVmetrics(classifier_object, X_train, Y_train):
    """
    Analytics-function that reports performance metrics for imbalanced binary classifcation (Cross Validation)
    classifier object = classification method e.g. DecisionTreeClassifier()
    X_train = input (training data)
    Y_train = output (training data)
    """
    
    cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = seed)
    
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
    
############### METHODOLOGICAL RESEARCH ###############

### Support Vector Machines 

# SVM_poly2 = SVC(kernel = 'poly', degree = 2)
# SVM_poly2_metrics = algo_CVmetrics(classifier_object = SVM_poly2, X_train = X_train, Y_train = Y_train)

# SVM_poly3 = SVC(kernel = 'poly', degree = 3)
# SVM_poly3_metrics = algo_CVmetrics(classifier_object = SVM_poly3, X_train = X_train, Y_train = Y_train)

SVM_rbf   = SVC(kernel = 'rbf', probability = True, random_state = seed)
SVM_rbf_metrics = algo_CVmetrics(classifier_object = SVM_rbf, X_train = X_train, Y_train = Y_train)

# polynomial kernels yield considerably lower F2-Scores and other metrics in general.
# seems to make sense as literature points out the strength of the radial basis kernel function for many cases.

### Decision Tree (Gini Criterion)

DT = DecisionTreeClassifier(criterion = 'gini', splitter = "best", min_samples_split = 2, max_depth = None, max_features = None, random_state = seed)
DT_metrics = algo_CVmetrics(classifier_object = DT, X_train = X_train, Y_train = Y_train)

### Hellinger Distance Decision Tree

hdc = HellingerDistanceCriterion(1, np.array([2],dtype='int64')) # define hellinger distance criterion  

HDDT = DecisionTreeClassifier(criterion = hdc, splitter = "best", min_samples_split = 2, max_depth = None, max_features = None, random_state = seed)
HDDT_metrics = algo_CVmetrics(classifier_object = HDDT, X_train = X_train, Y_train = Y_train)

# hellinger distance is less sensitive as a criterion to class imbalance.
# strength of hellinger distance does not show that much. tradeoff between precision/recall. (higher precision but lower recall.)

##### ENSEMBLE ALGOS

### Random Forest (Gini Criterion)

RF = RandomForestClassifier(criterion = 'gini', min_samples_split = 2, max_depth = None, max_features = 'sqrt', n_estimators = 200, random_state = seed)
RF_metrics = algo_CVmetrics(classifier_object = RF, X_train = X_train, Y_train = Y_train)

### Hellinger Distance Random Forest

HDRF = RandomForestClassifier(criterion = hdc, min_samples_split = 2, max_depth = None, max_features = None, n_estimators = 200, random_state = seed)
HDRF_metrics = algo_CVmetrics(classifier_object = HDRF, X_train = X_train, Y_train = Y_train)

# similar performance compared to gini-based random forest. 
# again precision/recall trade off (higher precision, lower recall) but tradeoff is smaller compared to single HDDT.

### Bagging (w Decision Trees)

BAGDT = BaggingClassifier(base_estimator = DT, n_estimators = 50, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed)
BAGDT_metrics = algo_CVmetrics(classifier_object = BAGDT, X_train = X_train, Y_train = Y_train)

# marginal improvements for 50 bags vs. 10 bags.
# slightly worse than gini-based RF with precision/recall tradeoff.

### Bagging (w HDDT)

# BAGHDDT = BaggingClassifier(base_estimator = HDDT, n_estimators = 25, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed)
# BAGHDDT_metrics = algo_CVmetrics(classifier_object = BAGHDDT, X_train = X_train, Y_train = Y_train)

# slightly worse than gini-based RF with precision/recall tradeoff.
# seems to have a similar tradeoff like HDRF.

### Adaptive Boosting (w Decision Trees)

ADADT = AdaBoostClassifier(base_estimator = DT, n_estimators = 10, random_state = seed)
ADADT_metrics = algo_CVmetrics(classifier_object = ADADT, X_train = X_train, Y_train = Y_train)

# too many boosting iterations lead to overfitting and poor CV validation performance.
# seems better than random forests when inspecting the f2-score, however only due to the larger weights on recall.

### Adaptive Boosting (w HDDT)

# ADAHDDT = AdaBoostClassifier(base_estimator = HDDT, n_estimators = 10, random_state = seed)
# ADADT_metrics = algo_CVmetrics(classifier_object = ADAHDDT, X_train = X_train, Y_train = Y_train)
# worse than stand-alone-HDDT

### Extreme Gradient Boosting (w Decision Trees)

XGB = XGBClassifier(booster = 'gbtree', n_estimators = 300, random_state = seed)
XGB_metrics = algo_CVmetrics(classifier_object = XGB, X_train = X_train, Y_train = Y_train)

# good performance

### Adaptive Boosting (w Support Vector Machines)

ADASVM = AdaBoostClassifier(base_estimator = SVM_rbf, n_estimators = 5, random_state = seed)
ADASVM_metrics = algo_CVmetrics(classifier_object = ADASVM, X_train = X_train, Y_train = Y_train)

# poor performance. adaptive boosting w SVM as weak learners does not seem to make sense.
# also computationally intensive to compute. 

### Bagging (w Support Vector Machines)

BAGSVM = BaggingClassifier(base_estimator = SVM_rbf, n_estimators = 200, max_samples = 0.8, bootstrap = True, n_jobs = -1, random_state = seed)
BAGSVM_metrics = algo_CVmetrics(classifier_object = BAGSVM, X_train = X_train, Y_train = Y_train)

# good performance, outperforms bagged trees by a large margin!

############### SUMMARIZE RESULTS ###############

############### DEFINE CLASSIFIERS ###############

classifiers = {
    "Hellinger Distance Decision Tree": HDDT,
    "Random Forest": RF,
    "Hellinger Distance Random Forest": HDRF,
    "Adaptive Boosting (Decision Trees)": ADADT,
    "Adaptive Boost (SVMs)": ADASVM,
    "Bagging (Decision Trees)": BAGDT,
    "Bagging (SVMs)": BAGSVM,
    "Extreme Gradient Boosting (Decision Trees)": XGB
    }

classifiernames = list(['HDDT', 'RF', 'HDRF', 'ADADT', 'ADASVM', 'BAGDT', 'BAGSVM', 'XGB'])

############### CROSS VALIDATION RESULTS ( ###############

cv = RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = seed)

frames = []

i = 0
for key, classifier in classifiers.items():
    
    cv_results   = algo_CVmetrics(classifier, X_train, Y_train)
    frames.append(cv_results)
    i = i+1
    
cv_df = pd.concat(frames)
cv_df.index = classifiernames
    
############### TRAINING AND TEST PERFORMANCE ###############
    
scorenames_small = list(['F2-Score', 'Balanced Accuracy', 'Precision','Recall'])

train_df = pd.DataFrame(np.zeros((len(classifiers), len(scorenames_small))), columns = scorenames_small, index = classifiernames)
test_df = pd.DataFrame(np.zeros((len(classifiers), len(scorenames_small))), columns = scorenames_small, index = classifiernames)
          
i = 0
for key, classifier in classifiers.items():
    model = classifier.fit(X_train, Y_train)
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)
    
    train_df.iloc[i,0] = round(metrics.fbeta_score(Y_train, Y_pred_train, beta = 2), 4)
    train_df.iloc[i,1] = round(metrics.balanced_accuracy_score(Y_train, Y_pred_train), 4)
    train_df.iloc[i,2] = round(metrics.precision_score(Y_train, Y_pred_train), 4)
    train_df.iloc[i,3] = round(metrics.recall_score(Y_train, Y_pred_train), 4)

    test_df.iloc[i,0] = round(metrics.fbeta_score(Y_test, Y_pred_test, beta = 2), 4)
    test_df.iloc[i,1] = round(metrics.balanced_accuracy_score(Y_test, Y_pred_test), 4)
    test_df.iloc[i,2] = round(metrics.precision_score(Y_test, Y_pred_test), 4)
    test_df.iloc[i,3] = round(metrics.recall_score(Y_test, Y_pred_test), 4)

    i = i+1
    
# print(pd.DataFrame.to_latex(cv_df, index = True))  
# print(pd.DataFrame.to_latex(test_df, index = True))  
# print(pd.DataFrame.to_latex(train_df, index = True))
    
############### BOXPLOT ###############
    
## DEFINE MODELS
    
models = []
models.append(('HDDT', HDDT))
models.append(('RF', RF))
models.append(('HDRF', HDRF))
models.append(('ADADT', ADADT))
models.append(('ADASVM', ADASVM))
models.append(('BAGDT', BAGDT))
models.append(('BAGSVM', BAGSVM))
models.append(('XGB', XGB))

# COMPUTE RESULTS
results = []
names = []
scoring = make_scorer(metrics.fbeta_score, beta = 2)
for name, model in models:
	repkfold = model_selection.RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3, random_state = seed)
	cv_results_new = model_selection.cross_val_score(model, X_train, Y_train, cv=repkfold, scoring=scoring)
	results.append(cv_results_new)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results_new.mean(), cv_results_new.std())
	print(msg)
    
# BOXPLOT
fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(results, showmeans = True)
ax.set_xticklabels(names)
plt.show()

############### PLOT PRECISION RECALL CURVE ###############

# fig = plt.figure(figsize = [10,10])
# ax1 = fig.add_subplot(1,1,1)

# i = 0
# for key, classifier in classifiers.items():
    
#     plot_precision_recall_curve(classifier, X_test, Y_test, ax = ax1)
#     i = i+1