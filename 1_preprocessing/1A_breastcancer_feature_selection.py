############### DATA PREPROCESSING ###############


############### PACKAGES ###############

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import stats 
import os

import warnings
warnings.filterwarnings("ignore")

############### DIRECTORY ###############

cancer_dir = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin'
data_dir = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin/data'

############### READ DATA ###############

df = pd.read_csv(os.path.join(cancer_dir, 'data.csv'), sep = ',') 
df.head()
df.columns

df['diagnosis'].replace('M', 1 ,inplace=True) # code output variable as binary variable with numerical values 0/1 instead of letters B/M
df['diagnosis'].replace('B', 0 ,inplace=True)

     
list1 = ['Unnamed: 32','id','diagnosis'] 

X = df.drop(list1,axis = 1) # drop 2 irrelevant variables and the output + define feature variables
Y = df.diagnosis # define output variable       

df_temp = pd.concat([X,Y], axis = 1)

df_temp.hist(bins = 25)

df_temp.to_csv(os.path.join(data_dir, 'breast_cancer_raw.csv'), index = False) # store compact dataset

############### SPLIT DATA (TRAINING/TEST) ###############

df = pd.read_csv(os.path.join(data_dir, 'breast_cancer_raw.csv'), sep = ',') 

X = df.drop(['diagnosis'], axis = 1) # define input/features
Y = df['diagnosis'] # define output 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 123)

############### CLASS DISTRIBUTION ###############

ax = sns.countplot(Y_train,label="Count") # B = 284, M = 171
B, M = Y_train.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)

# majority = approx. 62%, minority = approx. 38%

X_train.describe()

############### CORRELATION MATRIX 1 (ORIGINAL DATASET) ###############

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X_train.corr(), annot=False, linewidths=.5, fmt= '.1f',ax=ax)

# input features exhibit strong multicollinearity
# drop features with corr >= 0.9

list2 = ['perimeter_mean', 'area_mean', 'radius_worst', 'perimeter_worst', 'area_worst', 'texture_worst', 'compactness_worst', 
         'concavity_worst', 'concave points_worst', 'concavity_mean', 'perimeter_se', 'area_se']

X_train = X_train.drop(list2, axis = 1)

# check again

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X_train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

# still 6 more features with corr = 0.8 
# perform one way anova first to drop irrelevant features and then drop remaining strongly correlated "duplicate" features.

############### ONE-WAY ANOVA TESTS ###############
        
df_train = pd.concat([X_train, Y_train], axis = 1)        
        
df_B = df_train.loc[df_temp['diagnosis'] == 0]
df_M = df_train.loc[df_temp['diagnosis'] == 1]     

featnames = X_train.columns
        
anova_stats = pd.DataFrame(np.zeros((X_train.shape[1], 1)), columns = ['pvalue'], index = featnames)

for i in range(X_train.shape[1]):
    group_B = df_B.iloc[:,i]
    group_M = df_M.iloc[:,i]
    
    fval, pval = stats.f_oneway(group_B, group_M)
    
    anova_stats.iloc[i,0] = pval

# features with strongly overlapping group-densities, implying bad feature quality to consider in the model. 

# fractal_dimension_mean
# texture_se
# symmetry_se
# smoothness_se
# fractal_dimension_se

# kick features with pval > 0.05. reduced the dataset further to 13 features. 
    
list3 = ['fractal_dimension_mean', 'texture_se', 'symmetry_se', 'smoothness_se', 'fractal_dimension_se']

X_train = X_train.drop(list3, axis = 1)

# check again

f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X_train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

############### IF YOU WANT TO DROP VARIABLES WITH CORR = 0.8 AS WELL ###############

# drop out features iteratively that correlate highly (=0.8) with other variables.

# start with compactness_se

X_train = X_train.drop(['compactness_se'], axis = 1) 
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X_train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

# next concave points_mean 

X_train = X_train.drop(['concave points_mean'], axis = 1) 
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X_train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

# next concave_points_se

X_train = X_train.drop(['concave points_se'], axis = 1) 
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X_train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

# next smoothness_worst

X_train = X_train.drop(['smoothness_worst'], axis = 1) 
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(X_train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

# dataset remains with 9 variables left. 

############### SAVE NEW DATASET ###############

list_new = X_train.columns

df_new = df[list_new]

df_new.to_csv(os.path.join(data_dir, 'breast_cancer_compact.csv'), index = False) # store compact dataset

############### READ DATA ###############

df = pd.read_csv(os.path.join(data_dir, 'breast_cancer_compact.csv'), sep = ',') 

Y = df.diagnosis # define output variable    
X = df.drop(['diagnosis'], axis = 1)

############### TRIM THE DATASET BY +- 3 SIGMA ###############

# mean_df = X.mean()
# sd_df = X.std()

# X_trim = X

# for i in range(X.shape[1]):
#     mean = mean_df[i]
#     sd = sd_df[i]
    
#     X_trim = X_trim[X_trim.iloc[:,i] <= (mean + 3*sd)]
    
#     X_trim = X_trim[X_trim.iloc[:,i] >= (mean - 3*sd)]
    
# # removed approx. 65 obs = 12% of data
    
# df_trim = pd.DataFrame.dropna(pd.concat([X_trim,Y], axis = 1))

# Y_trim = df_trim['diagnosis']

############### CLASS DISTRIBUTION ###############

# ax = sns.countplot(Y_trim,label="Count") # M = 212, B = 357
# B, M = Y_trim.value_counts()
# print('Number of Benign: ',B) # 332
# print('Number of Malignant : ',M) # 173

# # majority = 2/3, minority = 1/3

# X.describe()

# new_ind = pd.Series(range(0,df_trim.shape[0])) # adjust indices
# df_trim.index = new_ind

# ############### SAVE DATASET ###############

# df_trim.to_csv(os.path.join(cancer_dir, 'breast_cancer_compact_trimmed.csv'), index = False) # store trimmed dataset

############### WINSORIZE THE (FULL) DATASET BY +- 3 SIGMA ###############


# ############### READ DATA ###############

# df = pd.read_csv(os.path.join(data_dir, 'breast_cancer_compact.csv'), sep = ',') 
# Y = df.diagnosis # define output variable    
# X = df.drop(['diagnosis'], axis = 1)

# X_wins = X

# for i in range(X_wins.shape[1]):
#     feat = X_wins.iloc[:,i]
#     mu = feat.mean()
#     sigma = feat.std()

#     lb = mu - 3*sigma # lower bound / cut-off value
#     ub = mu + 3*sigma # upper bound / cut-off value
    
#     X_wins.iloc[:,i] = feat.clip(lb, ub)
    
# df_wins = pd.concat([X_wins, Y], axis = 1) # merge features and output back together to a df
# X_wins.hist(bins = 25) # check histogram changes

# ############### SAVE DATASET ###############

# df_wins.to_csv(os.path.join(data_dir, 'breast_cancer_compact_winsor.csv'), index = False) # store winsorized dataset

############### WINSORIZE THE (SPLITTED TRAIN/TEST) DATASET BY +- 3 SIGMA ###############


############### READ DATA ###############

df = pd.read_csv(os.path.join(data_dir, 'breast_cancer_compact.csv'), sep = ',') 
Y = df.diagnosis # define output variable    
X = df.drop(['diagnosis'], axis = 1)

############### SPLIT DATA ###############

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 123)

mean_train = X_train.mean()
sd_train = X_train.std()

X_wins = X

# important: measure criteria on train dataset, and then apply the winsorization to test and train. 

for i in range(X_wins.shape[1]):
    feat = X_wins.iloc[:,i]
    mu = mean_train.iloc[i]
    sigma = sd_train.iloc[i]

    lb = mu - 3*sigma # lower bound / cut-off value
    ub = mu + 3*sigma # upper bound / cut-off value
    
    X_wins.iloc[:,i] = feat.clip(lb, ub)
    
df_wins = pd.concat([X_wins, Y], axis = 1) # merge features and output back together to a df
X_wins.hist(bins = 25) # check histogram changes

# miniscule changes in threshold values if computed from train set instead of the entire dataset..

############### SAVE DATASET ###############

df_wins.to_csv(os.path.join(data_dir, 'breast_cancer_compact_winsor.csv'), index = False) # store winsorized dataset

############### READ DATA ###############

df = pd.read_csv(os.path.join(cancer_dir, 'breast_cancer_compact_winsor.csv'), sep = ',') 

Y = df.diagnosis # define output variable    
X = df.drop(['diagnosis'], axis = 1)

############### DATA TRANSFORMATION ###############

X_trafo = X

X_trafo.iloc[:,0] = np.log(X_trafo.iloc[:,0] + 1) # radius_mean, [log, log, log], trimming & winsorization doesn't change much, normality tests not passed but looks decent.. => take log
X_trafo.iloc[:,1] = np.log(X_trafo.iloc[:,1] + 1) # texture_mean, [log, log > cbrt, log > cbrt], winsorization > trimming (surprisingly), densities are close to normal!! => take log
X_trafo.iloc[:,2] = np.cbrt(X_trafo.iloc[:,2]) # smoothness_mean, [cbrt > sqrt, cbrt > sqrt, cbrt > sqrt], trimming & winsorization helps slightly, densities look close to normal! 
                                               # passes normality tests well! => take cube root (visually doesn't look significantly different but tests are strongly suggesting it)
X_trafo.iloc[:,3] = np.cbrt(X_trafo.iloc[:,3]) # compactness_mean, [cbrt, cbrt, cbrt] - original & trimmed better => take cube root
X_trafo.iloc[:,4] = np.sqrt(X_trafo.iloc[:,4]) # concave points_mean, [sqrt,sqrt,sqrt] - original & trimmed better => take square root but fails normality tests
X_trafo.iloc[:,5] = np.cbrt(X_trafo.iloc[:,5]) # symmetry_mean, [cbrt, cbrt, cbrt > sqrt] - trimming helps a lot!!, winsorization is good but stacks on right tail end.. 
                                               # => take cube root (visually doesn't look significantly different but tests are strongly suggesting it)
                                               # fractal_dimension_mean [-, cbrt, cbrt ] - trimming helps a lot!!, winsorization is good but stacks on right tail end.. 
                                               # possible to apply cbrt but does not change much.. => skip trafo, keep original density with winsorization/trimming

X_trafo.iloc[:,7] = np.cbrt(X_trafo.iloc[:,7]) # radius_se, [cbrt, cbrt, cbrt] - winsorization/trimming helps! final density looks good but fails normality tests => take cube root
X_trafo.iloc[:,8] = np.cbrt(X_trafo.iloc[:,8]) # texture_se, [(cbrt), (cbrt), cbrt ] - winsorization helps but trimming is better! normality tests only passed in trimmed version. 
                                               # final density looks good! => take cube root
X_trafo.iloc[:,9] = np.cbrt(X_trafo.iloc[:,9]) # smoothness_se, [(cbrt), (cbrt), (cbrt) ] - winsorization/trimming helps, winsorization leads to fat right tail ends. 
                                               # normality tests not passed but density looks okay. => take cbrt 
X_trafo.iloc[:,10] = np.cbrt(X_trafo.iloc[:,10]) # compactness_se, [cbrt, cbrt, cbrt]  - original & trimmed => take cube root (square root also possible)
X_trafo.iloc[:,11] = np.sqrt(X_trafo.iloc[:,11]) # concavity_se, [sqrt but inconclusive, sqrt, sqrt] - trimmed looks better and yields better stats - needs special consideration!! => take square root
X_trafo.iloc[:,12] = np.log(X_trafo.iloc[:,12] + 1) # concave points_se, [sqrt/log but inconclusive, - , nat_log ] - trimming helps!! - needs special consideration!!
X_trafo.iloc[:,13] = np.cbrt(X_trafo.iloc[:,13]) # symmetry_se, [cbrt, cbrt, cbrt, KS hints to use log10] - winsorizing & trimming helps!! but winsorizing stacks observations on the right end of the tail.. 
X_trafo.iloc[:,14] = np.cbrt(X_trafo.iloc[:,14]) # fractal_dimension_se, [cbrt/log but inconclusive = needs trimming/winsorization, cbrt, cbrt] - trimmed + cbrt is a huge improvement!! final density looks okayish

X_trafo.iloc[:,15] = np.cbrt(X_trafo.iloc[:,15]) # smoothness_worst, [sqrt/cbrt, sqrt/cbrt, sqrt/cbrt, after trimming most trafos pass normality test] 
                                                 # winsorization & trimming yields improvement! final density looks fine => take cube root (but no strong visual changes)
X_trafo.iloc[:,16] = np.cbrt(X_trafo.iloc[:,16]) # symmetry_worst, [sqrt/cbrt, cbrt but inconclusive, sqrt/cbrt] - winsorization & trimming helps but winsorization stacks obs 
                                                 # on the end of the right tail.. final density looks acceptable => take cube root
X_trafo.iloc[:,17] = np.cbrt(X_trafo.iloc[:,17]) # fractal_dimension_worst, [cbrt, cbrt, cbrt] - winsorization & trimming helps, final density looks acceptable but normality tests don't look good.. => take cube root

# In general, KS-test rejects normality more often than the other tests! and seems perceive log trafos as more normal than root trafos..
# All other three methods mostly hint to use a root trafo (for the given the dataset ofc)
# 3 log trafos, 2 sqrt trafo, rest cbrt trafos

############### SAVE DATASET ###############

df_trafo = pd.concat([X_trafo,Y], axis = 1)

df_trafo.to_csv(os.path.join(cancer_dir, 'breast_cancer_winsor_trafo.csv'), index = False) # store trimmed dataset

############### READ DATA ###############

#df = pd.read_csv(os.path.join(cancer_dir, 'breast_cancer_winsor_trafo.csv'), sep = ',') 
df = pd.read_csv(os.path.join(cancer_dir, 'breast_cancer_compact.csv'), sep = ',') 

Y = df.diagnosis # define output variable    
X = df.drop(['diagnosis'], axis = 1)

############### FEATURE DENSITY PLOTS ###############

fig = plt.figure(figsize = [8,8])

ax1 = fig.add_subplot(5, 4, 1)
ax2 = fig.add_subplot(5, 4, 2)
ax3 = fig.add_subplot(5, 4, 3)
ax4 = fig.add_subplot(5, 4, 4)
ax5 = fig.add_subplot(5, 4, 5)
ax6 = fig.add_subplot(5, 4, 6)
ax7 = fig.add_subplot(5, 4, 7)
ax8 = fig.add_subplot(5, 4, 8)
ax9 = fig.add_subplot(5, 4, 9)
ax10 = fig.add_subplot(5, 4, 10)
ax11 = fig.add_subplot(5, 4, 11)
ax12 = fig.add_subplot(5, 4, 12)
ax13 = fig.add_subplot(5, 4, 13)
ax14 = fig.add_subplot(5, 4, 14)
ax15 = fig.add_subplot(5, 4, 15)
ax16 = fig.add_subplot(5, 4, 16)
ax17 = fig.add_subplot(5, 4, 17)
ax18 = fig.add_subplot(5, 4, 18)

axis_list = list([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18])
             

feat_names = X.columns

fig.suptitle("Transformed Densities", fontsize = 16)

it = 0

    
for name, ax_ind in zip(feat_names, axis_list): 
    sns.kdeplot(X.iloc[:,it], shade = True, color = "steelblue", ax = ax_ind, label = '').set_title(name)
    it = it+1

fig.tight_layout()
    
fig.savefig(cancer_dir + feat_names[it] + "_original.pdf", bbox_inches='tight', format = 'pdf')