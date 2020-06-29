############### PACKAGES ###############

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import warnings
warnings.filterwarnings("ignore")
#from scipy.stats import mstats # for winsorization
from scipy import stats 
import math
from sklearn.model_selection import train_test_split

############### DIRECTORY ###############

cancer_dir = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin'
data_dir = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin/data'

############### READ DATA ###############
#df = pd.read_csv(os.path.join(cancer_dir, 'breast_cancer_compact.csv'), sep = ',') 
#df = pd.read_csv(os.path.join(cancer_dir, 'breast_cancer_compact_trimmed.csv'), sep = ',') 
df = pd.read_csv(os.path.join(data_dir, 'breast_cancer_compact_winsor.csv'), sep = ',') 

Y = df.diagnosis # define output variable    
X = df.drop(['diagnosis'], axis = 1) # define features

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 123)

############### DATA TRANSFORMATIONS ###############

# Natural Logarithm (base e)
# Binary Logarithm (base 2)
# Common Logarithm (base 10)
# Square Root
# Cube Root

############### BOX-COX TRANSFORMATION ###############

# X_boxcox = pd.DataFrame(np.zeros((X.shape[1], X.shape[0])))
# lambdas = pd.DataFrame(np.zeros((X.shape[1], 3)), columns = ['lambda_opt', 'lb', 'ub'])


# for i in range(X.shape[1]):
#     feat = X.iloc[:,i]
#     feat_tf, lambda_opt, lambda_ci = stats.boxcox(feat + 1, lmbda = None, alpha = 0.05)
#     X_boxcox.iloc[:,i] = pd.Series(feat_tf)
#     lb, ub = lambda_ci
#     lambdas.iloc[i,0] = lambda_opt
#     lambdas.iloc[i,1] = lb
#     lambdas.iloc[i,2] = ub

# optimal lambdas imply what a suitable transformation would be to get an approximate normal distribution.
# however the method is intended for already close to normal distributions and transformations only make sense if lambda values are low (-5,+5).
# due to very high lambda values, doesn't seem to make sense to apply this in our toolset. 
# apparently the boxcox transformation is sensitive to outliers and therefore not suitable for our dataset due to generally having high skewness and kurtosis values. 

############### MOMENTS AND NORMALITY TESTS ###############

trafo_names = list(['feat','natlog', 'log2', 'log10', 'sqrt', 'cbrt'])
kurtstats = pd.DataFrame(np.zeros((len(trafo_names), X_train.shape[1])), columns = X_train.columns, index = trafo_names) # kurtosis for all feature transformations
skewstats = pd.DataFrame(np.zeros((len(trafo_names), X_train.shape[1])), columns = X_train.columns, index = trafo_names) # skewness for all feature transformations
stdstats = pd.DataFrame(np.zeros((len(trafo_names), X_train.shape[1])), columns = X_train.columns, index = trafo_names) # standard deviation for all feature transformations

SWstats = pd.DataFrame(np.zeros((len(trafo_names), X_train.shape[1])), columns = X_train.columns, index = trafo_names) # Shapiro Wilk statistic for all feature transformations
JBstats = pd.DataFrame(np.zeros((len(trafo_names), X_train.shape[1])), columns = X_train.columns, index = trafo_names) # Jarque Bera for all feature transformations
ADstats = pd.DataFrame(np.zeros((len(trafo_names), X_train.shape[1])), columns = X_train.columns, index = trafo_names) # Anderson Darling for all feature transformations
KSstats = pd.DataFrame(np.zeros((len(trafo_names), X_train.shape[1])), columns = X_train.columns, index = trafo_names) # Kolmogorov Smirnov feature transformations

# again: only use training data for the tests!

for i in range(X_train.shape[1]):
    feat = X_train.iloc[:,i]
    natlog = np.log(feat+1)
    log2 = np.log2(feat+1)
    log10 = np.log10(feat+1)
    sqrt = np.sqrt(feat)
    cbrt = np.cbrt(feat)
    
    df_list = list([feat, natlog, log2, log10, sqrt, cbrt])
  
    for j, trafo in zip(range(len(trafo_names)), df_list):
        kurtstats.iloc[j,i] = trafo.kurt()
        skewstats.iloc[j,i] = trafo.skew()
        stdstats.iloc[j,i] = trafo.std()
        
        stat1, pval1 = stats.shapiro(trafo)
        stat2, pval2 = stats.jarque_bera(trafo)
        AD, crit, sig = stats.anderson(trafo, dist = 'norm') 
        stat4, pval4 = stats.kstest(trafo, 'norm')
        
        AD_adj = AD*(1 + (.75/50) + 2.25/(50**2))

        if AD_adj >= .6:
            pval3 = math.exp(1.2937 - 5.709*AD_adj - .0186*(AD_adj**2))
        elif AD_adj >=.34:
            pval3 = math.exp(.9177 - 4.279*AD_adj - 1.38*(AD_adj**2))
        elif AD_adj >.2:
            pval3 = 1 - math.exp(-8.318 + 42.796*AD_adj - 59.938*(AD_adj**2))
        else:
            pval3 = 1 - math.exp(-13.436 + 101.14*AD_adj - 223.73*(AD_adj**2))
        
        SWstats.iloc[j,i] = pval1
        JBstats.iloc[j,i] = pval2
        ADstats.iloc[j,i] = pval3
        KSstats.iloc[j,i] = pval4
        
# the tables of moments and normality tests should help to gauge which transformation comes the closest to an approx. normal distribution.
# many tests are rejected, yet, the statistics are a good indicator what transformation might still yield a small improvement combined with visual analysis. 