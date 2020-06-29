############### PACKAGES ###############

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

############### DIRECTORY ###############

cancer_dir = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin'
data_dir = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin/data'
figdir1 = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin/class_densities/'
figdir2 = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin/densities/'
figdir3 = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin/trimmed_class/'
figdir4 = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin/trimmed_dens/'
figdir5 = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin/winsor_class/'
figdir6 = r'/Users/i_pri/Desktop/Thesis/Data/breast_cancer_wisconsin/winsor_dens/'

############### READ DATA ###############

df = pd.read_csv(os.path.join(data_dir, 'breast_cancer_compact.csv'), sep = ',') 
df.head() # 17 continuous/metric features, 1 binary output
df.columns

df['diagnosis'].replace('M', 1 ,inplace=True) # code output variable as binary numbers 0/1 instead of letters B/M
df['diagnosis'].replace('B', 0 ,inplace=True)

Y = df.diagnosis # define output variable    
X = df.drop(['diagnosis'], axis = 1)
X.hist(bins = 40) # different scales of features. 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 123)


############### SKEWNESS & KURTOSIS CHECK ###############

feat_skew = X_train.skew()
feat_kurt = X_train.kurt()

############### DATA TRANSFORMATIONS ###############

# Natural Logarithm (base e)
# Binary Logarithm (base 2)
# Common Logarithm (base 10)
# Square Root
# Cube Root

trafo_names = list(['sqrt', 'cbrt', 'natlog', 'log2', 'log10'])
feat_names = list(X_train.columns)

############### CLASSWISE VISUAL COMPARISON ###############

it = 0

for name in feat_names: 
        
        feat = X_train.iloc[:,it]
        df_temp = pd.concat([feat, Y], axis = 1) 
        sqrt = pd.concat([np.sqrt(feat),Y], axis = 1)
        cbrt = pd.concat([np.cbrt(feat),Y], axis = 1)
        natlog = pd.concat([np.log(feat+1),Y], axis = 1)
        log2 = pd.concat([np.log2(feat+1),Y], axis = 1)
        log10 = pd.concat([np.log10(feat+1),Y], axis = 1)
        
        feat_zero = df_temp[name].loc[df_temp['diagnosis'] == 0]
        feat_one = df_temp[name].loc[df_temp['diagnosis'] == 1]
        
        sqrt_zero = sqrt[name].loc[sqrt['diagnosis'] == 0]
        sqrt_one = sqrt[name].loc[sqrt['diagnosis'] == 1]
        
        cbrt_zero = cbrt[name].loc[cbrt['diagnosis'] == 0]
        cbrt_one = cbrt[name].loc[cbrt['diagnosis'] == 1]
        
        natlog_zero = natlog[name].loc[natlog['diagnosis'] == 0]
        natlog_one = natlog[name].loc[natlog['diagnosis'] == 1]
        
        log2_zero = log2[name].loc[log2['diagnosis'] == 0]
        log2_one = log2[name].loc[log2['diagnosis'] == 1]
        
        log10_zero = log10[name].loc[log10['diagnosis'] == 0]
        log10_one = log10[name].loc[log10['diagnosis'] == 1]
        
        fig = plt.figure(figsize = [20,12])
        
        ax1 = fig.add_subplot(2, 3, 1)
        ax2 = fig.add_subplot(2, 3, 2)
        ax3 = fig.add_subplot(2, 3, 3)
        ax4 = fig.add_subplot(2, 3, 4)
        ax5 = fig.add_subplot(2, 3, 5)
        ax6 = fig.add_subplot(2, 3, 6)
        
        fig.suptitle(name, fontsize = 16)
        
        sns.kdeplot(feat_zero, shade = True, color = "steelblue", ax = ax1, label = "Benign Tumor (No Cancer)").set_title('Original Distribution')
        sns.kdeplot(feat_one, shade = True, color = "indianred", ax = ax1, label = "Malignant Tumor (Cancer)")
        ax1.legend(loc="upper right")
        
        sns.kdeplot(sqrt_zero, shade = True, color = "steelblue", ax = ax2, label = "Benign Tumor (No Cancer)").set_title('Square Root')
        sns.kdeplot(sqrt_one, shade = True, color = "indianred", ax = ax2, label = "Malignant Tumor (Cancer)")
        ax2.legend(loc="upper right")
        
        sns.kdeplot(cbrt_zero, shade = True, color = "steelblue", ax = ax3, label = "Benign Tumor (No Cancer)").set_title('Cube Root')
        sns.kdeplot(cbrt_one, shade = True, color = "indianred", ax = ax3, label = "Malignant Tumor (Cancer)")
        ax3.legend(loc="upper right")
        
        sns.kdeplot(natlog_zero, shade = True, color = "steelblue", ax = ax4, label = "Benign Tumor (No Cancer)").set_title('Natural Logarithm')
        sns.kdeplot(natlog_one, shade = True, color = "indianred", ax = ax4, label = "Malignant Tumor (Cancer)")
        ax4.legend(loc="upper right")
        
        sns.kdeplot(log2_zero, shade = True, color = "steelblue", ax = ax5, label = "Benign Tumor (No Cancer)").set_title('Binary Logarithm')
        sns.kdeplot(log2_one, shade = True, color = "indianred", ax = ax5, label = "Malignant Tumor (Cancer)")
        ax5.legend(loc="upper right")
        
        sns.kdeplot(log10_zero, shade = True, color = "steelblue", ax = ax6, label = "Benign Tumor (No Cancer)").set_title('Common Logarithm')
        sns.kdeplot(log10_one, shade = True, color = "indianred", ax = ax6, label = "Malignant Tumor (Cancer)")
        ax6.legend(loc="upper right")
      
        fig.savefig(figdir1 + feat_names[it] + "_original_new.pdf", bbox_inches='tight', format = 'pdf')
        
        it = it+1

############### UNSEPERATED VISUAL COMPARISON ###############
    
it = 0

for name in feat_names: 
    feat = X_train.iloc[:,it]
    natlog = np.log(feat+1)
    log2 = np.log2(feat+1)
    log10 = np.log10(feat+1)
    sqrt = np.sqrt(feat)
    cbrt = np.cbrt(feat)
    
    fig = plt.figure(figsize = [20,12])
    
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    
    fig.suptitle("Transformed Densities of " + name, fontsize = 16)
    
    sns.kdeplot(feat, shade = True, color = "black", ax = ax1, label = '').set_title('Original Distribution')
    sns.kdeplot(sqrt, shade = True, color = "indianred", ax = ax2, label = '').set_title('Square Root')
    sns.kdeplot(cbrt, shade = True, color = "indianred", ax = ax3, label = '').set_title('Cube Root')
    sns.kdeplot(natlog, shade = True, color = "steelblue", ax = ax4, label = '').set_title('Natural Logarithm')
    sns.kdeplot(log2, shade = True, color = "steelblue", ax = ax5, label = '').set_title('Binary Logarithm')
    sns.kdeplot(log10, shade = True, color = "steelblue", ax = ax6, label = '').set_title('Common Logarithm')
    
    fig.savefig(figdir2 + feat_names[it] + "_original_new.pdf", bbox_inches='tight', format = 'pdf')
    
    it = it+1
    
# ############### PLOTS FOR TRIMMED DATA ###############
    
# ############### READ DATA ###############

# df_trim = pd.read_csv(os.path.join(cancer_dir, 'breast_cancer_compact_trimmed.csv'), sep = ',') 
# Y = df_trim.diagnosis # define output variable    
# X = df_trim.drop(['diagnosis'], axis = 1)

# trafo_names = list(['sqrt', 'cbrt', 'natlog', 'log2', 'log10'])
# feat_names = list(X.columns)

# ############### CLASSWISE VISUAL COMPARISON ###############

# it = 0

# for name in feat_names: 
        
#         feat = X.iloc[:,it]
#         df_temp = pd.concat([feat, Y], axis = 1) 
#         sqrt = pd.concat([np.sqrt(feat),Y], axis = 1)
#         cbrt = pd.concat([np.cbrt(feat),Y], axis = 1)
#         natlog = pd.concat([np.log(feat+1),Y], axis = 1)
#         log2 = pd.concat([np.log2(feat+1),Y], axis = 1)
#         log10 = pd.concat([np.log10(feat+1),Y], axis = 1)
        
#         feat_zero = df_temp[name].loc[df_temp['diagnosis'] == 0]
#         feat_one = df_temp[name].loc[df_temp['diagnosis'] == 1]
        
#         sqrt_zero = sqrt[name].loc[sqrt['diagnosis'] == 0]
#         sqrt_one = sqrt[name].loc[sqrt['diagnosis'] == 1]
        
#         cbrt_zero = cbrt[name].loc[cbrt['diagnosis'] == 0]
#         cbrt_one = cbrt[name].loc[cbrt['diagnosis'] == 1]
        
#         natlog_zero = natlog[name].loc[natlog['diagnosis'] == 0]
#         natlog_one = natlog[name].loc[natlog['diagnosis'] == 1]
        
#         log2_zero = log2[name].loc[log2['diagnosis'] == 0]
#         log2_one = log2[name].loc[log2['diagnosis'] == 1]
        
#         log10_zero = log10[name].loc[log10['diagnosis'] == 0]
#         log10_one = log10[name].loc[log10['diagnosis'] == 1]
        
#         fig = plt.figure(figsize = [20,12])
        
#         ax1 = fig.add_subplot(2, 3, 1)
#         ax2 = fig.add_subplot(2, 3, 2)
#         ax3 = fig.add_subplot(2, 3, 3)
#         ax4 = fig.add_subplot(2, 3, 4)
#         ax5 = fig.add_subplot(2, 3, 5)
#         ax6 = fig.add_subplot(2, 3, 6)
        
#         fig.suptitle(name, fontsize = 16)
        
#         sns.kdeplot(feat_zero, shade = True, color = "steelblue", ax = ax1, label = "Benign Tumor (No Cancer)").set_title('Original Distribution')
#         sns.kdeplot(feat_one, shade = True, color = "indianred", ax = ax1, label = "Malignant Tumor (Cancer)")
#         ax1.legend(loc="upper right")
        
#         sns.kdeplot(sqrt_zero, shade = True, color = "steelblue", ax = ax2, label = "Benign Tumor (No Cancer)").set_title('Square Root')
#         sns.kdeplot(sqrt_one, shade = True, color = "indianred", ax = ax2, label = "Malignant Tumor (Cancer)")
#         ax2.legend(loc="upper right")
        
#         sns.kdeplot(cbrt_zero, shade = True, color = "steelblue", ax = ax3, label = "Benign Tumor (No Cancer)").set_title('Cube Root')
#         sns.kdeplot(cbrt_one, shade = True, color = "indianred", ax = ax3, label = "Malignant Tumor (Cancer)")
#         ax3.legend(loc="upper right")
        
#         sns.kdeplot(natlog_zero, shade = True, color = "steelblue", ax = ax4, label = "Benign Tumor (No Cancer)").set_title('Natural Logarithm')
#         sns.kdeplot(natlog_one, shade = True, color = "indianred", ax = ax4, label = "Malignant Tumor (Cancer)")
#         ax4.legend(loc="upper right")
        
#         sns.kdeplot(log2_zero, shade = True, color = "steelblue", ax = ax5, label = "Benign Tumor (No Cancer)").set_title('Binary Logarithm')
#         sns.kdeplot(log2_one, shade = True, color = "indianred", ax = ax5, label = "Malignant Tumor (Cancer)")
#         ax5.legend(loc="upper right")
        
#         sns.kdeplot(log10_zero, shade = True, color = "steelblue", ax = ax6, label = "Benign Tumor (No Cancer)").set_title('Common Logarithm')
#         sns.kdeplot(log10_one, shade = True, color = "indianred", ax = ax6, label = "Malignant Tumor (Cancer)")
#         ax6.legend(loc="upper right")
      
#         fig.savefig(figdir3 + feat_names[it] + "_trimmed.pdf", bbox_inches='tight', format = 'pdf')
        
#         it = it+1

# ############### UNSEPERATED VISUAL COMPARISON ###############
    
# it = 0

# for name in feat_names: 
#     feat = X.iloc[:,it]
#     natlog = np.log(feat+1)
#     log2 = np.log2(feat+1)
#     log10 = np.log10(feat+1)
#     sqrt = np.sqrt(feat)
#     cbrt = np.cbrt(feat)
    
#     fig = plt.figure(figsize = [20,12])
    
#     ax1 = fig.add_subplot(2, 3, 1)
#     ax2 = fig.add_subplot(2, 3, 2)
#     ax3 = fig.add_subplot(2, 3, 3)
#     ax4 = fig.add_subplot(2, 3, 4)
#     ax5 = fig.add_subplot(2, 3, 5)
#     ax6 = fig.add_subplot(2, 3, 6)
    
#     fig.suptitle("Transformed Densities of " + name, fontsize = 16)
    
#     sns.kdeplot(feat, shade = True, color = "black", ax = ax1, label = '').set_title('Original Distribution')
#     sns.kdeplot(sqrt, shade = True, color = "indianred", ax = ax2, label = '').set_title('Square Root')
#     sns.kdeplot(cbrt, shade = True, color = "indianred", ax = ax3, label = '').set_title('Cube Root')
#     sns.kdeplot(natlog, shade = True, color = "steelblue", ax = ax4, label = '').set_title('Natural Logarithm')
#     sns.kdeplot(log2, shade = True, color = "steelblue", ax = ax5, label = '').set_title('Binary Logarithm')
#     sns.kdeplot(log10, shade = True, color = "steelblue", ax = ax6, label = '').set_title('Common Logarithm')
    
#     fig.savefig(figdir4 + feat_names[it] + "_trimmed.pdf", bbox_inches='tight', format = 'pdf')
    
#     it = it+1

############### PLOTS FOR WINSORIZED DATA ###############
    
############### READ DATA ###############

df_wins = pd.read_csv(os.path.join(data_dir, 'breast_cancer_compact_winsor.csv'), sep = ',') 
Y = df_wins.diagnosis # define output variable    
X = df_wins.drop(['diagnosis'], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 123)

# again: only use training data for visualization!

trafo_names = list(['sqrt', 'cbrt', 'natlog', 'log2', 'log10'])
feat_names = list(X_train.columns)

############### CLASSWISE VISUAL COMPARISON ###############

it = 0

for name in feat_names: 
        
        feat = X_train.iloc[:,it]
        df_temp = pd.concat([feat, Y], axis = 1) 
        sqrt = pd.concat([np.sqrt(feat),Y], axis = 1)
        cbrt = pd.concat([np.cbrt(feat),Y], axis = 1)
        natlog = pd.concat([np.log(feat+1),Y], axis = 1)
        log2 = pd.concat([np.log2(feat+1),Y], axis = 1)
        log10 = pd.concat([np.log10(feat+1),Y], axis = 1)
        
        feat_zero = df_temp[name].loc[df_temp['diagnosis'] == 0]
        feat_one = df_temp[name].loc[df_temp['diagnosis'] == 1]
        
        sqrt_zero = sqrt[name].loc[sqrt['diagnosis'] == 0]
        sqrt_one = sqrt[name].loc[sqrt['diagnosis'] == 1]
        
        cbrt_zero = cbrt[name].loc[cbrt['diagnosis'] == 0]
        cbrt_one = cbrt[name].loc[cbrt['diagnosis'] == 1]
        
        natlog_zero = natlog[name].loc[natlog['diagnosis'] == 0]
        natlog_one = natlog[name].loc[natlog['diagnosis'] == 1]
        
        log2_zero = log2[name].loc[log2['diagnosis'] == 0]
        log2_one = log2[name].loc[log2['diagnosis'] == 1]
        
        log10_zero = log10[name].loc[log10['diagnosis'] == 0]
        log10_one = log10[name].loc[log10['diagnosis'] == 1]
        
        fig = plt.figure(figsize = [20,12])
        
        ax1 = fig.add_subplot(2, 3, 1)
        ax2 = fig.add_subplot(2, 3, 2)
        ax3 = fig.add_subplot(2, 3, 3)
        ax4 = fig.add_subplot(2, 3, 4)
        ax5 = fig.add_subplot(2, 3, 5)
        ax6 = fig.add_subplot(2, 3, 6)
        
        fig.suptitle("Classwise Transformed Densities of " + name + " (From Winsorized Training Data)", fontsize = 16)
        
        sns.kdeplot(feat_zero, shade = True, color = "steelblue", ax = ax1, label = "Benign Tumor (No Cancer)").set_title('Original Distribution')
        sns.kdeplot(feat_one, shade = True, color = "indianred", ax = ax1, label = "Malignant Tumor (Cancer)")
        ax1.legend(loc="upper right")
        
        sns.kdeplot(sqrt_zero, shade = True, color = "steelblue", ax = ax2, label = "Benign Tumor (No Cancer)").set_title('Square Root')
        sns.kdeplot(sqrt_one, shade = True, color = "indianred", ax = ax2, label = "Malignant Tumor (Cancer)")
        ax2.legend(loc="upper right")
        
        sns.kdeplot(cbrt_zero, shade = True, color = "steelblue", ax = ax3, label = "Benign Tumor (No Cancer)").set_title('Cube Root')
        sns.kdeplot(cbrt_one, shade = True, color = "indianred", ax = ax3, label = "Malignant Tumor (Cancer)")
        ax3.legend(loc="upper right")
        
        sns.kdeplot(natlog_zero, shade = True, color = "steelblue", ax = ax4, label = "Benign Tumor (No Cancer)").set_title('Natural Logarithm')
        sns.kdeplot(natlog_one, shade = True, color = "indianred", ax = ax4, label = "Malignant Tumor (Cancer)")
        ax4.legend(loc="upper right")
        
        sns.kdeplot(log2_zero, shade = True, color = "steelblue", ax = ax5, label = "Benign Tumor (No Cancer)").set_title('Binary Logarithm')
        sns.kdeplot(log2_one, shade = True, color = "indianred", ax = ax5, label = "Malignant Tumor (Cancer)")
        ax5.legend(loc="upper right")
        
        sns.kdeplot(log10_zero, shade = True, color = "steelblue", ax = ax6, label = "Benign Tumor (No Cancer)").set_title('Common Logarithm')
        sns.kdeplot(log10_one, shade = True, color = "indianred", ax = ax6, label = "Malignant Tumor (Cancer)")
        ax6.legend(loc="upper right")
      
        fig.savefig(figdir5 + feat_names[it] + "_winsor_new.pdf", bbox_inches='tight', format = 'pdf')
        
        it = it+1

############### UNSEPERATED VISUAL COMPARISON ###############
    
it = 0

for name in feat_names: 
    feat = X_train.iloc[:,it]
    natlog = np.log(feat+1)
    log2 = np.log2(feat+1)
    log10 = np.log10(feat+1)
    sqrt = np.sqrt(feat)
    cbrt = np.cbrt(feat)
    
    fig = plt.figure(figsize = [20,12])
    
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    
    fig.suptitle("Transformed Densities of " + name + " (From Winsorized Training Data)", fontsize = 16)
    
    sns.kdeplot(feat, shade = True, color = "black", ax = ax1, label = '').set_title('Original Distribution')
    sns.kdeplot(sqrt, shade = True, color = "indianred", ax = ax2, label = '').set_title('Square Root')
    sns.kdeplot(cbrt, shade = True, color = "indianred", ax = ax3, label = '').set_title('Cube Root')
    sns.kdeplot(natlog, shade = True, color = "steelblue", ax = ax4, label = '').set_title('Natural Logarithm')
    sns.kdeplot(log2, shade = True, color = "steelblue", ax = ax5, label = '').set_title('Binary Logarithm')
    sns.kdeplot(log10, shade = True, color = "steelblue", ax = ax6, label = '').set_title('Common Logarithm')
    
    fig.savefig(figdir6 + feat_names[it] + "_winsor_new.pdf", bbox_inches='tight', format = 'pdf')
    
    it = it+1