############### PACKAGES ###############

import pandas as pd 
from sklearn.datasets import make_classification 
import os 

############### DIRECTORY ###############

data_dir = r'/Users/i_pri/Desktop/Thesis/Data/synthetic_data'

############### CREATE ARTIFICIAL DATA FOR BINARY CLASSIFICATION, IMBALANCE = 10:1 ###############

seed = 123

class_weight = list([0.90])

X, Y = make_classification(n_samples = 1000, n_classes = 2, n_clusters_per_class = 2, weights = class_weight,
                         n_features = 10, n_informative = 8, n_redundant = 2, flip_y = 0.02, class_sep = 1, shuffle = True, random_state = seed)

df_synth1 = pd.concat([pd.DataFrame(X), pd.DataFrame(Y)], axis = 1) # merge features and output back together to a df
df_synth1.columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'C']

############### SAVE DATASET ###############

df_synth1.to_csv(os.path.join(data_dir, 'syntheticdata_10to1.csv'), index = False) 

# df_temp = pd.read_csv(os.path.join(data_dir, 'syntheticdata_10to1.csv'), sep = ',') 