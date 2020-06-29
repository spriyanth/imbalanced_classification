### Inspired by: https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html#sphx-glr-auto-examples-over-sampling-plot-comparison-over-sampling-py

############### PACKAGES ###############

from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,
                                    KMeansSMOTE)
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, NeighbourhoodCleaningRule
from imblearn.base import BaseSampler

############### FUNCTIONS ###############

def create_dataset(n_samples=1000, weights=(0.01, 0.99), n_classes=2,
                   class_sep=0.8, n_clusters=1):
    return make_classification(n_samples=n_samples, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters,
                               weights=list(weights),
                               class_sep=class_sep, random_state=0)

def plot_resampling(X, y, sampling, ax):
    colors = ['tab:blue', 'tab:red']
    X_res, y_res = sampling.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k', cmap = matplotlib.colors.ListedColormap(colors))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    return Counter(y_res)

def plot_decision_function(X, y, clf, ax):
    colors = ['tab:blue', 'tab:red']
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap = matplotlib.colors.ListedColormap(colors))
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k', cmap = matplotlib.colors.ListedColormap(colors))

class FakeSampler(BaseSampler):

    _sampling_type = 'bypass'

    def _fit_resample(self, X, y):
        return X, y
    
############### PLOT IMPACT OF CLASS DISTRIBUTION ON DECISION RULE OF CLASSIFIER ###############

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

ax_arr = (ax1, ax2, ax3, ax4)
weights_arr = ((0.5, 0.5), (0.33, 0.67),
               (0.1, 0.9), (0.05, 0.95))

alpha = (('1:1'), ('1:2'), ('1:10'), ('1:20'))

for ax, weights, ratio in zip(ax_arr, weights_arr, alpha):
    X, y = create_dataset(n_samples=1000, weights=weights)
    clf = LinearSVC().fit(X, y)
    plot_decision_function(X, y, clf, ax)
    ax.set_title('Class distribution: {}'.format(ratio), fontsize = 20)
fig.tight_layout()

############### ORIGINAL VS. ROS PLOT ###############
ROS = RandomOverSampler

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
X, y = create_dataset(n_samples=1000, weights=(0.1, 0.9))
clf = LinearSVC().fit(X, y)
plot_decision_function(X, y, clf, ax1)
ax1.set_title('Decision function on training data (pre-ROS)', fontsize = 18)
sampler = ROS(sampling_strategy = 'auto', random_state=1234)
clf = make_pipeline(sampler, LinearSVC())
clf.fit(X, y)
plot_resampling(X, y, sampler, ax2)
ax2.set_title('Data transformation using ROS', fontsize = 18)
pipe = make_pipeline(sampler, LinearSVC())
pipe.fit(X, y)
plot_decision_function(X, y, pipe, ax3)
ax3.set_title('Decision function on training data (post-ROS)', fontsize = 18)
fig.tight_layout()

############### ORIGINAL VS. SMOTE PLOT ###############

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
X, y = create_dataset(n_samples=1000, weights=(0.1, 0.9))
clf = LinearSVC().fit(X, y)
plot_decision_function(X, y, clf, ax1)
ax1.set_title('Decision function on training data (pre-SMOTE)', fontsize = 18)
sampler = SMOTE(sampling_strategy = 'auto', k_neighbors = 5, random_state=1234)
clf = make_pipeline(sampler, LinearSVC())
clf.fit(X, y)
plot_resampling(X, y, sampler, ax2)
ax2.set_title('Data transformation using SMOTE', fontsize = 18)
pipe = make_pipeline(sampler, LinearSVC())
pipe.fit(X, y)
plot_decision_function(X, y, pipe, ax3)
ax3.set_title('Decision function on training data (post-SMOTE)', fontsize = 18)
fig.tight_layout()

############### ORIGINAL VS. ADASYN PLOT ###############

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
X, y = create_dataset(n_samples=1000, weights=(0.1, 0.9))
clf = LinearSVC().fit(X, y)
plot_decision_function(X, y, clf, ax1)
ax1.set_title('Decision function on training data (pre-ADASYN)', fontsize = 18)
sampler = ADASYN(sampling_strategy = 'auto', random_state=1234)
clf = make_pipeline(sampler, LinearSVC())
clf.fit(X, y)
plot_resampling(X, y, sampler, ax2)
ax2.set_title('Data transformation using ADASYN', fontsize = 18)
pipe = make_pipeline(sampler, LinearSVC())
pipe.fit(X, y)
plot_decision_function(X, y, pipe, ax3)
ax3.set_title('Decision function on training data (post-ADASYN)', fontsize = 18)
fig.tight_layout()

############### ORIGINAL VS. RUS PLOT ###############

RUS = RandomUnderSampler

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
X, y = create_dataset(n_samples=1000, weights=(0.1, 0.9))
clf = LinearSVC().fit(X, y)
plot_decision_function(X, y, clf, ax1)
ax1.set_title('Decision function on training data (pre-RUS)', fontsize = 18)
sampler = RUS(sampling_strategy = 'auto', random_state=1234)
clf = make_pipeline(sampler, LinearSVC())
clf.fit(X, y)
plot_resampling(X, y, sampler, ax2)
ax2.set_title('Data transformation using RUS', fontsize = 18)
pipe = make_pipeline(sampler, LinearSVC())
pipe.fit(X, y)
plot_decision_function(X, y, pipe, ax3)
ax3.set_title('Decision function on training data (post-RUS)', fontsize = 18)
fig.tight_layout()

############### ORIGINAL VS. ENN PLOT ###############

ENN = EditedNearestNeighbours

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
X, y = create_dataset(n_samples=1000, weights=(0.1, 0.9))
clf = LinearSVC().fit(X, y)
plot_decision_function(X, y, clf, ax1)
ax1.set_title('Decision function on training data (pre-ENN)', fontsize = 18)
sampler = ENN(sampling_strategy = 'auto', n_neighbors = 3)
clf = make_pipeline(sampler, LinearSVC())
clf.fit(X, y)
plot_resampling(X, y, sampler, ax2)
ax2.set_title('Data transformation using ENN', fontsize = 18)
pipe = make_pipeline(sampler, LinearSVC())
pipe.fit(X, y)
plot_decision_function(X, y, pipe, ax3)
ax3.set_title('Decision function on training data (post-ENN)', fontsize = 18)
fig.tight_layout()

############### ORIGINAL VS. NCR PLOT ###############

NCL = NeighbourhoodCleaningRule

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
X, y = create_dataset(n_samples=1000, weights=(0.1, 0.9))
clf = LinearSVC().fit(X, y)
plot_decision_function(X, y, clf, ax1)
ax1.set_title('Decision function on training data (pre-NCL)', fontsize = 18)
sampler = NCL(sampling_strategy = 'auto', threshold_cleaning = 0.5)
clf = make_pipeline(sampler, LinearSVC())
clf.fit(X, y)
plot_resampling(X, y, sampler, ax2)
ax2.set_title('Data transformation using NCL', fontsize = 18)
pipe = make_pipeline(sampler, LinearSVC())
pipe.fit(X, y)
plot_decision_function(X, y, pipe, ax3)
ax3.set_title('Decision function on training data (post-NCL)', fontsize = 18)
fig.tight_layout()