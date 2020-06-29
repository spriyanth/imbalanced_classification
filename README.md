# Supplementary Codes of the Master Thesis "Binary Classification on Imbalanced Datasets"

*Please change the directory and install the relevant packages before running the code.

## 0_datasets
- Relevant datasets (synthetic data and breast cancer) of the thesis.
- Disclaimer: I do not own any rights for the "Breast Cancer Wisconsin (Diagnostic) Dataset".
- Orginal download link: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

## 1_preprocessing
- Clear instructions about the data preprocessing methodology
- Feature selection based on: ANOVA, correlation.
- Data transformations (natural logarithm, square root, cube root)
- Normality tests
- Synthetic data generation
- Visualization of densities pre- and post-data-transformation

## 2_preliminary_study
- Model selection of the preliminary study
- Evaluation of various configurations for each classifier
- Cross validation and test performance
- Cross validation box-and-whisker-plots

## 3A_main_study_base_optimization
- Baseline model optimization (of bagged SVMs and XGB) through multi-random-search with customized function.

## 3B_main_study_sampling_optimization
- Optimization of sampling algorithms via grid search.
- Tuned parameters: Class distribution (alpha), number of neighbours.
- Considered sampling algorithms: Random Oversampling, Synthetic Minority Oversampling Technique, Adaptive Synthetic Sampling, Random Undersampling, Edited Nearest Neighbours, Neighbour Cleaning Rule.

## 3C_main_study_CSL_optimization
- Optimization of costs for weighted bagged SVMs and weighted XGB via multi-random-search. (can also be done via grid search)
- Comparison of results: heuristic weights vs. optimized costs

## 3D_main_study_results
- Cross validation results of sampling and CSL implementation
- Cross validation box-and-whisker-plots
- Test performance with default paramters
- Test performance with optimized parameters
- Test performance with repeated experiments for non-deterministic sampling algorithms. (Results based on a single seed are not representative due to the variance across seeds. Hence, 100 experiments with different seeds are conducted and the mean performance result therefrom is considered as the final statistic.)

## 4_main_study_hybrid_models
- Implementation of hybrid models: Underbagging (with Decision Trees), Underbagging (with SVMs), EasyEnsemble (original specification with AdaBoost), EasyEnsemble (with XGB => also called xEnsemble in some papers.)
- Cross validation results and box-and-whisker-plots
- Test performance

## 5_additional_plots
- 2 (inspired/adapted) codes for additional plots (visualization of the implementation of sampling algorithms and CSL) in the theory section.
