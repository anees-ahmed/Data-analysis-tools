# Data analysis tools

I. custom_transformers/preprocessing.py
contains preprocessing transformers. These transformers were written to help me with some common transformations I use in analyses. The input/ouput are pandas dataframes.
	1. LowFreqCombiner: renames all classes whose sample frequency is lower than certain threshold to 'Other'. This can significantly speed up computation when there are a large number of categories, and depending on the problem may not even hurt predictive power
	2. OHE: pd.get_dummies in transformer form
	3. Scaler: combines standard scaler, minmax scaler and robust scaler (all familiar transformers from sklearn). The real use is that it also provides a "binary" correction, where non-binary columns are rescaled by 0.5 after a Standard scaling, so that the standard deviations of binary and non-binary columns end up being comparable.
	3. KNNImputer: imputes missing data using KNeighbors algorithm. Uses sklearn.neighbors.KNeighborsClassifier if the column to be imputed is categorical, and sklearn.KNeighborsRegressor otherwise.
	4. PowerTransformer: applies transformation (such log, boxcox, yeo-johnson) to "unskew" to numerical columns. Can selectively apply transformations if a skewness threshold is provided


II. model_evaluation/evaluation.py
provides common model training functions such as cross-validation and parameter tuning via grid search. Each of these functions applies preprocessing transformers on the train set and test set separately in every fold and iteration, and thus eliminate train-test leakage. At the moment these functions don't accept arbitrary sklearn metrics (coming soon). Some common metrics are already provided, rest can be added manually. See comments in the file for more details. Four functions and their variants are available:
	1. CV: cross validation
	2. GridSearch: performs exhaustive grid search over provided parameter space to find optimal parameter combination
	3. BackFeatureSelect:  sequential backwards feature selection. Only for the brave, as sequential feature selection in general causes extreme overfitting
	4. Voting: performs exhaustive grid search over all combinations of (provided) models to find optimal voting ensemble

The variants are named adding suffixes/prefixes to the above function names. Prefixes 'clf' and 'reg' imply the functions will work for classification and regression problems, respectively. Suffix 'Robust' implies the function averages, within each iteration, over several random states


III. quickeda/quickeda.py
produces several plots and common statistical descriptions of the entire dataset. Some minor bugs that need fixing, but otherwise totally usable (as long as iPython is available)


TO DO:
1. Make notation uniform
2. Add more comments
3. Add option to use iPlotly plots instead of iPython widgets for interactive plots, as iPlotly is more widely supported (this will take time)