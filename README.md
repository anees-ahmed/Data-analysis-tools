# Data analysis tools
Some tools for data analysis 

## I. custom_transformers/preprocessing.py
contains preprocessing transformers. These transformers were written to help me with some common transformations I use in analyses. The input/ouput are pandas dataframes, which I find are more readable than numpy arrays.
* **LowFreqCombiner**: renames all classes whose sample frequency is lower than certain threshold to 'Other', or another string of choice. This can significantly speed up computation when there are a large number of relatively unimportant categories, and depending on the problem may not even hurt predictive power.
* **OHE**: standard pandas one hot encoding function [pd.get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html) in transformer form
* **Scaler**: offers the familiar scikit-learn transformers [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) and [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html). The real use is that it also provides a "binary" correction, where non-binary columns are rescaled by 0.5 after a standard scaling (i.e. mean and std.dev are set to 0 and 1 respectively), so that the standard deviations of binary and non-binary columns end up being comparable.
* **KNNImputer**: imputes missing data using KNeighbors algorithm. Uses scikit-learn's [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) if the column to be imputed is categorical, and [KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) otherwise.
* **PowerTransformer**: offers transformations such log, boxcox and yeo-johnson to "unskew" numerical columns. Can selectively apply transformations if a skewness threshold is provided


## II. model_evaluation/evaluation.py
provides common model training functions such as cross-validation and parameter tuning via grid search. Each of these functions applies preprocessing transformers on the train set and test set separately in every fold and iteration, and thus eliminate train-test leakage. At the moment these functions don't accept arbitrary sklearn metrics (coming soon). Some common metrics are already provided, rest can be added manually as functions of two numpy arrays. See comments in the file for more details.

Four functions and their variants are available
* **CV**: k-fold cross validation
* **GridSearch**: performs exhaustive grid search over provided parameter space to find optimal parameter combination
* **BackFeatureSelect**:  sequential backwards feature selection. Only for the brave, as sequential feature selection in general causes extreme overfitting
* **Voting**: performs exhaustive search over all combinations of the provided models to find the optimal voting ensemble

All variants of these four functions are named by adding suffixes/prefixes to their names.
* Prefixes 'clf' and 'reg' imply the functions will work for classification and regression problems, respectively.
* Suffix 'Robust' implies the function averages, within each iteration, over several random states.


## III. quickeda/quickeda.py
produces several plots and common statistical descriptions of the entire dataset. Some minor bugs that need fixing, but otherwise totally usable (as long as iPython is available)


## Planned updates:
1. Make notation uniform
2. Add more comments
3. Add option to use iPlotly plots instead of iPython widgets for interactive plots, as iPlotly is more widely supported (this will take time)