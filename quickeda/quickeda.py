import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import dcor

from sklearn import preprocessing as skl_preproc

from IPython.display import display
from ipywidgets import widgets, interactive

import itertools

np.set_printoptions(precision=2)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


def dashes(n=10): # make stuff prettier
    print('-'*n)


class EDA():
    """
    does basic univariate and bivariate analysis
    also makes plots

    -----Parameters---------------------------------------------------------------------------------------------------


        target_feature: (string)
                        Target feature in the dataframe to be analyzed
        is_target_cat : (boolean)
                        Whether target is categorical or numerical
        alpha_pvalue  : (float, [0,1])
                        Threhold above which the pvalue is considered to be significant, and the null hypothesis is
                        accepted
        norm_test     : (None, 'KS', 'SW', 'custom')
                        If None, no normality tests are performed. 'KS' and 'SW' are Kolmogorov-Smirnov and
                        Shapiro-Wilk resp. For using a custom function, use 'Custom'
        norm_test_func: (function)
                        Only used when norm_test is 'custom'. The function should accept a 1D array-like, and return
                        pvalue
        corr_tol      : (float, [0, 1])
                        defines 'strong correlation'.  If absolute value of correlation between two variables is above
                        corr_tol, they are strongly correlated for the purposes of this analysis
        corr_tol_low  : (float, [0, 1])
                        defines 'weak correlation'.  If absolute value of correlation between two variables is below
                        corr_tol_low, they are weakly correlated for the purposes of this analysis. If distance
                        correlation between two vectors is zero (0) then they are independent. 
        corr_how      : (string)
                        Correlation coefficient to use (e.g. 'spearman' or 'pearson')
        cont_tol      : (float, positive)
                        Used to determine if chi squared test of independence is valid. If value of each cell in
                        expected contingency table is atleast cont_tol, the test is valid.
        mode_thresh   : (float, [0, 100])
                        If a mode has frequency less than mode_thresh, it (and the corresponding column) are not
                        printed
        imbalance_factor: (float)
                        See the function findImbalancedFeatures
        skew_tol      : (float, positive)
                        If skewness of a column is greater than skew_tol, then the column is "too skewed"
        plot_cols     : (int, minimum 1)
                        Number of columns when there are several plots in a single figure arranged in a grid
        class_thresh  : (int, minimum 1)
                        If number of categories in a column is more than this, the column is ignored when creating
                        count plots
        VIF_thresh    : (None or float, non negative)
                        If float, only those features whose VIF (variance inflation factor) is above VIR_tresh are
                        returned
        make_plots    : (boolean)
                        If False, no plots are made
        marker_size   : (int, positive)
                        Size of markers in scatterplots
        marker_alpha  : (float, (0, 1])
                        Alpha value of markers in scatterplots (transparency)
        colorblind    : (boolean)
                        If True, sets searborn palette to colorblind


    -----Methods---------------------------------------------------------------------------------------------------
        missing_values_df_      : Dataframe with number/percent of null values per feature
        doesnot_distinguish_    : Only for binary target, otherwise returns None. Dataframe with p values of
                                    Welch test to test if mean values of numerical features grouped by target
                                    classes are equal. In other words, the features who pass the test do not
                                    distinguish between the two target classes. Also returns None in another case,
                                    for which see function doWelchTest
        too_skewed_             : Dataframe with skewness, excesskurtosis and pvalue (of normality test) for every
                                    feature whose skewness is larger than self.skew_tol
        too_skewed_transf_      : Dataframe with skewness, excess kurtosis and pvalue (of normality test) for every
                                    YJ transformed feature whose skewness is larger than self.skew_tol
        categ_dependency_       : Dataframe containing dependent categorical pairs, and pvalues of independence
                                    tests. Is None if no dependent pairs are found.
        imbalanced_columns_     : Tuple of 1. list of imbalanced columns, and 2. uncommon categories within these
                                    columns
        strongly_corr_with_targ_: Dataframe containing list of numerical features which are strongly correlated
                                    with the target
        weakly_corr_with_targ_  : Dataframe containing list of numerical features which are weakly correlated
                                    with the target
        indep_not_analyzed_     : list of pairs of features whose mutual independence could not be analyzed

    -----To Do---------------------------------------------------------------------------------------------------
    1. Add QQPlots
    2. Add measures of dispersion
    3. Use ANOVA to investigate dependence of numerical targets on categorical features
    4. Fix issues with plots (spacing, legends, etc)
    5. Sort strongly and weakly correlated features dataframe by distance correlation instead of spearman correlation
    
    -------------------------------------------------------------------------------------------------------------
    """

    def __init__(self, target_label, is_target_cat, alpha_pvalue=0.05,
                 norm_test=None, norm_test_func=None, corr_tol=0.60, corr_tol_low=0.10,
                 corr_how='spearman', make_plots=True, cont_tol=1.0,
                 mode_thresh=40.0, imbalance_factor=2, VIF_thresh=5.0,
                 skew_tol=1.0, class_thresh=6, marker_size=15,
                 marker_alpha=0.5, plot_cols=2, figsize=(8, 4), colorblind=True):
        self.target = target_label
        self.alpha = alpha_pvalue
        self.norm_test = norm_test
        self.norm_test_func = norm_test_func
        self.make_plots = make_plots
        self.corr_tol = corr_tol
        self.corr_tol_low = corr_tol_low
        self.corr_how = corr_how
        self.is_target_cat = is_target_cat
        self.make_plots = make_plots
        self.marker_size = marker_size
        self.marker_alpha = marker_alpha
        self.figsize = figsize
        self.cont_tol = cont_tol
        self.mode_thresh = mode_thresh
        self.imbalance_factor = imbalance_factor
        self.skew_tol = skew_tol
        self.plot_cols = plot_cols
        self.class_thresh = class_thresh
        self.colorblind = colorblind
        self.VIF_thresh = VIF_thresh

    def doEDA(self, df):
        # set colorblind palette
        if self.colorblind:
            sns.set_palette('colorblind')

        # main EDA function
        df = df.copy()

        X = df.drop(self.target, axis=1)
        Y = df[self.target].values
        is_target_object = Y.dtype == 'O'

        # plot dimension
        width = self.figsize[0]
        height = self.figsize[1]

        # compute/print number of features and samples
        N = X.shape[0]
        M = X.shape[1]
        print('Number of samples', N)
        print('Number of features', M)

        # separate feature types (categorical and numerical)
        categ_cols = X.dtypes[(X.dtypes == 'O') | (X.dtypes == 'category')].index.values
        num_cols = [col for col in X.columns
                    if col not in categ_cols]
        

        # ---------------------------------------------------------------------------------------------------
        # compute/print number of missing values
        heading('MISSING VALUES')
        self.missing_values_df_ = countNull(df)
        if self.missing_values_df_.empty:
            print('No missing values')
        else:
            print(self.missing_values_df_)

        # UNIVARTIATE ANALYSIS
        # ---------------------------------------------------------------------------------------------------
        heading('TARGET')
        if self.is_target_cat:
            # make count plot to show target category populations
            print('Numbers displayed on bars are counts relative to the least common category')
            plt.figure(figsize=self.figsize)
            g = sns.countplot(x=self.target, data=df)
            # add text that displays relative size of bars
            heights = np.array([p.get_height() for p in g.patches],
                                dtype='float')
            y_offset = 0.05*heights.min() # additive
            heights = 100.0*heights/heights.min() # rescale heights relative to smallest bar
            for i, p in enumerate(g.patches):
                g.text(p.get_x(),
                       p.get_height() + y_offset,
                       '{0:.0f}'.format(heights[i]),
                       ha="left") 
            plt.show()
        else:
            # make distplot for continuous target
            plt.figure(figsize=self.figsize)
            bins = int(len(df[self.target]) / 25) 
            g = sns.distplot(df[self.target],
                             bins=bins)
            plt.show()
        
        
        
        # ---------------------------------------------------------------------------------------------------
        heading('NUMERICAL FEATURES VS TARGET VARIABLE')

        if not is_target_object:
            self.strongly_corr_with_targ_ = stronglyCorrelatedWithTarget(df[num_cols + [self.target]],
                                                                         target=self.target,
                                                                         tol=self.corr_tol,
                                                                         method=self.corr_how)
            if self.strongly_corr_with_targ_.empty:
                print('No features are strongly correlated with target\n')
            else:
                print('Features which are strongly correlated (abs value >= {0}) with target ({1}):\n'.format(self.corr_tol, self.target))
                print(self.strongly_corr_with_targ_)
                print()
                
            self.weakly_corr_with_targ_ = stronglyCorrelatedWithTarget(df[num_cols + [self.target]],
                                                                         target=self.target,
                                                                         tol=self.corr_tol_low,
                                                                         method=self.corr_how,
                                                                         return_weakly_corr=True)
            if self.weakly_corr_with_targ_.empty:
                print('No features are weakly correlated with target\n')
            else:
                print('Features which are weakly correlated (abs value <= {0}) with target ({1}):\n'.format(self.corr_tol_low, self.target))
                print(self.weakly_corr_with_targ_)
                print()

            if not self.is_target_cat:
                if self.make_plots:
                    # make scatterplots of all features against target variable
                    print("Scatterplots of target vs numerical features")
                    to_plot = np.append(num_cols, self.target)
                    self.makeScatterPlots(df[to_plot],
                                          target=self.target,
                                          size=self.figsize)
        if self.make_plots:
            if self.is_target_cat:
                # make bar and violin plots of all features against target variable
                print("Bar\violin plots of target vs numerical features")
                to_plot = np.append(num_cols, self.target)
                self.makeBarAndViolinPlots(df[to_plot],
                                           target=self.target,
                                           size=(2*width, height))
                
        if self.is_target_cat:
            if df[self.target].unique().__len__()==2:
                print('Feature which passed (null hyp.=True) Welch test for difference of means:')
                print(('(Null hypothesis: means of feature values grouped by target variable categories are equal)'))
                print('(Small p values would imply features do distinguish between target variable\'s categories)\n')
                to_search_from = np.append(num_cols, self.target)
                self.doesnot_distinguish_ = doWelchTest(df[to_search_from],
                                                        target=self.target,
                                                        alpha=self.alpha,
                                                        output_only_passed=True)
                if self.doesnot_distinguish_.empty:
                    self.doesnot_distinguish_ = None
                else:
                    print(self.doesnot_distinguish_)
            else:
                self.doesnot_distinguish_ = None
        
        # ---------------------------------------------------------------------------------------------------
        if len(categ_cols) >= 1:
            heading('CATEGORICAL FEATURES VS TARGET VARIABLE')
            to_plot = np.append(categ_cols, self.target)
            if not is_target_object:
                if not self.is_target_cat:
                    if self.make_plots:
                        # make barplots of all features against target variable
                        print("Bar plots of target vs numerical features\n")
                        self.makeBarPlotsForNumTarget(df[to_plot],
                                          target=self.target,
                                          size=self.figsize)
            if self.make_plots:
                if self.is_target_cat:
                    # make contingency tables of all features against target variable
                    print("Contingency tables - target vs categorical features")
                    self.makeContTablesForNumTarget(df[to_plot],
                                                    target=self.target,
                                                    size=self.figsize)

        # ---------------------------------------------------------------------------------------------------
        heading('UNIVARIATE ANALYSIS')

        # print modes with their frequencies
        print('Modes with percent frequencies above {0}'.format(self.mode_thresh))
        print('(Imbalance = ratio of observed and expected frequencies = frequency/(100/n_categories))\n')
        print(findModes(df, thresh=self.mode_thresh))
        print()

        # find imbalanced features
        if self.is_target_cat:
            to_search_from = np.append(categ_cols, self.target)
        else:
            to_search_from = categ_cols

        imbalanced, uncommon = imbalancedFeatures(df[to_search_from],
                                                  thresh_factor=self.imbalance_factor)
        # print imbalanced columns (and uncommon categories), if any
        if imbalanced:
            print('Imbalanced columns:')
            print('(Uncommon categories and resp. percent frequencies)\n')
            for col, clss in zip(imbalanced, uncommon):
                print(col, ':\n', clss)
            print()
        else:
            print('No imbalanced categorical columns\n')
        self.imbalanced_columns_ = (imbalanced, uncommon)

        # make histograms of all numerical variables
        if self.make_plots:
            self.makeHistOrCountPlots(df.drop(self.target, axis=1), size=self.figsize)

        dashes(80)

        # ---------------------------------------------------------------------------------------------------

        # find columns who are too skewed
        # too skewed means skewness is over self.skew_tol
        print('Features which have skewness above {0}\n'.format(self.skew_tol))

        # define normality tests
        if self.norm_test == None:
            normality_test = None
        elif self.norm_test == 'KS':  # KS test
            normality_test = lambda x: stats.kstest(x, 'norm')[1]
            print('(P-values are for Kolmogorov-Smirnov test of normality)\n')
        elif self.norm_test == 'SW':  # SW test
            normality_test = lambda x: stats.shapiro(x)[1]
            print('(P-values are for Shapiro-Wilk test of normality)\n')
        elif self.norm_test == 'custom':
            normality_test = self.norm_test_func
            print('(P-values are for supplied test of normality)\n')
        else:
            normality_test = None
            print('No normality test performed')

        if self.is_target_cat:
            to_search_from = np.append(num_cols, self.target)
        else:
            to_search_from = num_cols

        self.too_skewed_ = tooSkewedFeatures(df[to_search_from],
                                             tol=self.skew_tol,
                                             test=normality_test)
        print(self.too_skewed_)

        # apply Yeo-Johnson transformation on features which are "too skewed"
        print('\nAfter applying Yeo-Johnson transformation\n')
        df_temp = df.copy()  # this will hold the transformed features

        # transformation
        for col in self.too_skewed_.index:
            # ignore null valued elemenets without deleting them
            pt_vals = skl_preproc.power_transform(df[col].dropna().values.reshape(-1, 1),
                                      method='yeo-johnson',
                                      standardize=False)
            df_temp[col][~df_temp[col].isnull()] = pt_vals

        self.too_skewed_transf_ = tooSkewedFeatures(df_temp[self.too_skewed_.index],
                                                    tol=self.skew_tol,
                                                    test=normality_test,
                                                    ignore_unskewed=False)
        print(self.too_skewed_transf_)
        print()

        if self.make_plots:
            self.makeProbPlots(df[self.too_skewed_.index],
                               titles=('Original data', 'YJ transformed data'),
                               size=(2 * width, height))

        # ---------------------------------------------------------------------------------------------------

        # BIVARIATE ANALYSIS
        # only do the following computations if there are atleast two numerical columns
        if len(num_cols) > 1:
            # multicollinearity
            # pairs of variables with high linear correlation (abs value > corr_tol)
            heading('INDEPENDENCE OF NUMERICAL FEATURES')
            corr_pairs = []
            corr_vals = []
            dcorr_vals = []
            
            for pair in itertools.combinations(num_cols, 2):
                # drop null values (for dcor.distance_correlation)
                X_temp = X[list(pair)].dropna()
            
                # compute correlations
                lin_corr = X_temp[list(pair)].corr(method=self.corr_how).iloc[0, 1]
                dcorr = dcor.distance_correlation(X_temp[pair[0]], X_temp[pair[1]])
            
                if (np.abs(lin_corr) >= self.corr_tol) or (dcorr >= self.corr_tol):
                    corr_pairs.append(pair)
                    corr_vals.append(lin_corr)
                    dcorr_vals.append(dcorr)
                

            self.corr_pairs_df_ = pd.DataFrame({'Pair': corr_pairs, 'Linear': corr_vals, 'DistanceCorr': dcorr_vals})
        
            print('Pairwise correlations above {0}\n'.format(self.corr_tol))
            if self.corr_pairs_df_.empty:
                print('No strongly correlated pairs\n')
            else:
                indices = range(1, len(corr_vals) + 1)
                self.corr_pairs_df_ = self.corr_pairs_df_.sort_values(by='DistanceCorr',
                                                                      ascending=False)
                self.corr_pairs_df_.index = indices
                print(self.corr_pairs_df_)
            
        
            # compute variance inflation factors
            vif_values = VIF(df[num_cols],
                             threshold=self.VIF_thresh)
            if vif_values.empty:
                print('\nNo features with variance inflation factors above {0}\n'.format(self.VIF_thresh))
            else:
                print('\nFeatures with variance infation factors above {0}\n'.format(self.VIF_thresh))
                print(vif_values)
                print()
    
        
            # make pairwise scatter plots
            if self.make_plots:
                n_cols = self.plot_cols
                n_rows = max(int(np.ceil((len(num_cols) - 1) / n_cols)),
                             1)
                self.makePairWiseScatterPlots(X[num_cols],
                                              size=(n_cols * width, n_rows * height),
                                              n_rows=n_rows,
                                              n_cols=n_cols)

        # ---------------------------------------------------------------------------------------------------

        # only do the following computations if there are atleast two categorical columns
        if len(categ_cols) > 1:
            # Independence of categorical variables
            heading('INDEPENDENCE OF CATEGORICAL FEATURES')

            # if expected cell value in a contingency table is above self.cont_tol,
            # then the feature pair qualifies for chi squared test
            # otherwise use Fisher's "exact" test
            # (for now Fisher's exact test can only be performed for pairs of binary features)

            test_used = []
            indep_pvalues = []
            self.indep_not_analyzed_ = []
            dependent_categ_pairs = []
            qualified_for_chi2 = []
            qualified_for_fisher = []

            for pair in itertools.combinations(categ_cols, 2):
                X_temp = X[list(pair)].copy()
                # drop null values and contruct contingency table
                X_temp.dropna(inplace=True)
                cont_table = pd.crosstab(X_temp[pair[0]], X_temp[pair[1]]).values
                if cont_table.size > 0:  # removing null values might cause entire categories to vanish
                    validity, is_indep, pval = self.doChi2IndepTest(table=cont_table,
                                                                    tol=self.cont_tol,
                                                                    alpha=self.alpha)
                    if validity:
                        # these pairs qualify for chi squared test
                        qualified_for_chi2.append(pair)
                        if not is_indep:
                            # these pairs are not independent
                            dependent_categ_pairs.append(pair)
                            test_used.append('Chi2')
                            indep_pvalues.append(pval)
                    else:
                        if cont_table.shape == (2, 2):
                            qualified_for_fisher.append(pair)
                            # these pairs qualify for Fisher's exact test
                            # perform the test
                            is_indep, pval = self.doFisherExactTest(table=cont_table,
                                                                    alpha=self.alpha)
                            if not is_indep:
                                # these pairs failed Fisher's test
                                dependent_categ_pairs.append(pair)
                                test_used.append('Fisher')
                                indep_pvalues.append(pval)
                        else:
                            # these pairs qualify for neither test
                            self.indep_not_analyzed_.append(pair)

            # if any pairs were found to be dependent,
            # construct a dataframe with test results
            if not dependent_categ_pairs:
                self.categ_dependency_ = None
            else:
                categ_dependency = {'Pair': dependent_categ_pairs,
                                    'P-value': indep_pvalues,
                                    'Test': test_used}
                self.categ_dependency_ = pd.DataFrame(categ_dependency)
            
            print('Dependent features:')
            print(self.categ_dependency_)
            print()

            # print number of pairs analyzed and (if any) not analyzed 
            n_analyzed = len(qualified_for_fisher) + len(qualified_for_chi2)
            print('%d pairs were checked for independence' % n_analyzed)
            if self.indep_not_analyzed_:
                print('%d pairs could not be checked for independence' % len(self.indep_not_analyzed_))

            # print count plots of all categorical features
            # ignore features with too many categories (more than self.class_thresh)
            if self.is_target_cat:
                to_plot = np.append(categ_cols, self.target)
            else:
                to_plot = categ_cols
            few_classes = [col for col in to_plot
                           if df[col].unique().__len__() <= self.class_thresh]
            n_cols = self.plot_cols
            n_rows = max(int(np.ceil((len(few_classes) - 1) / n_cols)),
                         1)

            if self.make_plots:
                print('\nCount plots of categorical features with less than %d categories:\n' % self.class_thresh)
                self.makeCountPlots(df[few_classes],
                                    size=(n_cols * width, n_rows * height),
                                    n_rows=n_rows,
                                    n_cols=n_cols)
                print('\nContingency tables for categorical features with less than %d categories:\n' % self.class_thresh)
                self.makeContingencyHeatmaps(df[few_classes],
                                             size=(n_cols * width, n_rows * height),
                                             n_rows=n_rows,
                                             n_cols=n_cols)

        # ---------------------------------------------------------------------------------------------------

        return None


    # helper functions

    def makeHistOrCountPlots(self, df, size):
        # makes histograms or countplots depending on the feature
        # histogram if feature is numerical, otherwise boxplot
        # displays histograms as ipython widgets
        # if a column has more categories than self.class_thresh, it is ignored

        def makeHistOrBox(feature):
            # wrapper function to plot a single plot

            # drop null
            values = df[feature].dropna()

            plt.figure(figsize=size)
            if df[feature].dtype == 'O':
                # make boxplot
                if df[feature].unique().__len__() > self.class_thresh:
                    plt.plot(x=0, y=0)
                    plt.gca().set(title='Plotting aborted, too many (>%d) categories' % self.class_thresh)
                else:
                    print('Numbers displayed on bars are their relative (%) heights')
                    g = sns.countplot(x=feature, data=df)
                    # add text that displays relative size of bars
                    heights = np.array([p.get_height() for p in g.patches],
                                       dtype='float')
                    y_offset = 0.05*heights.min() # additive
                    heights = 100.0*heights/heights.min() # rescale heights relative to smallest bar
                    for i, p in enumerate(g.patches):
                        g.text(p.get_x(),
                               p.get_height() + y_offset,
                               '{0:.0f}'.format(heights[i]),
                               ha="left") 
            else:
                # make histogram
                bins = int(len(values) / 25)  # "adaptive" binning
                sns.distplot(values,
                             bins=bins,
                             label=feature)
                hehe = """plt.gca().set(xlabel=feature,
                              ylabel='count')
                plt.legend()"""
            plt.show()

            return None

        # make dropdown menu
        features = df.columns.values
        dropdown = widgets.Dropdown(options=features,
                                    value=features[0],
                                    description='Feature')

        w = interactive(makeHistOrBox, feature=dropdown)
        display(w)

        return None

    def makeScatterPlots(self, df, target, size):
        # makes scatterplots of all features against target variable
        # displays plots as ipython widgets
        # df should contain target

        def makeScatter(feature):
            # wrapper function to plot a single scatterplot

            plt.figure(figsize=size)
            plt.scatter(df[feature], df[target],
                        s=self.marker_size,
                        alpha=self.marker_alpha)
            plt.gca().set(xlabel=feature,
                          ylabel=target)
            plt.show()

            return None

        # make dropdown menu
        features = df.drop(target, axis=1).columns.values
        dropdown = widgets.Dropdown(options=features,
                                    value=features[0],
                                    description='Feature')

        w = interactive(makeScatter, feature=dropdown)
        display(w)

        return None

    def makePairWiseScatterPlots(self, df, size, n_rows, n_cols):
        # makes scatterplots of all possible pairs of features
        # displays plots as ipythonw widgets
        # n_rows and n_cols are number of rows and columns in matplotlib subplot

        def makeScatterPlot(feature):
            # wrapper function to plot scatterplots for each column in 'df' vs 'feature'

            to_plot_against = [col for col in df.columns
                               if col != feature]

            f, ax = plt.subplots(n_rows, n_cols, figsize=size)
            for i, col in enumerate(to_plot_against):
                if n_rows == 1:
                    axes = ax[i % n_cols]
                else:
                    axes = ax[i // n_cols, i % n_cols]

                axes.scatter(df[feature], df[col],
                             s=self.marker_size,
                             alpha=self.marker_alpha)
                axes.set(xlabel=feature,
                         ylabel=col)
            title = feature + ' vs other numeric features'
            plt.suptitle(title)
            plt.show()

        # make dropdown menu
        features = df.columns.values
        dropdown = widgets.Dropdown(options=features,
                                    value=features[0],
                                    description='Feature')

        w = interactive(makeScatterPlot, feature=dropdown)
        display(w)

        return None

    def makeBarAndViolinPlots(self, df, target, size):
        # makes barplots of all features against target variable
        # displays plots as ipython widgets
        # df should contain target
        df = df.copy()
        
        def makeBarAndViolinPlot(feature):
            # wrapper function to plot a single barplot

            # drop null
            df_temp = df[[feature, target]].dropna()
            
            # print a short note on relative bar heights
            print('(Numbers displayed are relative (%) heights of bars)')
            f, ax = plt.subplots(1, 2, figsize=size)
            
            # make barplot
            g = sns.barplot(x=target, y=feature, data=df_temp,
                            ax=ax[0])
            # add text that displays relative size of bars
            heights = np.array([p.get_height() for p in g.patches], dtype='float')
            y_offset = 0.05*heights.min()
            heights = 100.0*heights/heights.min() # rescale heights relative to smallest bar
            for i, p in enumerate(g.patches):
                g.text(p.get_x(),
                       p.get_height() + y_offset,
                       '{0:.0f}'.format(heights[i]),
                       ha="left") 
            g.set(xlabel=target,
                  ylabel=feature)
            
            #make violinplot
            if len(df_temp[target].unique())==2:
                # make a single split "violin" if target if binary
                df_temp[' '] = ''
                sns.violinplot(x=' ', y=feature, hue=target, data=df_temp,
                               ax=ax[1], split=True)
            else:
                # make several "violins" if target isnt binary
                sns.violinplot(x=target, y=feature, data=df_temp,
                               ax=ax[1])
            plt.show()

            return None

        # make dropdown menu
        features = df.drop(target, axis=1).columns.values
        dropdown = widgets.Dropdown(options=features,
                                    value=features[0],
                                    description='Feature')

        w = interactive(makeBarAndViolinPlot, feature=dropdown)
        display(w)

        return None
    
    
    def makeBarPlotsForNumTarget(self, df, target, size):
        # makes barplots of all features against target variable
        # displays plots as ipython widgets
        # df should contain target
        
        # print a short note on relative bar heights
        print('(Numbers displayed are relative (%) heights of bars)')

        def makeBarPlot(feature):
            # wrapper function to plot a single barplot

            # drop null
            values = df[feature].dropna()
            target_adjusted = df[target][~df[feature].isnull()]
            
            # make barplot
            plt.figure(figsize=size)
            if df[feature].unique().__len__() > self.class_thresh:
                plt.plot(x=0, y=0)
                plt.gca().set(title='Plotting aborted, too many (>%d) categories' % self.class_thresh)
            else:
                g = sns.barplot(x=values, y=target_adjusted)
                # add text that displays relative size of bars
                heights = np.array([p.get_height() for p in g.patches],
                                   dtype='float')
                y_offset = 0.05*heights.min()
                heights = 100.0*heights/heights.min() # rescale heights relative to smallest bar
                for i, p in enumerate(g.patches):
                    g.text(p.get_x(),
                           p.get_height() + y_offset,
                           '{0:.0f}'.format(heights[i]),
                           ha="left") 
                g.set(xlabel=feature,
                      ylabel=target)
            plt.show()

            return None

        # make dropdown menu
        features = df.drop(target, axis=1).columns.values
        dropdown = widgets.Dropdown(options=features,
                                    value=features[0],
                                    description='Feature')

        w = interactive(makeBarPlot, feature=dropdown)
        display(w)

        return None
    
    
    def makeContTablesForNumTarget(self, df, target, size):
        # makes contingency tables of all features against target variable, and converts them to heatmaps
        # displays plots as ipython widgets
        # df should contain target
        
        # print a short note on relative cell populations
        print('(Numbers displayed are cell populations relative (%) to the most populated cell)')

        def makeContTable(feature):
            # wrapper function to plot a single contingency table/heatmap

            plt.figure(figsize=size)
            if df[feature].unique().__len__() > 2*self.class_thresh:
                plt.plot(x=0, y=0)
                plt.gca().set(title='Plotting aborted, too many (>%d) categories' % int(2*self.class_thresh))
            else:
                table = pd.crosstab(df[target], df[feature])
                table_max = table.values.max()
                table = np.round(100.0*table/table_max, 0)
                g = sns.heatmap(table,
                                annot=True,
                                cmap='coolwarm',
                                vmin=0,
                                vmax=100,
                                fmt='g')
                g.set(title='Max cell population: %s' % table_max)
            plt.show()

            return None

        # make dropdown menu
        features = df.drop(target, axis=1).columns.values
        dropdown = widgets.Dropdown(options=features,
                                    value=features[0],
                                    description='Feature')

        w = interactive(makeContTable, feature=dropdown)
        display(w)

        return None
    

    def makeCountPlots(self, df, size, n_rows, n_cols):
        # makes countplots of each feature
        # displays plots as ipython widgets
        # n_rows and n_cols are number of rows and columns in matplotlib subplot
        
        def makeCountPlot(feature):
            # wrapper function to plot a single countplot
            
            # print note about bar heights
            print('(Numbers displayed are heights of bars relative (%%) to the most common category in %s)' %feature)
            
            to_plot_against = [col for col in features
                               if col != feature]
            n_cls = df[feature].dropna().unique().__len__()
            
            f, ax = plt.subplots(n_rows, n_cols, figsize=size)
            f.tight_layout()
            for i, col in enumerate(to_plot_against):
                if n_rows == 1:
                    axes = ax[i % n_cols]
                else:
                    axes = ax[i // n_cols, i % n_cols]
                g = sns.countplot(x=feature, data=df,
                                  hue=col,
                                  ax=axes)
                heights = np.array([p.get_height() for p in g.patches],
                                   dtype='float')
                # take care of any nans in heights (they are usually reserved for bars with 0 heights)
                heights = np.nan_to_num(heights)
                for i in range(n_cls):
                    mm = heights[i::n_cls].max()
                    if mm==0:
                        heights[i::n_cls] = 0
                    else:
                        heights[i::n_cls] = 100*heights[i::n_cls]/mm # rescale
                y_offset = 0.05*heights.mean() 
                for i, p in enumerate(g.patches):
                    if np.isnan(p.get_height()):
                        hh = 0
                    else:
                        hh = p.get_height()
                    g.text(p.get_x() + p.get_width()/2.0,
                           y_offset + hh,
                           '{0:.0f}'.format(heights[i]),
                           ha="center") 
            title = feature
            plt.suptitle(title)
            plt.show()

            return None

        # make dropdown menu
        features = df.columns.values
        dropdown = widgets.Dropdown(options=features,
                                    value=features[0],
                                    description='Feature')

        w = interactive(makeCountPlot, feature=dropdown)
        display(w)

        return None
    
    
    def makeContingencyHeatmaps(self, df, size, n_rows, n_cols):
        # makes contingency tables for each feature pair, and converts it to heatmap
        # displays plots as ipython widgets
        # n_rows and n_cols are number of rows and columns in matplotlib subplot
        
        # print note about relative cell populations
        print('(Numbers displayed are cell populations relative (%) to the most populated cell)')
        
        def makeContingencyHeatmap(feature):
            # wrapper function to plot a single countplot
            to_plot_against = [col for col in features
                               if col != feature]
            n_cls = df[feature].dropna().unique().__len__()
            
            f, ax = plt.subplots(n_rows, n_cols, figsize=size)
            f.tight_layout()
            for i, col in enumerate(to_plot_against):                
                # construct contingency table, and normalize it to the most populated cell
                table = pd.crosstab(df[feature], df[col])
                table_max = table.values.max()
                table = np.round(100.0*table/table_max, 0)
                
                # construct heatmap
                if n_rows == 1:
                    axes = ax[i % n_cols]
                else:
                    axes = ax[i // n_cols, i % n_cols]
                g = sns.heatmap(table,
                                annot=True,
                                cmap='coolwarm',
                                vmin=0,
                                vmax=100,
                                ax=axes,
                                fmt='g')
                g.set(title='Max cell population: %s' % table_max)
            title = feature
            plt.suptitle(title)
            plt.show()

            return None

        # make dropdown menu
        features = df.columns.values
        dropdown = widgets.Dropdown(options=features,
                                    value=features[0],
                                    description='Feature')

        w = interactive(makeContingencyHeatmap, feature=dropdown)
        display(w)

        return None

    def makeProbPlots(self, df, titles, size):
        # makes probability plots (ordered sample values vs quantiles) of all columns
        # also adds a histogram
        # displays plots as ipython widgets
        # titles is a tuple, must contain two strings

        def makeProbPlot(feature):
            # wrapper function to plot a single prob plot

            # drop null
            values = df[feature].dropna().values

            # yeo-johnson transformation
            values_pt = skl_preproc.power_transform(values.reshape(-1, 1),
                                        method='yeo-johnson',
                                        standardize=False).flatten()
            values_paired = [values, values_pt]

            for val, title in zip(values_paired, titles):
                # compute various quantities for probability plots
                result = stats.probplot(val)
                x_vals = result[0][0]
                y_vals = result[0][1]
                slope = result[1][0]
                intercept = result[1][1]
                r2 = result[1][2]

                # make prob plot and add a trendline
                f, ax = plt.subplots(1, 2, figsize=size)
                label = 'Fit (R^2 = {0:.3f})'.format(r2)

                ax[0].scatter(x_vals, y_vals,
                              s=self.marker_size,
                              alpha=self.marker_alpha)
                ax[0].plot(x_vals, slope * x_vals + intercept,
                           c='k')
                ax[0].set(xlabel='Theoretical normal quantiles',
                          ylabel='Ordered sample values',
                          title='Probability plot')
                ax[0].legend([feature, label])

                # make histogram for visual aid
                bins = int(len(val) / 25)  # "adaptive" binning
                ax[1].hist(val,
                           bins=bins)
                ax[1].set(xlabel=feature,
                          ylabel='count',
                          title='Histogram')
                plt.suptitle(title)
                plt.show()

            return None

        # make dropdown menu
        features = df.columns.values
        dropdown = widgets.Dropdown(options=features,
                                    value=features[0],
                                    description='Feature')

        w = interactive(makeProbPlot, feature=dropdown)
        display(w)

    def doChi2IndepTest(self, table, tol=1.0, alpha=0.05):
        # performs chi squared test of independence
        # test is valid only if each cell in the expected contingency table has atleast tol value

        # returns tuple
        # where first element is whether the pair (col_1, col_2) qualifies for the test
        # and second element is whether the pair passed the test (regardless of validity)
        # last element is pvalue

        # perform test
        _, pval, _, expected_table = stats.chi2_contingency(table)

        # check if the expected contingency table satisfies the assumption of the test
        is_test_valid = (expected_table < tol).sum() == 0

        if pval < alpha:
            return is_test_valid, True, pval
        else:
            return is_test_valid, False, pval

    def doFisherExactTest(self, table, alpha=0.05):
        # performs Fisher's exact test of independence
        # scipy limitation: contingency table can not be larger than 2x2
        # (meaning both feature should be binary)

        # returns tuple
        # where first element is a boolean, whether the pair (col_1, col_2) passed the test or not
        # and second element is the pvalue

        # perform test
        _, pval, = stats.fisher_exact(table)

        return (pval < alpha, pval)

    
    
# more helper functions

def heading(string, length=80, char='-'):
    # appends characters before and after string, and prints the result
    # useful for headings

    string_length = len(string)
    if (length - string_length) % 2 == 0:
        n_dashes = length - string_length
    else:
        n_dashes = length - string_length + 1
    to_print = '\n' + char * int(n_dashes / 2) + string + char * int(n_dashes / 2) + '\n'
    # print(char*length)
    print(to_print)
    return None


def countNull(df):
    # returns a dataframe with number and percent of null values in every feature
    # output ignores features with zero null values

    df = df.copy()
    N = df.shape[0]

    df_null = df.isnull().sum()[df.isnull().sum() > 0].to_frame('Absolute')
    df_null['Percent'] = df_null.Absolute * 100 / N

    return df_null


def stronglyCorrelatedWithTarget(df, target, tol, method='spearman', return_weakly_corr=False):
    # returns features which have high/low linear/nonlinear correlation with target variable
    # depending on whether return_weakly_corr is False or True resp.
    # "high" means abs(correlation) > tol, "low" means abs(correlation) < tol
    # df must contain target

    strongly_corr = []
    corr_vals = []
    dcorr_vals = []

    cols_to_check = [col for col in df.columns
                     if col != target]
    for col in cols_to_check:
        to_find_corr = df[[col, target]]
        lin_corr = to_find_corr.corr(method=method).iloc[0, 1]
        dcorr = dcor.distance_correlation(df[col], df[target])
        
        if return_weakly_corr:
            if (np.abs(lin_corr) < tol) or (dcorr < tol):
                strongly_corr.append(col)
                corr_vals.append(lin_corr)
                dcorr_vals.append(dcorr)
        else:
            if (np.abs(lin_corr) >= tol) or (dcorr >= tol):
                strongly_corr.append(col)
                corr_vals.append(lin_corr)
                dcorr_vals.append(dcorr)

    strongly_correlated = pd.DataFrame({'Linear': corr_vals, 'DistanceCorr': dcorr_vals},
                                       index=strongly_corr)

    # sort by 'Linear' and return
    if return_weakly_corr:
        return strongly_correlated.sort_values(by=['Linear'], ascending=True)
    else:
        return strongly_correlated.sort_values(by=['Linear'], ascending=False)




def tooSkewedFeatures(df, tol, test, ignore_unskewed=True):
    # returns 1. columns # 2. their respective moments
    # perform the supplied test and also output the 3. pvalue
    # output is a dataframe
    # if ignore_unskewed=True, columns which have skewness below tol are ignored

    # compute skews
    means = []
    stds = []
    skews = []
    kurts = []
    pvals = []

    for col in df.columns:
        # drop null values
        values = df[col].dropna().values

        # perform supplied test
        if test is None:
            pval = None
        else:
            pval = test(values)

        # append
        means.append(values.mean())
        stds.append(values.std())
        skews.append(stats.skew(values))
        kurts.append(stats.kurtosis(values))
        pvals.append(pval)

    # prepare output dataframe
    to_df = {'Mean': means, 'Std dev': stds, 'Skewness': skews, 'Ex kurtosis': kurts, 'P-value': pvals}
    results = pd.DataFrame(to_df, index=df.columns)
                 
    if ignore_unskewed:
        # dont output columns with small skewness
        results = results[results.Skewness >= tol]

    if test is None:
        # if no test was supplied, drop the (fake) pvalue column
        results.drop('P-value', axis=1, inplace=True)

    return results


def findModes(df, thresh):
    # returns modes of each column, along with the percent frequency
    # if the freq (in %) is less than thresh, then the corresponding column is ignored
    # output is a dataframe

    n = int(100 / thresh)  # if a column has more than n modes, it is ignored

    df = df.copy()
    N = df.shape[0]

    all_modes = []
    freqs = []
    to_keep = []
    n_unique_values = []
    imbalances = []

    for col in df.columns:
        modes = df[col].mode().values
        if len(modes) <= n:
            # compute frequency
            freq = (df[col] == modes[0]).sum() * 100 / N
            if freq > thresh:
                # append
                all_modes.append(modes)
                freqs.append(freq)
                to_keep.append(col)
                n_unique_values.append(df[col].unique().__len__())

    # create an 'imbalance' feature
    imbalances = np.array(freqs) / (100.0 / np.array(n_unique_values))
    
    # combine lists into a dataframe and return
    to_df = {'Modes': all_modes,
             'Frequency': freqs,
             'UniqueValuesCount': n_unique_values,
             'Imbalance': imbalances}

    return pd.DataFrame(to_df, index=to_keep)


def imbalancedFeatures(df, thresh_factor=2.0):
    # finds all categories with very low frequency for each column in df
    # very low frequency is defined as being less than expected_freq/thresh_factor
    # where expected_freq is expected frequency assuming all categories are equally represented
    # DO NOT feed numerical columns into this function, the output will be large and useless

    # returns two lists
    # one is a list of imbalanced columns
    # and other is a nested list of categories which have low frequencies

    all_low_counts = []
    imbalanced_cols = []

    for col in df.columns:
        counts = df[col].value_counts() * 100 / df.shape[0]
        unique_vals = counts.index.values
        expected_freq = 100.0 / len(unique_vals)
        freq_thresh = expected_freq / thresh_factor

        low_counts = counts[counts < freq_thresh]

        if not low_counts.empty:
            all_low_counts.append(np.round(low_counts, 2).to_dict())
            imbalanced_cols.append(col)

    return imbalanced_cols, all_low_counts


def VIF(df, threshold=None):
    # returns variance inflation factor for each columns in df
    # df must be purely numerical
    # output is a dataframe
    # if threshold is None, returns all VI factors for all columns
    # otherwise returns only those columns who VIF >= threshold
    
    df_corr = df.corr()
    inverted_df_corr = pd.DataFrame(np.linalg.inv(df_corr.values), index = df_corr.index, columns=df_corr.columns)
    
    to_return = pd.DataFrame({'VIF': np.diag(inverted_df_corr)},
                             index=inverted_df_corr.index)
    
    # take care of large values caused by division by zero, as well as anomalous negative values
    to_return[to_return > 100.0] = np.inf
    to_return[to_return < 0] = np.inf

    if threshold is not None:
         to_return = to_return[to_return.VIF>threshold]
            
    return to_return


def gini(X, method='exact', epsilon=1e-6):
    # return Gini coefficient of inequality
    # input must be non negative, 1-D array-like
    # if some values in X are zero, they are increased to epsilon
    # method = 'matrix' or 'sorting'
    # method used matrix computation, but may be too memory intensive
    # sorting method sorts x first, then uses a simple list comprehension to implement an alternate formula
    # stumbled upon alternate gini formula at: https://github.com/oliviaguest/gini
    # which links to https://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    X = np.array(X, dtype='float')


    if method=='exact':
        X = np.array(X).reshape(-1,1)
        X[X==0] += epsilon
        mu = X.mean()
        return np.abs(X - X.swapaxes(0,1)).mean()/(2*mu)
    elif method=='sorting':
        X = np.array(sorted(X))
        X[X==0] += epsilon
        n = len(X)
        mu = X.mean()
        return np.sum([(2*i - n - 1)*x for i, x in enumerate(X)])/(mu*n**2)
    

def doWelchTest(df, target, alpha, output_only_passed=True):
    # performs T test for the hypothesis that means of feature grouped by classes of target are identical
    # target must be binary
    # feature must be numeric
    # returns pvalue
    # and whether or not the hypothesis was rejected , based on the provided p-value threshold (alpha)
    # output_only_passed=True means only features who passed test will be in the output
    
    # drop null values
    pvals=[]
    results = []
    df = df.copy()
    
    features = [col for col in df.columns if col != target]
    for col in features:
        df_temp = df[[target, col]].dropna()
    
    # create samples
        cls1, cls2 = df_temp[target].unique()
        s1 = df_temp[df_temp[target]==cls1][col].values
        s2 = df_temp[df_temp[target]==cls2][col].values

    # perform the test
        pval = stats.ttest_ind(s1, s2, equal_var=False).pvalue
        results.append(pval < alpha) # True if hypothesis was accepted
        pvals.append(pval)
        
    # create output dataframe
    to_return = pd.DataFrame({'P-value': pvals, 'Passed': results},
                             index=features)
    
    if output_only_passed:
        to_return = to_return[~to_return.Passed].drop('Passed', axis=1)
    else:
        to_return = to_return.drop('Passed', axis=1)
    
    return to_return