import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_absolute_error, explained_variance_score, mean_squared_error, log_loss
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import VotingClassifier
import itertools

def dashes(n=10): # make stuff prettier
    print('-'*n)


def neg_log_loss(y_true, y_pred):
    return -1.0*log_loss(y_true, y_pred, eps=1e-15, normalize=True)
    

# Cross validation function for classification

def clfCV(X, y, classifier, steps=None, n_splits=5, stratify=True, shuffle=True, random_state=0):
    """
    Returns cross validation scores (averaged over folds)
    Every fold is fed into preprocessing pipeline and regressor separately.
    This ensures there is no leakage between train and test sets.
    
    ---- Parameters -----
    X, Y      : predictos and target, resp
    steps     : steps in preprocessing pipeline, can be None too
    classifier: sklearn classifier object
    """
    
    # list of scorer/metric functions
    # these scorers must be defined such that higher score indicates better performance
    # decision_scorers = [accuracy_score, f1_score, roc_auc_score]
    # prob_scorers = [neg_log_loss]
    decision_scorers = [accuracy_score, f1_score]
    prob_scorers = []
    scorers = decision_scorers + prob_scorers  
    n_scorers = len(scorers)
    
    # stratify folds or not
    if stratify==True:
        KF = StratifiedKFold(n_splits=n_splits,
                             shuffle=shuffle,
                             random_state=random_state)
    else:
        KF = KFold(n_splits=n_splits,
                   shuffle=shuffle,
                   random_state=random_state)
        
    # empty lists to hold fold scores
    train_scores, test_scores = [], []
    
    # cross validation loop
    for train_index, test_index in KF.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # preprocessing pipeline
        if steps is None:
            X_train_mod = X_train
            X_test_mod = X_test
        else:
            pipe = Pipeline(steps)
            X_train_mod = pipe.fit_transform(X_train)
            X_test_mod = pipe.transform(X_test)

        # fit estimator
        classifier.fit(X_train_mod, y_train)

        # store scores 
        train_score = []
        test_score = []
        for scorer in scorers:
            if scorer in decision_scorers:
                train_score.append(scorer(y_train, classifier.predict(X_train_mod)))
                test_score.append(scorer(y_test, classifier.predict(X_test_mod)))
            elif scorer in prob_scorers:
                train_score.append(scorer(y_train, classifier.predict_proba(X_train_mod)))
                test_score.append(scorer(y_test, classifier.predict_proba(X_test_mod)))
            else:
                raise ValueError('Input scorer not found in preset scorer list')

        # store fold scores
        train_scores.append(train_score)
        test_scores.append(test_score)
        

    return np.array(train_scores).mean(axis=0), np.array(test_scores).mean(axis=0)



def rmse_score(y1, y2):
    return np.sqrt(mean_squared_error(y1, y2))


def negative_explained_variance(y1, y2):
    return  -1.0*explained_variance_score(y1, y2)

# Cross validation for regression

def regCV(X, y, regressor, steps=None, n_splits=5, shuffle=True, random_state=0):
    """
    Returns cross validation scores (averaged over folds)
    Every fold is fed into preprocessing pipeline and regressor separately.
    This ensures there is no leakage between train and test sets.
    
    ---- Parameters -----
    X, Y      : predictos and target, resp
    steps     : steps in preprocessing pipeline, can be None too
    regressor : sklearn regressor object
    """
    
    # list of scorer/metric functions
    # these scores must be defined such that lower score indicates better performance
    scorers = [mean_absolute_error, rmse_score, negative_explained_variance]
    n_scorers = len(scorers)
    
    KF = KFold(n_splits=n_splits,
               shuffle=shuffle,
               random_state=random_state)
        
    # empty lists to hold fold scores
    train_scores, test_scores = [], []
    
    # cross validation loop
    for train_index, test_index in KF.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # preprocessing pipeline
        if steps is None:
            X_train_mod = X_train
            X_test_mod = X_test
        else:
            pipe = Pipeline(steps)
            X_train_mod = pipe.fit_transform(X_train)
            X_test_mod = pipe.transform(X_test)

        # fit estimator
        regressor.fit(X_train_mod, y_train)

        # compute scores
        y_train_pred = regressor.predict(X_train_mod)
        y_test_pred = regressor.predict(X_test_mod)
        
        # store scores
        train_score = [f(y_train, y_train_pred) for f in scorers]
        test_score = [f(y_test, y_test_pred) for f in scorers]

        # store fold scores
        train_scores.append(train_score)
        test_scores.append(test_score)
        

    return np.array(train_scores).mean(axis=0), np.array(test_scores).mean(axis=0)



# Grid search function for classification

def clfGridSearch(X, Y, classifier, param_dict, steps=None, n_splits=5, scorer='acc',
                  random_state=0, stratify=True, shuffle=True, verbose=True):
    
    best_score = -np.inf
    
    # the following names must correspond to the scorers in clfCV function 
    possible_scorers = ['acc', 'f1', 'roc', 'logloss'] # make sure to modify output values if modifications of common metrics are used
    to_flip_sign = ['logloss'] 
    scorer_idx = possible_scorers.index(scorer)
        
    # create parameter grid
    n_params = param_dict.keys().__len__()
    all_param_vals = param_dict.values()
    param_grid = [i for i in itertools.product(*all_param_vals)]
    
    # loop over parameter values
    test_scores = [] # will hold test score for each parameter combination
    
    for param_list in param_grid:
        params = {param: param_val
                  for param, param_val
                  in zip(param_dict.keys(), param_list)}
        # find cross-validated score for current parameter combination
        clf = classifier.set_params(**params)
        cv_train_scores, cv_test_scores = clfCV(X=X,
                                                y=Y,
                                                classifier=clf,
                                                steps=steps,
                                                n_splits=n_splits,
                                                stratify=stratify,
                                                shuffle=shuffle,
                                                random_state=random_state)            
        test_score = cv_test_scores[scorer_idx]
        train_score = cv_train_scores[scorer_idx]
        test_scores.append(test_score)
            
        # store score if it improved
        if test_score > best_score:
            best_score = test_score
            corresponding_train_score = train_score
            best_params = params
            
    if scorer in to_flip_sign:
        best_score = -1.0*best_score
        corresponding_train_score = -1.0*corresponding_train_score
        
    if verbose==True:
        # print best test score, and corresponding train score
        print('{0: <16}  {1:.3f}'.format('Max score', best_score))
        print('{0: <16}  {1:.3f}'.format('Train score', corresponding_train_score))
        dashes(25)
        
        # print optimal parameters
        for i, j in best_params.items():
            if type(j)==str:
                print('{0: <16}  {1}'.format(i, j))
            elif type(j)==tuple: # specially for MLPClassifier
                print('{0: <16} {1}'.format(i, j))
            else:
                print('{0: <16}  {1:.2f}'.format(i, j))
    
    return best_params, best_score, corresponding_train_score



def clfGridSearchRobust(X, Y, classifier, param_dict, steps=None, n_splits=5, scorer='acc',
                  random_states=np.arange(100,121,5), stratify=True, shuffle=True, verbose=True):
    # this averages over several random states so that parameter tuning is more robust
    best_score = -np.inf
    
    # the following names must correspond to the scorers in clfCV function 
    possible_scorers = ['acc', 'f1', 'roc', 'logloss'] # make sure to modify output values if modifications of common metrics are used
    to_flip_sign = ['logloss'] 
    scorer_idx = possible_scorers.index(scorer)
        
    # create parameter grid
    n_params = param_dict.keys().__len__()
    all_param_vals = param_dict.values()
    param_grid = [i for i in itertools.product(*all_param_vals)]
    
    # loop over parameter values
    for param_list in param_grid:
        params = {param: param_val
                  for param, param_val
                  in zip(param_dict.keys(), param_list)}
        
        # find cross-validated score for current parameter combination
        # make sure to average over all random states
        cv_train_scores = []
        cv_test_scores = []
        for seed in random_states:
            clf = classifier.set_params(**params)
            current_cv_results = clfCV(X=X,
                                       y=Y,
                                       classifier=clf,
                                       steps=steps,
                                       n_splits=n_splits,
                                       stratify=stratify,
                                       shuffle=shuffle,
                                       random_state=seed)
            cv_train_scores.append(current_cv_results[0][scorer_idx])
            cv_test_scores.append(current_cv_results[1][scorer_idx])

        test_score = np.mean(cv_test_scores)
        train_score = np.mean(cv_train_scores)
            
        # store score if it improved
        if test_score > best_score:
            best_score = test_score
            avg_train_score = train_score
            best_params = params
            
    if scorer in to_flip_sign:
        best_score = -1.0*best_score 
        avg_train_score = -1.0*avg_train_score
        
    if verbose==True:
        # print best test score, and corresponding train score
        print('{0: <16}  {1:.3f}'.format('Max avg score', best_score))
        print('{0: <16}  {1:.3f}'.format('Avg train score', avg_train_score))
        dashes(25)
        
        # print optimal parameters
        for i, j in best_params.items():
            if type(j)==str:
                print('{0: <16}  {1}'.format(i, j))
            elif type(j)==tuple: # specially for MLPClassifier
                print('{0: <16} {1}'.format(i, j))
            else:
                print('{0: <16}  {1:.2f}'.format(i, j))
    
    return best_params, best_score, avg_train_score



# Grid search function for regression

def regGridSearch(X, Y, regressor, param_dict, steps=None, n_splits=5, scorer='rmse',
                  random_state=0, shuffle=True, verbose=True):
    
    best_score = np.inf
    
    # the following names must correspond to the scorers in regCV function 
    possible_scorers = ['mae', 'rmse', 'nev']
    scorer_idx = possible_scorers.index(scorer)
        
    # create parameter grid
    n_params = param_dict.keys().__len__()
    all_param_vals = param_dict.values()
    param_grid = [i for i in itertools.product(*all_param_vals)]
    
    # loop over parameter values
    test_scores = [] # will hold test score for each parameter combination
    
    for param_list in param_grid:
        params = {param: param_val
                  for param, param_val
                  in zip(param_dict.keys(), param_list)}
        # find cross-validated score for current parameter combination
        reg = regressor.set_params(**params)
        cv_train_scores, cv_test_scores = regCV(X=X,
                                                y=Y,
                                                regressor=reg,
                                                steps=steps,
                                                n_splits=n_splits,
                                                shuffle=shuffle,
                                                random_state=random_state)            
        test_score = cv_test_scores[scorer_idx]
        train_score = cv_train_scores[scorer_idx]
        test_scores.append(test_score)
            
        # store score if it improved
        if test_score < best_score:
            best_score = test_score
            corresponding_train_score = train_score
            best_params = params
    
    if verbose==True:
        # print best test score, and corresponding train score
        print('{0: <16}  {1:.3f}'.format('Max score', best_score))
        print('{0: <16}  {1:.3f}'.format('Train score', corresponding_train_score))
        dashes(25)
        
        # print optimal parameters
        for i, j in best_params.items():
            if type(j)==str:
                print('{0: <16}  {1}'.format(i, j))
            elif type(j)==tuple: # specially for MLPClassifier
                print('{0: <16} {1}'.format(i, j))
            else:
                print('{0: <16}  {1:.2f}'.format(i, j))
    
    return best_params, best_score, corresponding_train_score



def regGridSearchRobust(X, Y, regressor, param_dict, steps=None, n_splits=5, scorer='rmse',
                        random_states=np.arange(222, 623,100), shuffle=True, verbose=True):
    
    best_score = np.inf
    
    # the following names must correspond to the scorers in regCV function 
    possible_scorers = ['mae', 'rmse', 'nev']
    scorer_idx = possible_scorers.index(scorer)
        
    # create parameter grid
    n_params = param_dict.keys().__len__()
    all_param_vals = param_dict.values()
    param_grid = [i for i in itertools.product(*all_param_vals)]
    
    # loop over parameter values
     # will hold test score for each parameter combination
    
    for param_list in param_grid:
        params = {param: param_val
                  for param, param_val
                  in zip(param_dict.keys(), param_list)}
        cv_test_scores = []
        cv_train_scores = []
        for seed in random_states:
            # find cross-validated score for current parameter combination
            reg = regressor.set_params(**params)
            current_cv_scores = regCV(X=X,
                                      y=Y,
                                      regressor=reg,
                                      steps=steps,
                                      n_splits=n_splits,
                                      shuffle=shuffle,
                                      random_state=seed)            
            cv_test_scores.append(current_cv_scores[1][scorer_idx])
            cv_train_scores.append(current_cv_scores[0][scorer_idx])

        mean_test_score = np.mean(cv_test_scores)
        mean_train_score = np.mean(cv_train_scores)
        
        # store score if it improved
        if mean_test_score < best_score:
            best_score = mean_test_score
            corresponding_train_score = mean_train_score
            best_params = params
    
    if verbose==True:
        # print best test score, and corresponding train score
        print('{0: <16}  {1:.3f}'.format('Best mean score', best_score))
        print('{0: <16}  {1:.3f}'.format('Mean train score', corresponding_train_score))
        dashes(25)
        
        # print optimal parameters
        for i, j in best_params.items():
            if type(j)==str:
                print('{0: <16}  {1}'.format(i, j))
            elif type(j)==tuple: # specially for MLPClassifier
                print('{0: <16} {1}'.format(i, j))
            else:
                print('{0: <16}  {1:.2f}'.format(i, j))
    
    return best_params, best_score, corresponding_train_score



# Backwards sequential feature selection for classification

def clfBackFeatureSelect(X, Y, classifier, steps=None, scorer='acc', n_splits=5,
                         sig_fig=3, stratify=True, shuffle=True, random_state=0):
    """ performs backwards sequential feature selection """
    
    # define a wrapper cross validation function whose sole argument is the predictor feature set
    def wrapperCV(predictor_set):
        return clfCV(X=predictor_set,
                     y=Y,
                     classifier=classifier,
                     steps=steps,
                     n_splits=n_splits,
                     stratify=stratify,
                     shuffle=shuffle,
                     random_state=random_state)
    
    # the following names must correspond to the scorers in clfCV function 
    possible_scorers = ['acc', 'f1', 'roc', 'logloss']
    scorer_idx = possible_scorers.index(scorer)

    # find cross-validated score for data with all features
    tt, all_features_scores = wrapperCV(X)
    all_features_score = all_features_scores[scorer_idx]

    # best_current_score will be updated through loops over all features
    best_current_score = np.round(all_features_score, sig_fig)
    
    # all features
    features = X.columns.values
    
    
    #  if there is only one feature in X, return X and all_features_scores
    if len(features) == 1:
        return X, all_features_score
    
    # loop over all features and check if removing any improves score
    to_remove_features = []
    to_remove_scores = []
    for feature in features:
        # reduced feature set
        features_reduced = [i for i in features if i != feature]
        # compute CV score
        _, cv_scores = wrapperCV(X[features_reduced])
        current_score = np.round(cv_scores[scorer_idx], sig_fig)

        # compute list of all features whose removal improves score
        if current_score > best_current_score:
            # if removal of a single feature increases score, store it
            to_remove_features = [feature]
            to_remove_scores = [current_score]
            best_current_score = current_score
        elif ((current_score == best_current_score) and (current_score >= all_features_score)):
            # take care of the case when several features satisfy the above if clause
            to_remove_features.append(feature)
            to_remove_scores.append(current_score)
            
    # remove features, and return dataframe
    if (len(to_remove_features) > 1) & (X.shape[1] - len(to_remove_features) >= 1):
        # if to_remove_features contains multiple features,
        # check if removing all of them together improves score
        X_reduced = X.drop(to_remove_features, axis=1)
        _, temp_cv_scores = wrapperCV(X_reduced)
        current_score_temp = np.round(temp_cv_scores[scorer_idx], sig_fig)
            
        if current_score_temp > all_features_scores[scorer_idx]:
            # if score is improved, return the function with reduced predictor set to iterate the process
            return clfBackFeatureSelect(X=X_reduced,
                                        Y=Y,
                                        classifier=classifier,
                                        steps=steps,
                                        scorer=scorer,
                                        n_splits=n_splits,
                                        sig_fig=sig_fig,
                                        stratify=stratify,
                                        shuffle=shuffle,
                                        random_state=random_state)
        else:
            # if score is not improved, return the original predictor set, and the corresponding set
            return X, all_features_scores[scorer_idx]
            
    elif len(to_remove_features) == 1:
        # if to_remove_features contains single features, return the function to iterate
        X_reduced = X.drop(to_remove_features, axis=1)
        
        return clfBackFeatureSelect(X=X_reduced,
                                    Y=Y,
                                    classifier=classifier,
                                    steps=steps,
                                    scorer=scorer,
                                    n_splits=n_splits,
                                    sig_fig=sig_fig,
                                    stratify=stratify,
                                    shuffle=shuffle,
                                    random_state=random_state)
    else:
        # if none of the features improve the score when removed,
        # return the original predictor set and the corresponding score
        return X, all_features_score
    
    
    
def clfBackFeatureSelectRobust(X, Y, classifier, steps=None, scorer='acc', n_splits=5,
                         sig_fig=3, stratify=True, shuffle=True, random_states=np.arange(100,121,5)):
    """ performs backwards sequential feature selection """
    
    # the following names must correspond to the scorers in clfCV function 
    possible_scorers = ['acc', 'f1', 'roc', 'logloss']
    scorer_idx = possible_scorers.index(scorer)
    
    # define a wrapper cross validation function whose sole argument is the predictor feature set
    def avgCVScore(predictor_set, idx, random_seeds):
        seed_scores = []
        for seed in random_seeds:
            seed_score = clfCV(X=predictor_set,
                               y=Y,
                               classifier=classifier,
                               steps=steps,
                               n_splits=n_splits,
                               stratify=stratify,
                               shuffle=shuffle,
                               random_state=seed)
            seed_scores.append(seed_score[1][idx])
    
        return np.mean(seed_scores)

    # find cross-validated score for data with all features
    all_features_score = avgCVScore(X, scorer_idx, random_states)
    
    # best_current_score will be updated through loops over all features
    best_current_score = np.round(all_features_score, sig_fig)
    
    # all features
    features = X.columns.values
    
    
    #  if there is only one feature in X, return X and all_features_scores
    if len(features) == 1:
        return X, all_features_score
    
    # loop over all features and check if removing any improves score
    to_remove_features = []
    to_remove_scores = []
    for feature in features:
        # reduced feature set
        features_reduced = [i for i in features if i != feature]

        # compute score over current feature set
        current_score = np.round(avgCVScore(X[features_reduced], scorer_idx, random_states),
                                 sig_fig)

        # compute list of all features whose removal improves score
        if current_score > best_current_score:
            # if removal of a single feature increases score, store it
            to_remove_features = [feature]
            to_remove_scores = [current_score]
            best_current_score = current_score
        elif ((current_score == best_current_score) and (current_score >= all_features_score)):
            # take care of the case when several features satisfy the above if clause
            to_remove_features.append(feature)
            to_remove_scores.append(current_score)
            
    # remove features, and return dataframe
    if (len(to_remove_features) > 1) & (X.shape[1] - len(to_remove_features) >= 1):
        # if to_remove_features contains multiple features,
        # check if removing all of them together improves score
        X_reduced = X.drop(to_remove_features, axis=1)
        current_score_temp = np.round(avgCVScore(X_reduced, scorer_idx, random_states),
                                      sig_fig)
            
        if current_score_temp > np.round(all_features_score, sig_fig):
            # if score is improved, return the function with reduced predictor set to iterate the process
            return clfBackFeatureSelectRobust(X=X_reduced,
                                              Y=Y,
                                              classifier=classifier,
                                              steps=steps,
                                              scorer=scorer,
                                              n_splits=n_splits,
                                              sig_fig=sig_fig,
                                              stratify=stratify,
                                              shuffle=shuffle,
                                              random_states=random_states)
        else:
            # if score is not improved, return the original predictor set, and the corresponding set
            return X, all_features_score
            
    elif len(to_remove_features) == 1:
        # if to_remove_features contains single features, return the function to iterate
        X_reduced = X.drop(to_remove_features, axis=1)
        
        return clfBackFeatureSelectRobust(X=X_reduced,
                                          Y=Y,
                                          classifier=classifier,
                                          steps=steps,
                                          scorer=scorer,
                                          n_splits=n_splits,
                                          sig_fig=sig_fig,
                                          stratify=stratify,
                                          shuffle=shuffle,
                                          random_states=random_states)
    else:
        # if none of the features improve the score when removed,
        # return the original predictor set and the corresponding score
        return X, all_features_score



# Backwards sequential feature selection for regression

def regBackFeatureSelect(X, Y, regressor, steps=None, scorer='rmse', n_splits=5,
                         sig_fig=3, stratify=True, shuffle=True, random_state=0):
    """ performs backwards sequential feature selection """
        
    # define a wrapper CV function whose sole argument is the predictor feature set
    def wrapperCV(predictor_set):
        return regCV(X=predictor_set,
                     y=Y,
                     regressor=regressor,
                     steps=steps,
                     n_splits=n_splits,
                     shuffle=shuffle,
                     random_state=random_state)
    
    # the following names must correspond to the scorers in regCV function 
    possible_scorers = ['mae', 'rmse', 'nev']
    scorer_idx = possible_scorers.index(scorer)

    # find cross-validated score for data with all features
    _, all_features_scores = wrapperCV(X) 
    # best_current_score will be updated through loops over all features
    all_features_score = all_features_scores[scorer_idx]
    best_current_score = all_features_score
    
    # all features
    features = X.columns.values
    
    # loop over all features and check if removing any improves score
    to_remove_features = []
    to_remove_scores = []
    for feature in features:
        # reduced feature set
        features_reduced = [i for i in features if i != feature]
        # compute CV score
        _, cv_scores = wrapperCV(X[features_reduced])
        current_score = np.round(cv_scores[scorer_idx], sig_fig)

        # compute list of all features whose removal improves score
        if current_score < best_current_score:
            # if removal of a single feature increases score, store it
            to_remove_features = [feature]
            to_remove_scores = [current_score]
            best_current_score = current_score
        elif ((current_score == best_current_score) and (current_score < all_features_score)):
            # take care of the case when several features satisfy the above if clause
            to_remove_features.append(feature)
            to_remove_scores.append(current_score)
            
    # remove features, and return dataframe
    if len(to_remove_features) > 1:
        # if to_remove_features contains multiple features,
        # check if removing all of them together improves score
        X_reduced = X.drop(to_remove_features, axis=1)
        _, temp_cv_scores = wrapperCV(X_reduced)
        current_score_temp = np.round(temp_cv_scores[scorer_idx], sig_fig)
            
        if current_score_temp < all_features_scores[scorer_idx]:
            # if score is improved, return the function with reduced predictor set to iterate the process
            return regBackFeatureSelect(X=X_reduced,
                                        Y=Y,
                                        regressor=regressor,
                                        steps=steps,
                                        scorer=scorer,
                                        n_splits=n_splits,
                                        sig_fig=sig_fig,
                                        shuffle=shuffle,
                                        random_state=random_state)
        else:
            # if score is not improved, return the original predictor set, and the corresponding set
            return X, all_features_scores[scorer_idx]
            
    elif len(to_remove_features) == 1:
        # if to_remove_features contains single features, return the function to iterate
        X_reduced = X.drop(to_remove_features, axis=1)
        
        return regBackFeatureSelect(X=X_reduced,
                                    Y=Y,
                                    regressor=regressor,
                                    steps=steps,
                                    scorer=scorer,
                                    n_splits=n_splits,
                                    sig_fig=sig_fig,
                                    shuffle=shuffle,
                                    random_state=random_state)
    else:
        # if none of the features improve the score when removed,
        # return the original predictor set and the corresponding score
        return X, all_features_score



def clfVoting(X, y, classifiers, classifier_names, steps=None, voting='soft', scorer='acc', n_splits=5, random_state=111):
    """ finds optimal combination of classifiers
    returns the combination with the highest CV score """
    n_classifiers = len(classifier_names)
    
    # the following names must correspond to the scorers in classification_CV function 
    possible_scorers = ['acc', 'f1', 'roc', 'logloss']
    scorer_idx = possible_scorers.index(scorer)
    
    # loop over all possible combinations with atleast two classifiers
    best_score = -np.inf # modifiy best_score to np.inf if the score decreases when model performance improves
    best_combination = []
    
    for i in range(1, n_classifiers):
        for indices in itertools.combinations(range(0, n_classifiers),
                                              i+1):
            clf_list = [classifiers[i] for i in indices]
            clf_names = [classifier_names[i] for i in indices]
            
            # voting
            voting_clf = VotingClassifier(estimators=tuple(zip(clf_names, clf_list)),
                                          voting=voting)
            _, test_score = clfCV(X=X,
                                  y=y,
                                  classifier=voting_clf,
                                  steps=steps,
                                  n_splits=n_splits,
                                  stratify=True,
                                  shuffle=True,
                                  random_state=random_state)
            
            score = test_score[scorer_idx]
            if score > best_score:
                best_score = score
                best_combination = clf_names
                
    return best_combination, best_score    


def clfVotingRobust(X, y, classifiers, classifier_names, steps=None, voting='soft', scorer='acc', n_splits=5, random_states=np.arange(100,121,5)):
    """ finds optimal combination of classifiers
    returns the combination with the highest CV score
    different from clfVoting, as this one averages over several random states """
    n_classifiers = len(classifier_names)
    
    # the following names must correspond to the scorers in classification_CV function 
    possible_scorers = ['acc', 'f1', 'roc', 'logloss']
    scorer_idx = possible_scorers.index(scorer)
    
    # loop over all possible combinations with atleast two classifiers
    best_score = -np.inf # modifiy best_score to np.inf if the score decreases when model performance improves
    best_combination = []
    
    for i in range(1, n_classifiers):
        for indices in itertools.combinations(range(0, n_classifiers),
                                              i+1):
            clf_list = [classifiers[i] for i in indices]
            clf_names = [classifier_names[i] for i in indices]
            
            # voting
            voting_clf = VotingClassifier(estimators=tuple(zip(clf_names, clf_list)),
                                          voting=voting)
            
            # make sure to average over all random states
            test_scores = []
            for seed in random_states:
                _, test_score = clfCV(X=X,
                                      y=y,
                                      classifier=voting_clf,
                                      steps=steps,
                                      n_splits=n_splits,
                                      stratify=True,
                                      shuffle=True,
                                      random_state=seed)
                test_scores.append(test_score[scorer_idx])
            
            # update best_score, if current score is better 
            score = np.mean(test_scores)
            if score > best_score:
                best_score = score
                best_combination = clf_names
                
    return best_combination, best_score    